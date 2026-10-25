import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import toml
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import (
    create_todays_games_from_odds,
    get_json_data,
    to_data_frame,
    get_todays_games_json,
    create_todays_games,
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.toml"

# Only one segment of the scores feed is season-dependent, so it is the only
# part templated here. The stats request is NOT templated here either: it is
# built from config.toml's data_url, the same template Get_Data.py formats, so
# there is exactly one copy of that URL in the repo instead of one per caller.
TODAYS_GAMES_URL_TEMPLATE = "https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{scores_year}/scores/00_todays_scores.json"

# Last-resort copy of the data_url template, reached only when config.toml is
# absent or has no data_url key. It carries no season of its own: every year in
# it is a placeholder. Placeholders are {0}=DateTo month, {1}=DateTo day,
# {2}=DateFrom year, {3}=DateTo year, {4}=Season.
FALLBACK_DATA_URL_TEMPLATE = "https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=10%2F01%2F{2}&DateTo={0}%2F{1}%2F{3}&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={4}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="

# Rest days for a team with no earlier game in the schedule, matching the
# Get_Odds_Data default so live and training rows agree on the fallback.
DEFAULT_DAYS_REST = 7


def load_config():
    if not CONFIG_PATH.exists():
        return {}
    return toml.load(CONFIG_PATH)


def season_start_year(season):
    """The season's first year: '1999-00' -> '1999'.

    Both season-named artefacts in this repo are named after the first year, not
    the last: the data.nba.com feed lives under .../nba/<start year>/ and the
    schedule is Data/nba-<start year>-UTC.csv. Getting that backwards is the
    easiest way to have everything look configured and point at last season
    anyway.
    """
    head = str(season).split("-")[0].strip()
    return head if len(head) == 4 and head.isdigit() else ""


def newest_configured_season(config):
    """The latest season key in config.toml, chosen by its start year.

    The fallback when [prediction] does not name a season. Season rollover
    already means editing the season blocks, so deriving from them means an
    untouched [prediction] still follows the season forward instead of silently
    predicting last year's teams.
    """
    keys = list(config.get("get-data", {})) or list(config.get("create-games", {}))
    dated = [key for key in keys if season_start_year(key)]
    if not dated:
        return ""
    return max(dated, key=lambda key: (season_start_year(key), key))


def resolve_snapshot_date(today):
    """DateTo for the live snapshot: YESTERDAY, never today.

    stats.nba.com reads DateFrom and DateTo as INCLUSIVE, so a snapshot asked
    for through today carries the results of games that already finished today.
    A team that has tipped off finds its own W/L, GP, W_PCT, PTS and all 27
    *_RANK columns inside the feature vector the model is about to predict: on
    a snapshot cut today, the rule "did this team's W go up between D and D-1"
    recovers the game's own outcome 96.11% of the time. Yesterday is the most
    recent date complete for every team, and it is the same convention
    Create_Games.py uses when it joins a game onto the newest snapshot strictly
    before the game date.

    `today` is passed in rather than read here so that one clock read dates the
    whole snapshot; a run that straddles midnight must not query two different
    days' worth of season state.
    """
    return today - timedelta(days=1)


def resolve_live_settings(config=None, today=None):
    """Season-dependent URLs and the schedule path, read from config.toml.

    Season rollover is a config edit, not a code edit. These three used to be
    literals in this file, which is how a stale season produced a 404 on the
    scores feed, a full and plausible frame of the wrong season from the stats
    endpoint, and a FileNotFoundError on the schedule CSV. Every value has a
    fallback so a missing key degrades to what config still knows instead of
    raising at startup.
    """
    if config is None:
        config = load_config()
    if today is None:
        today = datetime.now(timezone.utc).date()
    prediction = config.get("prediction", {})

    season = str(prediction.get("season") or newest_configured_season(config))
    scores_year = str(prediction.get("scores_year") or season_start_year(season))
    schedule_file = prediction.get("schedule_file") or f"Data/nba-{scores_year}-UTC.csv"
    snapshot_date = resolve_snapshot_date(today)

    # Same format call Get_Data.py makes, in the same placeholder order:
    # month, day, season start year, year, season.
    data_url = (config.get("data_url") or FALLBACK_DATA_URL_TEMPLATE).format(
        snapshot_date.month,
        snapshot_date.day,
        season_start_year(season),
        snapshot_date.year,
        season,
    )

    return {
        "season": season,
        "snapshot_date": snapshot_date,
        "data_url": data_url,
        "todays_games_url": TODAYS_GAMES_URL_TEMPLATE.format(scores_year=scores_year),
        # Anchored to the repo, not the CWD: every other module here resolves
        # paths off __file__, and main.py was the one that only ran from the
        # repo root.
        "schedule_path": BASE_DIR / schedule_file,
    }


# The live settings, resolved once at import. These names are part of the
# contract scripts/check_feature_alignment.py imports, and they are derived from
# config.toml rather than being literals kept in sync with it by hand.
LIVE_SETTINGS = resolve_live_settings()
DATA_URL_TEMPLATE = load_config().get("data_url") or FALLBACK_DATA_URL_TEMPLATE
TODAYS_GAMES_URL = LIVE_SETTINGS["todays_games_url"]
DATA_URL = LIVE_SETTINGS["data_url"]
SCHEDULE_PATH = LIVE_SETTINGS["schedule_path"]


def resolve_game_day(schedule_df, games):
    """The UTC date the predicted games are played on.

    Taken from the schedule rather than the machine clock. The CSV is UTC and
    datetime.today() is a naive LOCAL time-of-day: comparing the two is what
    makes the same game score 2 days of rest for a runner in Los Angeles and 1
    for a runner in Kolkata. The date on which the most of tonight's teams play
    wins; ties go to the next such date that has not happened yet, and only
    then to the latest, so a single-game slate whose teams met earlier in the
    season cannot resolve to a matchup weeks gone.
    """
    teams = {team for game in games for team in game}
    involved = pd.Series(False, index=schedule_df.index)
    for team in teams:
        involved |= (schedule_df['Home Team'] == team) | (schedule_df['Away Team'] == team)
    if not involved.any():
        return datetime.now(timezone.utc).date()

    dates_by_team = involved.groupby(schedule_df['Date'].dt.date).sum()
    candidates = dates_by_team[dates_by_team == dates_by_team.max()].index.sort_values()
    upcoming = [day for day in candidates if day >= datetime.now(timezone.utc).date()]
    return upcoming[0] if upcoming else candidates[-1]


def days_off_since_last_game(schedule_df, team, today):
    """UTC days between `today` and the team's most recent EARLIER game.

    Strictly earlier on both sides. The game being predicted is scheduled for
    `today`, so including it makes every team look like it is playing a
    back-to-back and collapses its rest to 1 -- the `timedelta(days=1)` that
    used to paper over this was compensating for that off-by-one, not adding a
    rest day. A team with no earlier game keeps the 7-day default.
    """
    team_games = schedule_df[
        (schedule_df['Home Team'] == team) | (schedule_df['Away Team'] == team)
    ]
    prior_dates = team_games.loc[team_games['Date'].dt.date < today, 'Date'].dt.date
    if prior_dates.empty:
        return DEFAULT_DAYS_REST
    return (today - prior_dates.max()).days


def create_todays_games_data(games, df, odds, schedule_df, today):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    for game in games:
        home_team, away_team = game
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        if odds:
            game_key = f"{home_team}:{away_team}"
            game_odds = odds[game_key]
            todays_games_uo.append(game_odds['under_over_odds'])
            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])
        else:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))
            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))

        # calculate days rest for both teams
        home_days_off = days_off_since_last_game(schedule_df, home_team, today)
        away_days_off = days_off_since_last_game(schedule_df, away_team, today)
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off
        stats['Days-Rest-Away'] = away_days_off
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def load_schedule(schedule_path=SCHEDULE_PATH):
    return pd.read_csv(schedule_path, parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')


def resolve_games(odds, sportsbook, todays_games_url=TODAYS_GAMES_URL):
    if odds:
        games = create_todays_games_from_odds(odds)
        if len(games) == 0:
            print("No games found.")
            return None, None
        game_key = f"{games[0][0]}:{games[0][1]}"
        if game_key not in odds:
            print(game_key)
            print(
                Fore.RED,
                "--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------",
            )
            print(Style.RESET_ALL)
            return games, None
        print(f"------------------{sportsbook} odds data------------------")
        for game_key in odds.keys():
            home_team, away_team = game_key.split(":")
            print(
                f"{away_team} ({odds[game_key][away_team]['money_line_odds']}) @ "
                f"{home_team} ({odds[game_key][home_team]['money_line_odds']})"
            )
        return games, odds

    games_json = get_todays_games_json(todays_games_url)
    return create_todays_games(games_json), None


def run_models(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args):
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(
            data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc
        )
        print("-------------------------------------------------------")
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        # The runner applies the saved train-fitted scaler. main.py must not
        # pre-scale: a second, differently fitted transform is the defect this
        # path is fixing (issue #16).
        NN_Runner.nn_runner(
            frame_ml, todays_games_uo, games, home_team_odds, away_team_odds, args.kc
        )
        print("-------------------------------------------------------")


def main(args):
    settings = resolve_live_settings()
    odds = None
    if args.odds:
        odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
    games, odds = resolve_games(odds, args.odds, settings["todays_games_url"])
    if games is None:
        return

    stats_json = get_json_data(settings["data_url"])
    df = to_data_frame(stats_json)
    schedule_df = load_schedule(settings["schedule_path"])
    today = resolve_game_day(schedule_df, games)
    # A wrong season is the one misconfiguration that produces output looking
    # completely normal, so name the season and the snapshot it was built from
    # before any prediction is printed.
    print(
        f"Season {settings['season']}, stats snapshot through "
        f"{settings['snapshot_date']} (yesterday)."
    )
    print(f"Predicting {len(games)} games for {today}.")
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = create_todays_games_data(
        games, df, odds, schedule_df, today
    )

    if args.A:
        args.xgb = True
        args.nn = True

    run_models(
        data,
        todays_games_uo,
        frame_ml,
        games,
        home_team_odds,
        away_team_odds,
        args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    main(args)
