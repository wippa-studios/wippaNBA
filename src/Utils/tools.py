import re
from datetime import datetime

import pandas as pd
import requests

from .Dictionaries import team_index_current

games_header = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/57.0.2987.133 Safari/537.36',
    'Dnt': '1',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en',
    'origin': 'http://stats.nba.com',
    'Referer': 'https://github.com'
}

data_headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Priority": "u=3, i",
    "Referer": "https://www.nba.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0.1 Safari/605.1.15"
}

# Bound on every outbound stats.nba.com/data.nba.com call.
REQUEST_TIMEOUT_SECONDS = 60


class DataFetchError(RuntimeError):
    """Raised when an upstream NBA endpoint cannot be read."""


def get_json_data(url):
    try:
        raw_data = requests.get(url, headers=data_headers, timeout=REQUEST_TIMEOUT_SECONDS)
        raw_data.raise_for_status()
        result_sets = raw_data.json().get('resultSets')
    except Exception as e:
        raise DataFetchError(f"failed to read stats response from {url}") from e
    if not result_sets:
        raise DataFetchError(f"stats response from {url} contained no resultSets")
    return result_sets


def get_todays_games_json(url):
    try:
        raw_data = requests.get(url, headers=games_header, timeout=REQUEST_TIMEOUT_SECONDS)
        raw_data.raise_for_status()
        games = raw_data.json().get('gs')
    except Exception as e:
        raise DataFetchError(f"failed to read games response from {url}") from e
    if not isinstance(games, dict) or 'g' not in games:
        raise DataFetchError(f"games response from {url} is missing the 'gs.g' payload")
    return games['g']


def to_data_frame(data):
    if not isinstance(data, (list, tuple)) or not data:
        raise DataFetchError("expected a non-empty resultSets list")
    data_list = data[0]
    if not isinstance(data_list, dict):
        raise DataFetchError("expected the first resultSet to be an object")
    if 'headers' not in data_list or 'rowSet' not in data_list:
        raise DataFetchError("resultSet is missing headers or rowSet")
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))


def create_todays_games(input_list):
    games = []
    for game in input_list:
        home = game.get('h')
        away = game.get('v')
        home_team = home.get('tc') + ' ' + home.get('tn')
        away_team = away.get('tc') + ' ' + away.get('tn')
        games.append([home_team, away_team])
    return games


def create_todays_games_from_odds(input_dict):
    games = []
    for game in input_dict.keys():
        home_team, away_team = game.split(":")
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        games.append([home_team, away_team])
    return games


def get_date(date_string):
    year1, month, day = re.search(r'(\d+)-\d+-(\d\d)(\d\d)', date_string).groups()
    year = year1 if int(month) > 8 else int(year1) + 1
    return datetime.strptime(f"{year}-{month}-{day}", '%Y-%m-%d')
