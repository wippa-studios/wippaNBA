"""Model-side code.

Historically this repository had no models package -- the trainers lived under
``src/Train-Models`` (a hyphenated directory, so it is loaded by file path
rather than imported) and the predictors under ``src/Predict``. This package
holds models that are neither: things that are imported as ordinary Python and
are expected to have their own tests.

``src/Models/ratings.py`` is the first entry. It is a rating-systems baseline
(Elo, Bradley-Terry, SRS) that every feature-based model in this repository has
to beat.
"""