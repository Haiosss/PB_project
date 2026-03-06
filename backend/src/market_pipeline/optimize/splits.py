from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def walkforward_splits(
    d0: date,
    d1: date,
    train_months: int,
    test_months: int,
    step_months: int | None = None,
) -> list[Fold]:

    #rolling splits inside [d0, d1).
    #each fold: train and test

    if train_months <= 0 or test_months <= 0:
        raise ValueError("train_months and test_months must be > 0")

    if step_months is None:
        step_months = test_months

    t = pd.Timestamp(d0)
    end = pd.Timestamp(d1)

    folds: list[Fold] = []
    while True:
        train_end = t + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        folds.append(
            Fold(
                train_start=t.date(),
                train_end=train_end.date(),
                test_start=train_end.date(),
                test_end=test_end.date(),
            )
        )

        t = t + pd.DateOffset(months=step_months)

    return folds