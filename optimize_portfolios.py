from scipy.optimize import minimize
from dataclasses import dataclass
from functools import partial

from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from tqdm import tqdm

from typing import *


import numpy as np
import pandas as pd
import json


@dataclass
class Result:
    w: np.array
    func: np.float64


def sharpe_objective(W, df_returns: pd.DataFrame) -> float:
    sigma_p = np.sqrt(W @ df_returns.cov() @ W.T)
    R = df_returns.mean(axis=0)

    return -np.dot(W, R) / sigma_p


def weights_constr(W):
    return W.sum() - 1


def optimize_portfolio(df_returns: pd.DataFrame) -> Result:
    # select all columns (bonds) containing no NaNs
    all_cols: List[str] = df_returns.columns
    cols: List[str] = df_returns.columns[df_returns.isnull().any() == False].tolist()
    na_cols: List[str] = list(set(all_cols) - set(cols))

    constrs = [{"type": "eq", "fun": weights_constr}]

    x0 = [1 / len(cols)] * len(cols)  # start from equally weighted portfolio
    bounds = [(0, 1)] * len(cols)  # no shorting bonds

    sol = minimize(
        fun=partial(
            sharpe_objective, df_returns=df_returns[cols]
        ),  # Returns to risk ratio
        x0=x0,
        constraints=constrs,
        bounds=bounds,
    )

    weights = []
    w: Iterator = iter(sol.x)

    for col in all_cols:
        weights.append(0 if col in na_cols else next(w))

    return Result(w=np.array(weights), func=-sol.fun)


def hold_balanced_portfolio(df_test: pd.DataFrame, res: Result) -> np.float64:
    """Calculate performance of the weights on df_test. Returns 1 + return"""
    W: Dict[str, np.float64] = res.w
    test_returns = (1 + df_test).prod().values
    return W @ test_returns.T


def backtest_portfolio(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """
    Optimizes the portfolio using optimize_portfolio function and then runs hold_balanced_portfolio
    on the test set. This function is created to be run in multiprocess Pool so everything is done
    within one function which is easy to map
    """
    res: Result = optimize_portfolio(df_returns=df_train)
    pnl: np.float64 = hold_balanced_portfolio(df_test=df_test, res=res)

    return {
        "train_start": df_train.index[0],
        "train_end": df_train.index[-1],
        "test_start": df_test.index[0],
        "test_end": df_test.index[-1],
        "backtest_pnl": pnl,
        "weights": res.w.tolist(),
    }


def optimize_multiprocess(
    df_returns: pd.DataFrame,  # returns data
    window_days: int = 90,
    test_size: int = 14,
    n_workers: int = 5,
) -> List[dict]:

    window_range = range(0, df_returns.shape[0] - window_days - test_size, test_size)

    with (
        tqdm(
            total=len(window_range),
            desc=f"Optimizing portfolios with n_workers {n_workers}:",
        ) as pbar,
        Pool(processes=n_workers) as pool,
    ):
        results = []

        for i in window_range:
            # It is not quite aggregation by days, but pandas doesn't have a good function
            # for grouping by time with intervals like in polars group_by_dynamic
            # So I implemented it kinda wrong but okay
            df_train, df_test = (
                df_returns.iloc[i : i + window_days],
                df_returns.iloc[i + window_days : i + window_days + test_size],
            )

            res: AsyncResult = pool.apply_async(
                partial(backtest_portfolio, df_train=df_train, df_test=df_test)
            )
            results.append(res)

        done_results: List[dict] = []

        for res in results:
            # Once result is obtained increment pbar
            res.get()
            done_results.append(res.get())

            pbar.update(1)

    return done_results


def main() -> int:
    # load returns dataframe from csv file
    df_returns: pd.DataFrame = pd.read_csv("seminar_data/returns.csv")
    df_returns.index = df_returns["Trade date"]
    df_returns = df_returns.drop(columns=["Trade date"])

    res: List[dict] = optimize_multiprocess(
        df_returns=df_returns,
        # run training, weights optimization using Covariance matrix and expected returns calculated over
        # 90 days
        window_days=180,
        # Use 14 day window to estimate performance of the optimal portfolio
        test_size=30,
        # run using 10 processes, parallelize window computations,
        n_workers=5,
    )

    with open("backtest.json", "w") as file:
        json.dump({"res": res}, file)


if __name__ == "__main__":
    main()
