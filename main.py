#!/usr/bin/env python3
# Copyright 2016 Grzegorz Milka grzegorzmilka@gmail.com
"""This script is used for analysing performance data of Polish investment
funds and finding optimal portfolio.

The script defines functions for:
* data preprocessing, where the output is a table with monthly returns.
* performing optimization calculations, i.e. maximing expected return given
  risk and additional constraints.
* plotting risk vs return curve and portfolio selection stackplot.
"""
import datetime
import functools
import os
import os.path

from cvxopt import matrix, spmatrix
from cvxopt.blas import dot
from cvxopt.solvers import options, qp
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_DIRECTORY = 'data/mstfun/'
MONTHS_PER_YEAR = 12
#  See clean_portfolio
CLEAN_PORTFOLIO_THRESHOLD = 0.98


# Functions for loading and preprocessing data
def read_mst_file_with_records(filename):
    """Reads 'filename' MST file with financial record. Returns a DataFrame with
    date and close price columns ('date', 'price')."""
    data = pd.read_csv(filename, usecols=['<DTYYYYMMDD>', '<CLOSE>'],
                       parse_dates=[0], infer_datetime_format=True)
    data.columns = ['date', 'price']
    data.price = data.price.astype('float64')

    return data


def calculate_difference_in_months(date_fst, date_snd):
    """Calculate by how many months do two days differ.

    >>> calculate_difference_in_months('2016-06-12', '2016-01-30')
    5
    """
    years = date_fst.year - date_snd.year
    return years * MONTHS_PER_YEAR + date_fst.month - date_snd.month


def is_next_month(date_fst, date_snd):
    """Is 'date_fst' in the month after 'date_snd'?"""
    return calculate_difference_in_months(date_fst, date_snd) == 1


def is_same_month(date_fst, date_snd):
    """Is 'date_fst' in the same month as 'date_snd'?"""
    return calculate_difference_in_months(date_fst, date_snd) == 0


def generate_months_between(date_fst, date_snd):
    """Returns a generator of months between 'date_snd' and 'date_fst'.

    >>> generate_months_between(datetime.date(2016, 12, 12),
                                datetime.date(2016, 9, 10))
    [datetime.date(2016, 10, 1), datetime.date(2016, 11, 1)]
    """
    date_fst = datetime.date(date_fst.year, date_fst.month, 1)
    date_snd = datetime.date(date_snd.year, date_snd.month, 1)

    while date_snd < date_fst:
        if date_snd.month == MONTHS_PER_YEAR:
            date_snd = datetime.date(date_snd.year + 1, 1, 1)
        else:
            date_snd = datetime.date(date_snd.year, date_snd.month + 1, 1)
        if date_snd < date_fst:
            yield date_snd


def get_one_per_month(data):
    """Undersamples records data by getting price from the first available
    record of each month. If there is a missing month, then data for the first
    available month is provided.

    Arguments:
        data - DataFrame with ('date', 'price') columns, where 'date' is at
            daily granularity.

    Returns:
        DataFrame with columns ('date', 'price'). 'date' contains each month
        exactly once."""
    monthly_records = []
    last_month = None
    for row in data.iterrows():
        row = row[1]
        if last_month is None or is_next_month(row.date, last_month.date):
            row.date = pd.to_datetime(datetime.date(
                row.date.year, row.date.month, 1))
            monthly_records.append(row)
            last_month = row
        else:
            if not is_same_month(row.date, last_month.date):
                assert row.date > last_month.date, \
                        'The data is not chronological'
                for month in generate_months_between(row.date, last_month.date):
                    row = last_month.copy(deep=True)
                    row.date = pd.to_datetime(month)
                    monthly_records.append(row)
                    last_month = row
    undersampled_data = pd.DataFrame(monthly_records, columns=['date', 'price'])
    return undersampled_data


def calculate_monthly_change(data):
    """Calculates the monthly change in stock price.

    Args:
        data - DataFrame with ('date', 'price') columns, where 'date' is at
        monthly granularity.

    Returns:
        DataFrame with columns ('date', 'change'). 'date' contains each month
        exactly once and the day part is set to be the first.
    """
    monthly_records = []
    last_month = None
    for row in data.iterrows():
        row = row[1]
        if last_month is None and row.price == 0:
            continue

        if last_month is not None:
            assert last_month.price != 0
            monthly_records.append((last_month.date,
                                    row.price / last_month.price))
        last_month = row

    new_data = pd.DataFrame(monthly_records, columns=['date', 'change'])
    return new_data


def load_and_preprocess_data():
    """Loads and preprocesses performance records.

    Returns:
        DataFrame with monthly price changes. The index is the names of funds.
        The columns are months."""
    stock_data = pd.DataFrame(columns=['name', 'date', 'change'])

    # Load monthly change data into tidy DataFrame
    for mst_file in os.listdir(INPUT_DIRECTORY):
        data = read_mst_file_with_records(
            os.path.join(INPUT_DIRECTORY, mst_file))
        data = get_one_per_month(data)
        data = calculate_monthly_change(data)
        stock_name = os.path.basename(mst_file)[:-4]
        name_column = pd.Series(len(data) * [stock_name], name='name',
                                index=data.index)
        data['name'] = name_column
        stock_data = stock_data.append(data, ignore_index=True)

    stock_data = stock_data.pivot(index='name', columns='date', values='change')
    stock_data = stock_data[sorted(stock_data.columns)]

    # Remove stocks that do not have data for last month
    last_col = stock_data.columns[-1]
    idx_to_drop = stock_data.index[stock_data[last_col].isnull()]
    stock_data = stock_data.drop(idx_to_drop)
    return stock_data


# Functions for portfolio analysis
def filter_to_time_range(data, month_count):
    """
    Args:
        data - DataFrame with monthly price changes. The columns represent
            consecutive months.
        month_count - Number of months in the output DataFrame.
    Returns:
        DataFrame created from 'data' with only 'month_count' last months.
    """
    data = data.drop(data.index[data[data.columns[-month_count]].isnull()])
    return data[data.columns[-month_count:]]


def expected_return(data):
    """
    Args:
        data - DataFrame with monthly price changes. The columns represent
            consecutive months.
    Returns:
        DataFrame with the same index as 'data' and with one column. The column
        contains the average yearly return of the security.
    """
    def m(x, y):
        return x * y
    year_count = data.shape[1] / MONTHS_PER_YEAR
    return (data.apply(lambda row: functools.reduce(m, row), 1, reduce=True)
            .apply(lambda n: n ** (1.0 / year_count)))


def covariance(data):
    """Calculates a covariance matrix from matrix of price changes per period.

    Args:
        data A DataFrame with periodical price changes. Securities are presented
        row-wise and periods column-wise.

    Returns:
        A DataFrame representing the covariance matrix. Row and column indexes
        contain names of securities from data's index."""
    return data.transpose().cov()


def calculate_sample_portfolios(returns, covariances, constraints=([], []),
                                portfolio_count=10):
    """Calculate sample portfolios that maximize return for given risk.

    Args:
        returns - DataFrame with expected yearly return of each security.
        covariances - 2D DataFrame representing the security covariance matrix.
        constraints - A 2-tuple (at_least, at_most) where at_* is a list of
            2-tuples ([sec_name], percent) specifying how much of given
            securities can be in given portfolio.
        portfolio_count - how many portfolios should be generated. More
            portfolios means better risk density.

    Returns:
        A tuple (portfolios, returns, risks)
        portofolios - A list of portfolios. Portfolio is a DataFrame such that
            i-th element is a number from 0.0 to 1.0 indicating the share of
            given security in the portfolio.
        returns, risks - Lists of 'portfolio_count' length containing the
            expected return and risk of corresponding portfolio.
    """
    sec_count = len(returns)
    S = matrix(covariances.values)
    pbar = matrix(returns.values)

    constraint_count = len(constraints[0]) + len(constraints[1])
    G = spmatrix(-1.0, range(sec_count), range(sec_count),
                 (sec_count + constraint_count, sec_count))
    h = matrix(0.0, (sec_count + constraint_count, 1))

    constraint_idx = sec_count
    # At least constraint
    for constraint in constraints[0]:
        for sec_name in constraint[0]:
            sec_idx = returns.index.get_loc(sec_name)
            G[constraint_idx, sec_idx] = -1.0
        h[constraint_idx] = -constraint[1]
        constraint_idx += 1

    # At most constraint
    for constraint in constraints[1]:
        for sec_name in constraint[0]:
            sec_idx = returns.index.get_loc(sec_name)
            G[constraint_idx, sec_idx] = 1.0
        h[constraint_idx] = constraint[1]
        constraint_idx += 1

    A = matrix(1.0, (1, sec_count))
    b = matrix(1.0)

    # mu is used to scale the covariance matrix. The scale influences how
    # risk-averse will be the found portfolio.
    N = portfolio_count
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    options['show_progress'] = False
    portfolios = [qp(mu * S, -pbar, G, h, A, b)['x'] for mu in mus]
    p_returns, p_risks = zip(*[(dot(pbar, p), dot(p, S * p))
                               for p in portfolios])
    portfolios = [pd.DataFrame(np.array(p), index=returns.index,
                               columns=['share']) for p in portfolios]
    return portfolios, list(p_returns), list(p_risks)


def clean_portfolio(portfolio, threshold=CLEAN_PORTFOLIO_THRESHOLD):
    """
    Returns:
        A portfolio such that its securities were responsible for threshold part
        of the 'portfolio' provided as an argument.
    """
    assert threshold > 0.0
    assert threshold <= 1.0
    sorted_portfolio = portfolio.sort_values(by='share', ascending=False)
    idx = 0
    found_sum = 1.0
    for i, val in enumerate(sorted_portfolio.cumsum().values):
        if val > threshold:
            idx = i
            found_sum = val
            break

    significant_securities = sorted_portfolio.index[:(idx + 1)]
    portfolio = portfolio.copy(deep=True)
    portfolio.loc[significant_securities] *= 1.0 / found_sum
    portfolio.loc[~portfolio.index.isin(significant_securities)] = 0.0
    return portfolio


def find_similar_subsequences(values, residuum):
    """
    Finds subsequences in 'values' such that numbers in each subsequence do not
    differ by more than 'residuum'.

    Args:
        values - Sorted list of floats.
        residuum - A float.
    Returns:
        List of indexes pointing to beginnings of found subsequences.
    """
    if not values:
        return []

    leaders = []
    leader = 0
    for i, v in enumerate(values):
        if abs(values[leader] - v) > residuum:
            leaders.append(leader)
            leader = i
    if not leaders or leaders[-1] != leader:
        leaders.append(leader)
    return leaders


# Plotting functions
def plot_risk_vs_return(variance, returns, title_aux=""):
    """Plots a curve with variance on OX and expected return on OY.

    Args:
        variance - array with portfolio variances for which expected return is
            calculated.
        returns - array with the same size as 'variance' containing expected
            return for corresponding variance.
        title_aux - string with additional information to be added to the title.
    """
    plt.figure(facecolor='w')
    plt.plot(variance, returns)
    plt.xlabel('risk')
    plt.ylabel('expected return')
    plt.title('Risk-return trade-off curve{0}'.format('(' + title_aux + ')' if
                                                      title_aux else ''))


def plot_stackplot(portfolios, index, title_aux='', x_label='Index'):
    securities = set()
    for p in portfolios:
        securities = securities.union(p.index)

    per_security_share = []
    for s in securities:
        share = np.zeros(len(portfolios))
        for i in range(len(portfolios)):
            if s in portfolios[i].index:
                share[i] = portfolios[i].loc[s]
        per_security_share.append(share)

    plt.figure(facecolor='w')
    ax = plt.gca()
    polys = ax.stackplot(index, np.row_stack(per_security_share))
    ax.set_xlabel(x_label)
    ax.set_ylabel('Share')
    ax.set_xlim((min(index), max(index)))
    ax.set_ylim((0.0, 1.0))
    plt.title('Sample of portfolios{0}'.format("(" + title_aux + ")" if
                                               title_aux else ""))

    legendProxies = []
    for poly in polys:
        legendProxies.append(Rectangle((0, 0), 1, 1,
                                       fc=poly.get_facecolor()[0]))

    legendProxies.reverse()
    securities = list(reversed(list(securities)))
    plt.legend(legendProxies, securities, loc='upper left')


# Helper functions that wrap it all up
def process_monthly_data(fund_monthly_returns, constraints=([], []),
                         portfolio_count=512, return_density=0.001):
    """
    Finds optimal portfolios given preprocessed data.

    Returns:
        A tuple (portfolios, returns, risks).
    """
    fund_yearly_returns = expected_return(fund_monthly_returns)
    fund_covariance = covariance(fund_monthly_returns)

    portfolios, p_returns, p_risks = calculate_sample_portfolios(
        fund_yearly_returns, fund_covariance, constraints=constraints,
        portfolio_count=portfolio_count)
    subsequences = find_similar_subsequences(p_returns, return_density)

    clean_portfolios = [clean_portfolio(p) for p in portfolios]
    clean_portfolios = [p[p.share > 0.0] for i, p in enumerate(clean_portfolios)
                        if i in subsequences]
    p_returns = [r for i, r in enumerate(p_returns) if i in subsequences]
    p_risks = [r for i, r in enumerate(p_risks) if i in subsequences]

    return (clean_portfolios, p_returns, p_risks)


def generate_constraints(index):
    """
    This is a configuration function that given index of available funds defines
    constraints for my preferred strategy.
    """
    at_least = []

    # at least 70% in ING funds
    ing_names = index[index.str.startswith('ING')]
    if list(ing_names.values):
        at_least.append((ing_names, 0.7))

    # at least 5% into ING gotówkowy
    if 'ING004' in ing_names:
        at_least.append((['ING004'], 0.05))

    # at least 10% into ING gotówkowy or obligacji
    names = []
    if 'ING004' in ing_names:
        names.append('ING004')
    if 'ING005' in ing_names:
        names.append('ING005')
    if names:
        at_least.append((names, 0.1))

    # at least 23% into AIG MOŚ
    if 'AIG014' in index:
        at_least.append((['AIG014'], 0.23))

    at_most = []

    # Do not invest in these funds
    no_invest = []
    for name in ['DWS027', 'DWS031', 'PKO026', 'PKO909']:
        if name in index:
            no_invest.append(name)
    if no_invest:
        at_most.append((no_invest, 0.0))

    # at most 33% in ING Globalny Spółek Dywidendowych
    if 'ING025' in ing_names:
        at_most.append((['ING025'], 0.33))

    return (at_least, at_most)


def main_processing(all_monthly_returns, plot=True):
    # only_ing = all_monthly_returns.index.str.startswith('ING')
    # all_monthly_returns = all_monthly_returns[only_ing]

    month_ranges = [36]
    return_data = {}
    for m in month_ranges:
        print("Analysing data from the last {0} months.".format(m))
        trimmed_returns = filter_to_time_range(all_monthly_returns,
                                               month_count=m)
        constraints = generate_constraints(trimmed_returns.index)
        (portfolios, returns, risks) = process_monthly_data(
            trimmed_returns, constraints=constraints)
        return_data[m] = (portfolios, returns, risks)
        if plot:
            title = '{0} months'.format(m)
            plot_risk_vs_return(risks, returns, title)
            plot_stackplot(portfolios, returns, title, 'Expected return')
    if plot:
        plt.show()
    return return_data


def main():
    all_monthly_returns = load_and_preprocess_data()
    main_processing(all_monthly_returns)


if __name__ == '__main__':
    main()
