import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'


def create_year_summary(group, column):
    return pd.Series({
        'mean': group[column].mean(),
        'median': group[column].median(),
        'std': group[column].std(),
        'sum': group[column].sum(),
        'count': group[column].count(),
        'q5': group[column].quantile(q=0.05),
        'q10': group[column].quantile(q=0.10),
        'q25': group[column].quantile(q=0.25),
        'q75': group[column].quantile(q=0.75),
        'q90': group[column].quantile(q=0.90),
        'q95': group[column].quantile(q=0.95),
    })


# TODO: make more configurable
def main(year_splits=15):
    """
    Generate variables for month, year, quarter, and season.
    Generate variables to identify extreme temperatures.
    Create a variable that chunks years into n segments.
    """
    df = pd.read_csv('data/clean/kc_weather_clean.csv')
    df['date'] = pd.to_datetime(df['date'])

    df['range_temp'] = df['max_temp'] - df['min_temp']
    df['extreme_low_min'] = np.where(
        df['min_temp'] <= 15,
        1,
        0
    )
    df['extreme_low_max'] = np.where(
        df['max_temp'] <= 15,
        1,
        0
    )
    df['extreme_high'] = np.where(
        df['max_temp'] >= 95,
        1,
        0
    )

    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['season'] = df['date'].apply(get_season)
    df['month'] = df['date'].dt.month

    year_splits = np.array_split(list(df['year'].unique()), year_splits)
    df['year_segment'] = np.nan
    for split in year_splits:
        df['year_segment'] = np.where(
            df['year'].isin(split),
            f'{split[0]}-{split[-1]}',
            df['year_segment']
        )

    cols = ['precip', 'snow', 'snow_depth', 'max_temp', 'min_temp', 'mean_temp', 'range_temp', 'extreme_low_min',
            'extreme_low_max', 'extreme_high']
    for col in cols:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=7).mean()
        df[f'{col}_rolling_median'] = df[col].rolling(window=7).median()
        df[f'{col}_rolling_sum'] = df[col].rolling(window=7).sum()

    df.to_csv('data/clean/kc_weather_analyze_full.csv', index=False)

    analyze_cols_base = ['precip', 'snow', 'snow_depth', 'max_temp', 'min_temp', 'mean_temp', 'range_temp',
                         'extreme_low_min', 'extreme_low_max', 'extreme_high']
    df_cols = list(df)
    analyze_cols = []
    for col in df_cols:
        for base_col in analyze_cols_base:
            if base_col in col:
                analyze_cols.append(col)
                break

    group_cols = ['year', 'year_segment']
    subgroup_cols = ['quarter', 'season', 'month']
    
    summaries = []
    for analyze_col in analyze_cols:
        for group_col in group_cols:
            group_df = df.groupby(group_col).apply(lambda y: create_year_summary(y, analyze_col)).reset_index()
            group_df['feature'] = analyze_col
            group_df['group'] = group_col
            group_df = group_df.rename(columns={group_col: 'value'})
            group_df = group_df[[col for col in group_df.columns if col != 'value'] + ['value']]
            summaries.append(group_df)
            
            for subgroup_col in subgroup_cols:
                subgroup_df = df.groupby([group_col, subgroup_col]).apply(
                    lambda y: create_year_summary(y, analyze_col)).reset_index()
                subgroup_df['feature'] = analyze_col
                subgroup_df['group'] = group_col
                subgroup_df['sub_group'] = subgroup_col
                subgroup_df = subgroup_df.rename(columns={group_col: 'value'})
                subgroup_df = subgroup_df.rename(columns={subgroup_col: 'sub_value'})
                subgroup_df = subgroup_df[[col for col in subgroup_df.columns if col != 'value'] + ['value']]
                subgroup_df = subgroup_df[[col for col in subgroup_df.columns if col != 'sub_value'] + ['sub_value']]
                summaries.append(subgroup_df)

    summary_df = pd.concat(summaries, axis=0)
    summary_df = summary_df.reset_index(drop=True)
    summary_df = summary_df.round(3)
    summary_df.to_csv('data/clean/kc_weather_analyze_summary.csv', index=False)


if __name__ == "__main__":
    main()
