import warnings

import pandas as pd

warnings.filterwarnings('ignore')


def main():
    """
    Data sourced from Midwestern Regional Climate Center's online data portal (https://mrcc.purdue.edu/CLIMATE/)

    Data pulled on 2/22/25.

    Data cleaning steps:
    - Remove headers and footers to isolate main dataset
    - Drop less-useful columns
    - Subset to start and end of full years of data
    - Update column names
    - Replace T (trace of precip) with 0.01
    - Replace M (missing indicator) with 0.00
    """
    df = pd.read_csv('data/raw/SD_b2dates.csv', skiprows=7, skipfooter=22)
    df = df.drop(labels=['Unnamed: 11', 'HDD1', 'CDD1', 'GDD2', 'MGDD3'], axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.loc[(df['Date'] >= '1889-01-01') & (df['Date'] <= '2024-12-31')]
    df.columns = ['date', 'precip', 'snow', 'snow_depth', 'max_temp', 'min_temp', 'mean_temp']
    df = df.replace('T', 0.01)
    df = df.replace('M', 0.00)
    df.to_csv('data/clean/kc_weather_clean.csv', index=False)


if __name__ == "__main__":
    main()
