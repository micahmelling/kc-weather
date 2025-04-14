import os
import warnings
import numpy as np
import pandas as pd

from explore.explore import make_ts_plot


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def main(variable, rolling_n, anomaly_trigger):
    df = pd.read_csv('data/clean/kc_weather_analyze_full.csv')
    df = df.sort_values(by=['date'], ascending=True)
    df = df[['date', variable]]
    df['rolling_mean'] = df[variable].rolling(rolling_n).mean()
    df['diff'] = abs(df['rolling_mean'] - df[variable])
    df['anomaly'] = np.where(
        df['diff'] >= anomaly_trigger,
        1,
        0
    )

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar()['week']

    anomaly_summary = pd.DataFrame(df.groupby(['year', 'week'])['anomaly'].sum())
    anomaly_summary = anomaly_summary.reset_index(drop=False)
    anomaly_summary['expected_anomalies'] = anomaly_summary.groupby('week')['anomaly'].transform(lambda x: x.expanding().mean())
    anomaly_summary['diff_from_expected'] = anomaly_summary['expected_anomalies'] - anomaly_summary['anomaly']

    grouped = pd.DataFrame(anomaly_summary.groupby('year')['diff_from_expected'].sum())
    grouped = grouped.reset_index(drop=False)
    make_ts_plot(grouped, 'year', 'diff_from_expected',
                 title=f'{variable}_{rolling_n}_{anomaly_trigger} anomaly',
                 save_directory='modeling/anomaly_detection')


if __name__ == "__main__":
    if not os.path.exists('modeling/anomaly_detection/plots'):
        os.makedirs('modeling/anomaly_detection/plots')

    main(
        variable='min_temp',
        rolling_n=7,
        anomaly_trigger=10
    )

    main(
        variable='max_temp',
        rolling_n=7,
        anomaly_trigger=10
    )
