import os
import warnings
import pandas as pd
import numpy as np
import ruptures as rpt
import seaborn as sns
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def split_list(lst, chunk_size=10):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def main(variable):
    df = pd.read_csv('data/clean/kc_weather_analyze_full.csv')
    df = df.sort_values(by=['date'], ascending=True)
    years = list(df['year'].unique())[::-1]
    chunks = split_list(years)
    chunks = chunks[::-1]
    results = []
    for count, chunk in enumerate(chunks):
        if count > 0:
            prequel = chunks[count - 1][:2]
            changepoint_df = df.loc[df['year'].isin(prequel + chunk)]
            changepoint_df['prequel'] = np.where(
                changepoint_df['year'].isin(prequel),
                1,
                0
            )
            changepoint_df = changepoint_df.reset_index(drop=True)
            srs = changepoint_df[variable]
            algo = rpt.Pelt(model="rbf").fit(srs.values)
            result = algo.predict(pen=np.log(len(changepoint_df)))
            changepoint_df['changepoint'] = np.where(
                changepoint_df.index.isin(result),
                1,
                0
            )
            changepoint_df = changepoint_df.loc[changepoint_df['year'].isin(chunk)]
            results.append(changepoint_df)

    changepoint_df = pd.concat(results, axis=0)
    changepoint_df = changepoint_df.reset_index(drop=True)
    changepoint_df = changepoint_df[['date', variable, 'changepoint']]
    changepoint_df['date'] = pd.to_datetime(changepoint_df['date'])
    changepoint_df['year'] = changepoint_df['date'].dt.year
    changepoint_df['month'] = changepoint_df['date'].dt.month
    changepoint_df['week'] = changepoint_df['date'].dt.isocalendar()['week']

    changepoint_summary = pd.DataFrame(changepoint_df.groupby(['year', 'week'])['changepoint'].sum())
    changepoint_summary = changepoint_summary.reset_index(drop=False)
    changepoint_summary = changepoint_summary.loc[changepoint_summary['changepoint'] >= 1]

    sns.catplot(data=changepoint_summary.tail(40), x="year", y="week", jitter=False)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'modeling/changepoint/plots/{variable}_changepoint.png')
    plt.clf()

    grouped = pd.DataFrame(changepoint_summary.groupby('year')['week'].mean())
    grouped = grouped.reset_index(drop=False)
    print(grouped)
    print()


if __name__ == "__main__":
    if not os.path.exists('modeling/changepoint/plots'):
        os.makedirs('modeling/changepoint/plots')

    main(
        variable='min_temp'
    )

    main(
        variable='max_temp'
    )
