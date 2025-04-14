import os
from copy import deepcopy
import pandas as pd
import numpy as np
import operator

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from explore.explore import make_ts_plot


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def run_kmeans_clustering(df, drop_list,  max_clusters):
    append_df = deepcopy(df)
    cluster_df = append_df.drop(drop_list, axis=1)
    cluster_df = pd.DataFrame(StandardScaler().fit_transform(cluster_df), columns=list(cluster_df))
    silhouette_dict = {}
    n_clusters = list(np.arange(2, max_clusters + 1, 1))
    for n in n_clusters:
        kmeans = KMeans(n_clusters=n, random_state=19)
        labels = kmeans.fit_predict(cluster_df)
        silhouette_mean = silhouette_score(cluster_df, labels)
        silhouette_dict[n] = silhouette_mean
    best_n = max(silhouette_dict.items(), key=operator.itemgetter(1))[0]
    kmeans = KMeans(n_clusters=best_n, random_state=19)
    labels = kmeans.fit_predict(cluster_df)
    append_df['cluster'] = labels
    return append_df


def get_cluster_summary(df, cluster_column_name):
    mean_df = df.groupby(cluster_column_name).mean().reset_index()
    sum_df = df.groupby(cluster_column_name).sum().reset_index()
    count_df = df.groupby(cluster_column_name).count().reset_index()
    mean_df = pd.melt(mean_df, id_vars=[cluster_column_name])
    mean_df.rename(columns={'value': 'mean'}, inplace=True)
    sum_df = pd.melt(sum_df, id_vars=[cluster_column_name])
    sum_df.rename(columns={'value': 'sum'}, inplace=True)
    count_df = pd.melt(count_df, id_vars=[cluster_column_name])
    count_df.rename(columns={'value': 'count'}, inplace=True)
    summary_df = pd.merge(mean_df, sum_df, how='inner', on=['cluster', 'variable'])
    summary_df = pd.merge(summary_df, count_df, how='inner', on=['cluster', 'variable'])
    return summary_df


def main(year_col, season, feature):
    df = pd.read_csv('data/clean/kc_weather_analyze_summary.csv')

    df = df.loc[df['group'] == year_col]
    df = df.loc[df['sub_value'] == season]
    df = df.loc[df['feature'] == feature]

    drop_list = ['feature', 'group', 'value', 'sub_group', 'sub_value', 'sum', 'count']
    cluster_df = run_kmeans_clustering(
        df,
        drop_list=drop_list,
        max_clusters=5
    )
    cluster_summary_df = get_cluster_summary(cluster_df.drop(drop_list, axis=1), cluster_column_name='cluster')
    print(cluster_summary_df)
    print()

    cluster_df['decade'] = cluster_df['value'].str[0:3]
    grouped = pd.DataFrame(cluster_df.groupby(['decade', 'cluster'])['cluster'].count())
    grouped.columns = ['count']
    grouped = grouped.reset_index(drop=False)
    grouped = grouped.loc[grouped['cluster'] == 0]
    make_ts_plot(grouped, 'decade', 'count', title=f'{season}_{year_col}_{feature} cluster',
                 save_directory='modeling/clustering')


if __name__ == "__main__":
    if not os.path.exists('modeling/clustering/plots'):
        os.makedirs('modeling/clustering/plots')

    main(
        year_col='year',
        season='winter',
        feature='min_temp'
    )

    main(
        year_col='year',
        season='summer',
        feature='max_temp'
    )
