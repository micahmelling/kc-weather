import os
import warnings

from scipy.stats import ks_2samp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def make_ts_plot(df, x_col, y_col, title, set_plot_limits=False, save_directory='explore'):
    sns.lineplot(x=x_col, y=y_col, data=df, label="Time Series", errorbar=None)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if set_plot_limits:
        plt.ylim(ymin=0)
        x_ticks = np.arange(0, len(df[x_col]), 10)
        plt.xticks(x_ticks, df[x_col].iloc[x_ticks], rotation=90)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{save_directory}/plots/{title.replace(' ', '_')}.png')
    plt.clf()


def make_ts_multiline_plot(df, cols, x_col, title):
    for col in cols:
        sns.lineplot(y=col, x=x_col, data=df, label=col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.ylabel('autocorrelation')
    x_ticks = np.arange(0, len(df[x_col]), 10)
    plt.xticks(x_ticks, df[x_col].iloc[x_ticks], rotation=90)
    plt.savefig(f'explore/plots/{title.replace(' ', '_')}.png')
    plt.clf()


def make_dist_plot(df, col, title):
    sns.histplot(df[col], color="red")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.savefig(f'explore/plots/{title.replace(' ', '_')}.png')
    plt.clf()


def run_skew_and_kurtosis_analysis(df, group_col, analyze_col):
    skew_year_df = pd.DataFrame(df.groupby(group_col)[analyze_col].skew())
    skew_year_df = skew_year_df.reset_index(drop=False)

    kurtosis_year_df = df[[analyze_col, group_col]].groupby(group_col).apply(pd.DataFrame.kurt)
    kurtosis_year_df = kurtosis_year_df.drop(labels=group_col, axis=1)
    kurtosis_year_df = kurtosis_year_df.reset_index(drop=False)

    return skew_year_df, kurtosis_year_df


def get_rolling_autocorrelation(df, analyze_col, n_lags):
    for lag in range(1, n_lags + 1):
        df[f'{analyze_col}_{lag}'] = df[analyze_col].shift(lag)

    cols_for_corr = [c for c in list(df) if analyze_col in c]
    corr = df[cols_for_corr].expanding().corr()
    corr = corr.reset_index(drop=False)

    corr = corr.loc[corr['level_1'] == analyze_col]
    corr = corr.drop(analyze_col, axis=1)
    corr = corr.drop(['level_0', 'level_1'], axis=1)
    corr.columns = [c.replace(analyze_col, 'lag') for c in corr.columns]
    corr['year'] = df['value'].tolist()
    return corr


def run_rolling_ks_test(df, year_col, analyze_col, lookback_n, analyze_n, stat_sig_rolling=5, ks_type='two-sided'):
    df = df.sort_values(by=[year_col], ascending=True)
    years = list(df[year_col].unique())
    pairs = []
    for counter, year in enumerate(years):
        pair = years[(counter - lookback_n): counter], years[counter:(counter + analyze_n)]
        pairs.append(pair)

    results = []
    for pair in pairs:
        look_back = pair[0]
        look_ahead = pair[1]
        if len(look_back) == lookback_n and len(look_ahead) == analyze_n:
            srs1 = df.loc[df[year_col].isin(look_back)]
            srs2 = df.loc[df[year_col].isin(look_ahead)]
            ks_test = ks_2samp(srs1[analyze_col], srs2[analyze_col], alternative=ks_type)
            look_back = [str(s) for s in look_back]
            look_ahead = [str(s) for s in look_ahead]
            pair_df = pd.DataFrame({
                'year_group_1': [look_back],
                'year_group_1_avg': [srs1[analyze_col].median()],
                'year_group_2': [look_ahead],
                'year_group_2_avg': [srs2[analyze_col].median()],
                'p_value': [ks_test.pvalue]
            })
            results.append(pair_df)

    results_df = pd.concat(results, axis=0)
    results_df['stat_sig'] = np.where(
        results_df['p_value'] <= 0.05,
        1,
        0
    )
    results_df['stat_sig_rolling'] = results_df['stat_sig'].rolling(stat_sig_rolling).mean()
    results_df = results_df.reset_index(drop=True)
    return results_df


def main(variables, year_col, season):
    if not os.path.exists('explore/plots'):
        os.makedirs('explore/plots')
    if not os.path.exists('explore/files'):
        os.makedirs('explore/files')

    full_df = pd.read_csv('data/clean/kc_weather_analyze_full.csv')
    summary_df = pd.read_csv('data/clean/kc_weather_analyze_summary.csv')

    for variable in variables:
        analysis_full_df = full_df.loc[full_df['season'] == season]
        output_name = f'{variable} {season} {year_col}'
        skew_year_df, kurtosis_year_df = run_skew_and_kurtosis_analysis(analysis_full_df, group_col=year_col,
                                                                        analyze_col=variable)
        make_ts_plot(skew_year_df, x_col=year_col, y_col=variable, title=f'Skew TS {output_name}')
        make_ts_plot(kurtosis_year_df, x_col=year_col, y_col=variable, title=f'Kurt TS {output_name}')
        ks_results_df = run_rolling_ks_test(analysis_full_df, year_col=year_col, analyze_col=variable,
                                            lookback_n=5, analyze_n=1)
        ks_results_df = ks_results_df.reset_index(drop=False)
        make_ts_plot(ks_results_df, x_col='index', y_col='stat_sig_rolling', title=f'KS Test {output_name}')
        ks_results_df.to_csv(f'explore/files/ks_test_{output_name}.csv', index=False)

        analysis_summary_df = summary_df.loc[summary_df['group'] == year_col]
        analysis_summary_df = analysis_summary_df.loc[analysis_summary_df['sub_value'] == season]
        analysis_summary_df = analysis_summary_df.loc[analysis_summary_df['feature'] == variable]

        make_ts_plot(analysis_summary_df, x_col='value', y_col='mean', title=f'TS {output_name}', set_plot_limits=True)
        make_dist_plot(analysis_summary_df, col='mean', title=f'mean {output_name}')
        autocorrelation_df = get_rolling_autocorrelation(analysis_summary_df, analyze_col='mean', n_lags=5)
        autocorrelation_cols = list(autocorrelation_df)
        autocorrelation_cols.remove(year_col)
        make_ts_multiline_plot(autocorrelation_df, cols=autocorrelation_cols, x_col=year_col,
                               title=f'autocorrelation mean {output_name}')


if __name__ == "__main__":
    main(
        variables=['min_temp', 'range_temp', 'extreme_low_min_rolling_mean', 'extreme_low_max_rolling_mean'],
        year_col='year',
        season='winter'
    )

    main(
        variables=['max_temp', 'range_temp', 'extreme_high_rolling_mean'],
        year_col='year',
        season='summer'
    )
