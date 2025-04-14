import os
from collections import namedtuple
import warnings

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from feature_engine.creation import CyclicalFeatures
import numpy as np
import pandas as pd
import joblib
from hyperopt import Trials, fmin, hp, space_eval, tpe
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def read_training_data():
    return pd.read_csv('data/clean/kc_weather_analyze_full.csv')


def create_target(df, col, shift_n):
    df['target'] = df[col].shift(-shift_n)
    df = df.loc[~df['target'].isnull()]
    return df


def create_rolling_features(df, cols, rolling_n):
    for col in cols:
        df[f'{col}_rolling_mean'] = df[col].rolling(rolling_n).mean()
        df[f'{col}_rolling_median'] = df[col].rolling(rolling_n).median()
        df[f'{col}_rolling_std'] = df[col].rolling(rolling_n).std()
    return df


def drop_columns(df, cols):
    return df.drop(labels=cols, axis=1, errors='ignore')


def generate_models_to_train():
    hgb_grid = {
        "model__learning_rate": hp.uniform("model__learning_rate",0.01, 1.0),
        "model__max_iter": hp.randint("model__max_iter",50, 500),
        "model__min_samples_leaf": hp.choice("model__min_samples_leaf", [10, 20, 30]),
        "model__max_leaf_nodes": hp.choice("model__max_leaf_nodes", [10, 20, 30]),
    }

    model_named_tuple = namedtuple('model_config', {'model_name', 'model', 'param_space', 'iterations',
                                                    'drop_year'})
    model_list = [
        model_named_tuple(
            model_name='hgb_mean',
            model=HistGradientBoostingRegressor(),
            param_space=hgb_grid,
            iterations=50,
            drop_year=True
        ),
        model_named_tuple(
            model_name='ridge',
            model=Ridge(),
            param_space={"model__alpha": hp.uniform("model__alpha",0.1, 100.0)},
            iterations=50,
            drop_year=False
        ),
        model_named_tuple(
            model_name='hgb_q1',
            model=HistGradientBoostingRegressor(quantile=0.10),
            param_space=hgb_grid,
            iterations=50,
            drop_year=True
        ),
        model_named_tuple(
            model_name='hgb_q25',
            model=HistGradientBoostingRegressor(quantile=0.25),
            param_space=hgb_grid,
            iterations=50,
            drop_year=True
        ),
        model_named_tuple(
            model_name='hgb_q75',
            model=HistGradientBoostingRegressor(quantile=0.75),
            param_space=hgb_grid,
            iterations=50,
            drop_year=True
        ),
        model_named_tuple(
            model_name='hgb_q90',
            model=HistGradientBoostingRegressor(quantile=0.90),
            param_space=hgb_grid,
            iterations=50,
            drop_year=True
        ),
    ]
    return model_list


def create_custom_ts_cv_splits(df: pd.DataFrame, start_year: int, end_year: int, cv_folds: int) -> list:
    """
    Creates a set of custom cross validation splits based on year. The function takes a start year and an end year along
    with a number of cv folds. Based on the number of years between the provided years, it will create an equal number
    of array splits based on cv folds. The folds are arranged as a time-series cross validation problem. That is,
    in the first split, the first split is the training data and the second split is the testing data. In the second
    split, the first two splits are the training data, and the third split is the testing data. And so on.

    :param df: pandas dataframe of training data
    :param start_year: start year of the cross validation folds
    :param end_year: end year of the cross validation folds
    :param cv_folds: number of cv folds
    :return: list of tuples, with each tuple containing two items - the first is the index of the training observations
    and the second is the index of the testing observations
    """
    cv_splits = []
    years = list(np.arange(start_year, end_year + 1, 1))
    year_splits = np.array_split(years, cv_folds)
    for n, year_split in enumerate(year_splits):
        if n != cv_folds - 1:
            train_ids = year_splits[:n + 1]
            train_ids = np.concatenate(train_ids)
            test_ids = year_splits[n + 1]
            train_indices = df.loc[df['year'].isin(train_ids)].index.values.astype(int)
            test_indices = df.loc[df['year'].isin(test_ids)].index.values.astype(int)
            cv_splits.append((train_indices, test_indices))
    return cv_splits


def train_model(estimator, x_train, y_train, search_space, cv, scoring, iterations, model_name):
    cv_scores_df = pd.DataFrame()

    def _model_objective(params):
        estimator.set_params(**params)
        score = cross_val_score(estimator, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        temp_cv_scores_df = pd.DataFrame(score)
        temp_cv_scores_df = temp_cv_scores_df.reset_index()
        temp_cv_scores_df['index'] = 'fold_' + temp_cv_scores_df['index'].astype(str)
        temp_cv_scores_df = temp_cv_scores_df.T
        temp_cv_scores_df = temp_cv_scores_df.add_prefix('fold_')
        temp_cv_scores_df = temp_cv_scores_df.iloc[1:]
        temp_cv_scores_df['mean'] = temp_cv_scores_df.mean(axis=1)
        temp_cv_scores_df['std'] = temp_cv_scores_df.std(axis=1)
        temp_params_df = pd.DataFrame(params, index=list(range(0, len(params) + 1)))
        temp_cv_scores_df = pd.concat([temp_params_df, temp_cv_scores_df], axis=1)
        temp_cv_scores_df = temp_cv_scores_df.dropna()
        nonlocal cv_scores_df
        cv_scores_df = pd.concat([cv_scores_df, temp_cv_scores_df], axis=0)
        return 1 - score.mean()

    trials = Trials()
    best = fmin(_model_objective, search_space, algo=tpe.suggest, max_evals=iterations, trials=trials)
    best_params = space_eval(search_space, best)

    cv_scores_df = cv_scores_df.sort_values(by=['mean'], ascending=False)
    cv_scores_df = cv_scores_df.reset_index(drop=True)
    cv_scores_df = cv_scores_df.reset_index()
    cv_scores_df = cv_scores_df.rename(columns={'index': 'ranking'})
    cv_scores_df.to_csv(f'modeling/ml/output/{model_name}_cv_scores.csv', index=False)

    estimator.set_params(**best_params)
    estimator.fit(x_train, y_train)

    joblib.dump(estimator, f'modeling/ml/output/{model_name}_model.pkl')
    return estimator


def run_permutation_importance(estimator, x_df, y_df, scoring, model_name):
    result = permutation_importance(estimator, x_df, y_df, n_repeats=10, random_state=0, scoring=scoring)
    df = pd.DataFrame({
        'permutation_importance_mean': result.importances_mean,
        'permutation_importance_std': result.importances_std,
        'feature': list(x_df)
    })
    df.sort_values(by=['permutation_importance_mean'], ascending=False, inplace=True)
    df.to_csv(f'modeling/ml/output/{model_name}_permutation_importance.csv', index=False)
    return df


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


def main(target, targets_ahead):
    df = read_training_data()
    df = df[['date', target]]
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['year'] = df['date'].dt.year
    df = create_target(df, col=target, shift_n=targets_ahead)

    cf = CyclicalFeatures(variables=['day_of_year'])
    cf.fit(df)
    df = cf.transform(df)

    df = create_rolling_features(df, cols=[target], rolling_n=7)
    df = df.dropna()
    df = df.reset_index(drop=True)
    dates = df['date'].tolist()

    models = generate_models_to_train()
    cv_splits = create_custom_ts_cv_splits(df, start_year=df['year'].min(), end_year=df['year'].max(), cv_folds=10)

    cols_to_drop = ['date', 'day_of_year']
    df = drop_columns(df, cols=cols_to_drop)
    df = df.astype(float)
    df['year'] = df['year'].astype(str)

    y = df['target']
    x = df.drop('target', axis=1)

    for model in models:
        if model.drop_year:
            estimator = Pipeline(
                steps=[
                    ('dropper', FunctionTransformer(drop_columns, validate=False, kw_args={'cols': ['year']})),
                    ('model', model.model)
                ]
            )
        else:
            categorical_transformer = Pipeline(steps=[('ohc', OneHotEncoder(handle_unknown='ignore'))])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('categorical_transformer', categorical_transformer, selector(dtype_exclude='number'))
                ],
                remainder='passthrough',
            )

            estimator = Pipeline(
                steps=[
                    ('preprocessor', preprocessor),
                    ('model', model.model)
                ]
            )

        estimator = train_model(estimator=estimator, x_train=x, y_train=y, search_space=model.param_space,
                                cv=cv_splits, scoring='neg_mean_absolute_error', iterations=model.iterations,
                                model_name=model.model_name)
        run_permutation_importance(estimator, x_df=x, y_df=y, scoring='neg_mean_absolute_error',
                                   model_name=model.model_name)

        if not model.drop_year:
            coef_df = pd.DataFrame({
                'coef': estimator['model'].coef_,
                'feature': estimator[:-1].get_feature_names_out()
            })
            coef_df = coef_df.sort_values(by=['coef'])
            coef_df.to_csv(f'modeling/ml/output/{model.model_name}_coefs.csv', index=False)

        predictions_df = pd.DataFrame({
            'date': dates,
            'actual': y,
            'prediction': estimator.predict(x),
        })
        predictions_df['error'] = abs(predictions_df['actual'] - predictions_df['prediction'])
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        predictions_df['year'] = predictions_df['date'].dt.year

        predictions_df['season'] = predictions_df['date'].apply(get_season)
        predictions_df.to_csv(f'modeling/ml/output/{model.model_name}_predictions.csv', index=False)

        grouped = pd.DataFrame(predictions_df.groupby(['year', 'season'])['error'].mean())
        grouped = grouped.reset_index(drop=False)
        grouped['season'] = pd.Categorical(grouped['season'], categories=['winter', 'spring', 'summer', 'fall'])
        grouped = grouped.sort_values(by=['year', 'season'])
        grouped.to_csv(f'modeling/ml/output/{model.model_name}_grouped_errors.csv', index=False)

        sns.lineplot(data=grouped, x='year', y='error', hue='season')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'modeling/ml/output/{model.model_name}_error.png')
        plt.clf()


if __name__ == "__main__":
    if not os.path.exists('modeling/ml/output'):
        os.makedirs('modeling/ml/output')

    main(
        target='max_temp',
        targets_ahead=7,
    )
