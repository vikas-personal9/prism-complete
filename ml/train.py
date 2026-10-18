"""Train start and first-checkpoint models with a chronological holdout."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.features import build_features, START_FEATURES, CHECKPOINT_FEATURES

MODELS = {'start': (START_FEATURES, 'runtime_min'), 'checkpoint': (CHECKPOINT_FEATURES, 'remaining_min')}

def make_pipeline(features, frame):
    categories = [x for x in features if frame[x].dtype == 'object' or str(frame[x].dtype) == 'bool']
    numbers = [x for x in features if x not in categories]
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True))])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return Pipeline([('preprocess', ColumnTransformer([('numeric', numeric, numbers),
                                                      ('categorical', categorical, categories)])),
                     ('regressor', RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                         n_jobs=-1, random_state=42))])

def train(data_dir='data', out_dir='artifacts'):
    df = build_features(data_dir).sort_values(['start_time', 'workflow_id']).reset_index(drop=True)
    # Split on entire days: no future-day labels appear in fitting data.
    dates = sorted(df.start_time.dt.normalize().unique())
    cutoff = dates[max(1, int(len(dates) * 0.8))]
    train_df = df[df.start_time < cutoff]
    test_df = df[df.start_time >= cutoff]
    if len(train_df) < 10 or len(test_df) < 2:
        raise ValueError('Not enough training/test rows for chronological validation')
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {'data_rows': len(df), 'workflows': int(df.workflow_id.nunique()),
              'train_rows': len(train_df), 'test_rows': len(test_df),
              'train_end': str(train_df.start_time.max()),
              'test_start': str(test_df.start_time.min()),
              'warning': 'Small synthetic-like history, all runs successful; no failure model or causal savings estimate.',
              'models': {}}
    for name, (features, target) in MODELS.items():
        model = make_pipeline(features, train_df)
        model.fit(train_df[features], train_df[target])
        predictions = np.maximum(model.predict(test_df[features]), 0)
        truth = test_df[target].to_numpy()
        reference = test_df['prior_3_mean_min'].fillna(train_df.runtime_min.median())
        if name == 'checkpoint':
            reference = (reference - test_df.elapsed_min).clip(lower=0)
        baseline = reference.to_numpy()
        report['models'][name] = {
            'target': target, 'features': features,
            'mae_minutes': round(float(mean_absolute_error(truth, predictions)), 3),
            'rmse_minutes': round(float(np.sqrt(mean_squared_error(truth, predictions))), 3),
            'baseline_mae_minutes': round(float(mean_absolute_error(truth, baseline)), 3),
            'holdout_samples': len(test_df)}
        joblib.dump({'pipeline': model, 'features': features, 'stage': name, 'target': target},
                    output / f'{name}.joblib')
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='artifacts')
    args = ap.parse_args()
    train(args.data_dir, args.out_dir)
