"""Azure ML managed online endpoint scoring contract.

Input: {"stage":"checkpoint","features":{...actual feature names...}}
Output: {"stage":"checkpoint","prediction_minutes":number,"target":"remaining_min","unit":"minutes"}
"""
import json
import os
from pathlib import Path
import joblib
import pandas as pd

MODELS = {}

def init():
    model_dir = Path(os.environ.get('AZUREML_MODEL_DIR', 'artifacts'))
    for stage in ('start', 'checkpoint'):
        matches = list(model_dir.rglob(f'{stage}.joblib'))
        if not matches:
            raise FileNotFoundError(f'{stage}.joblib missing under {model_dir}')
        MODELS[stage] = joblib.load(matches[0])

def run(raw_data):
    try:
        if not MODELS:
            init()
        payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        stage = payload.get('stage', payload.get('model', 'start'))
        if stage not in MODELS:
            raise ValueError('stage must be start or checkpoint')
        spec = MODELS[stage]
        features = payload['features']
        missing = sorted(set(spec['features']) - set(features))
        if missing:
            raise ValueError(f'Missing input features: {missing}')
        frame = pd.DataFrame([features])[spec['features']]
        # Consistent nullable numeric and bool preparation for JSON callers.
        prediction = max(0.0, float(spec['pipeline'].predict(frame)[0]))
        return {'stage': stage, 'prediction_minutes': round(prediction, 3),
                'target': spec['target'], 'unit': 'minutes'}
    except (KeyError, ValueError, TypeError) as exc:
        return {'error': str(exc)}
