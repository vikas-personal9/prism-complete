"""Azure ML REST predictions or strictly-local predictions."""
import os
from pathlib import Path
import requests

def predict(features, stage='checkpoint', backend=None):
    backend = (backend or os.getenv('PRISM_PREDICTION_BACKEND', 'local')).lower()
    if backend == 'local':
        # Avoid Azure dependencies in local demo; score.py loads same exact artifacts.
        from ml.score import run
        return run({'stage': stage, 'features': features})
    if backend != 'azure':
        raise ValueError(f'Unknown prediction backend: {backend}')
    uri = os.environ['AML_SCORING_URI']
    key = os.getenv('AML_ENDPOINT_KEY')
    if key:
        auth = f'Bearer {key}'
    else:
        from azure.identity import DefaultAzureCredential
        auth = f"Bearer {DefaultAzureCredential().get_token('https://ml.azure.com/.default').token}"
    r = requests.post(uri, json={'stage': stage, 'features': features},
                      headers={'Authorization': auth, 'Content-Type': 'application/json'}, timeout=90)
    r.raise_for_status()
    payload = r.json()
    # Some frontends wrap inference output in a JSON string.
    if isinstance(payload, str):
        import json
        payload = json.loads(payload)
    if payload.get('error'):
        raise RuntimeError(payload['error'])
    return payload
