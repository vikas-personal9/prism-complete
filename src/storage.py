"""Result/review store. Local JSON files for demos; Azure Blob for deployed app."""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def _name(identifier):
    clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(identifier))
    if not clean or clean.startswith('.'):
        raise ValueError('Invalid identifier')
    return clean[:160]

def _backend():
    return os.getenv('PRISM_STORAGE_BACKEND', 'local').lower()

def _container():
    from azure.storage.blob import BlobServiceClient
    conn = os.getenv('STORAGE_CONNECTION_STRING')
    if conn:
        service = BlobServiceClient.from_connection_string(conn)
    else:
        from azure.identity import DefaultAzureCredential
        service = BlobServiceClient(os.environ['STORAGE_ACCOUNT_URL'], credential=DefaultAzureCredential())
    return service.get_container_client(os.getenv('PRISM_RESULTS_CONTAINER', 'prism-results'))

def write_json(category, identifier, value):
    category = _name(category)
    filename = _name(identifier) + '.json'
    raw = json.dumps(value, indent=2, default=str)
    if _backend() == 'local':
        folder = Path(os.getenv('PRISM_LOCAL_RESULTS', 'runtime_results')) / category
        folder.mkdir(parents=True, exist_ok=True)
        (folder / filename).write_text(raw)
    else:
        _container().upload_blob(f'{category}/{filename}', raw, overwrite=True)
    return f'{category}/{filename}'

def read_all(category, limit=500):
    category = _name(category)
    if _backend() == 'local':
        folder = Path(os.getenv('PRISM_LOCAL_RESULTS', 'runtime_results')) / category
        files = sorted(folder.glob('*.json'), reverse=True)[:limit] if folder.exists() else []
        return [json.loads(f.read_text()) for f in files]
    result = []
    container = _container()
    for entry in container.list_blobs(name_starts_with=f'{category}/'):
        result.append(json.loads(container.download_blob(entry.name).readall()))
        if len(result) >= limit:
            break
    return result

def record_approval(run_id, decision, reviewer, note=''):
    if decision not in {'APPROVED', 'REJECTED'}:
        raise ValueError('Approval decision must be APPROVED or REJECTED')
    if not reviewer or not reviewer.strip():
        raise ValueError('Reviewer is required')
    record = {'run_id': _name(run_id), 'decision': decision,
              'reviewer': reviewer.strip(), 'note': note.strip(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'execution_status': 'NOT_EXECUTED'}
    # Unique review ID, keep previous reviews as audit events.
    from uuid import uuid4
    write_json('approvals', f'{_name(run_id)}-{uuid4().hex}', record)
    return record
