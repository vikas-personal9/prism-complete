"""Deploy from REPOSITORY ROOT with `func azure functionapp publish <appname>`.

Event Hub -> Azure Storage Queue -> Azure ML inference -> role agents -> Blob results.
Event Hub and Queue triggers are separate so messages can retry independently.
"""
import json
import logging
import os
import azure.functions as func
from azure.storage.queue import QueueClient
from src.pipeline import process_event

app = func.FunctionApp()

@app.function_name(name='IngestOperationalMetrics')
@app.event_hub_message_trigger(arg_name='event', event_hub_name='prism-metrics',
                               connection='EVENTHUB_CONNECTION', consumer_group='prism-functions')
def ingest_operational_metrics(event: func.EventHubEvent):
    payload = json.loads(event.get_body().decode('utf-8'))
    if not {'schema_version', 'run_id', 'workflow_id', 'stage', 'features'}.issubset(payload):
        raise ValueError('Invalid PRISM event, missing required keys')
    client = QueueClient.from_connection_string(os.environ['STORAGE_CONNECTION_STRING'],
                                                queue_name='prism-inference')
    client.send_message(json.dumps(payload))
    logging.info('Queued PRISM event %s / %s', payload['run_id'], payload['stage'])

@app.function_name(name='RunPrismInference')
@app.queue_trigger(arg_name='event', queue_name='prism-inference', connection='AzureWebJobsStorage')
def run_prism_inference(event: func.QueueMessage):
    payload = json.loads(event.get_body().decode('utf-8'))
    record = process_event(payload, backend=os.getenv('PRISM_PREDICTION_BACKEND', 'azure'))
    logging.info('Scored PRISM event: run=%s stage=%s pred=%s', record['run_id'],
                 record['stage'], record['prediction']['prediction_minutes'])
