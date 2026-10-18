"""Replay an observed run as a synthetic operational event; no ground truth sent."""
import argparse
import json
import os
from src.features import build_features, sample_event

def create_event(run_id, stage='checkpoint', data_dir='data'):
    df = build_features(data_dir)
    hit = df[df.run_id == run_id]
    if hit.empty:
        raise ValueError(f'Unknown run_id {run_id}')
    return sample_event(hit.iloc[0], stage)

def publish(event):
    from azure.eventhub import EventHubProducerClient, EventData
    conn = os.environ['EVENTHUB_CONNECTION']
    hub = os.getenv('EVENTHUB_NAME', 'prism-metrics')
    with EventHubProducerClient.from_connection_string(conn, eventhub_name=hub) as producer:
        batch = producer.create_batch()
        batch.add(EventData(json.dumps(event)))
        producer.send_batch(batch)
    return event['run_id']

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', default='WF001-R10')
    p.add_argument('--stage', choices=['start', 'checkpoint'], default='checkpoint')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--send', action='store_true')
    x = p.parse_args()
    event = create_event(x.run_id, x.stage, x.data_dir)
    if x.send:
        publish(event)
        print('Sent to Event Hubs:', event['run_id'])
    else:
        print(json.dumps(event, indent=2))
