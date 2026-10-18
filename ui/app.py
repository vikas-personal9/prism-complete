"""Streamlit dashboard for the four supplied PRISM datasets and live Azure output."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import plotly.express as px
import streamlit as st
from src.features import build_features, build_task_summary, build_telemetry_summary, load_sources, sample_event
from src.predict import predict
from src.pipeline import process_event
from src.storage import read_all, record_approval

st.set_page_config(page_title='PRISM | Runtime Intelligence', page_icon='📊', layout='wide')
st.title('PRISM | Predictive Runtime Intelligence & Sizing Monitor')
st.caption('Historical CSV simulation + optional Azure ML/Foundry live processing. Historical figures are not live Databricks measurements.')

@st.cache_data(show_spinner='Validating source files and building features')
def load_data(root):
    return build_features(root), build_task_summary(root), build_telemetry_summary(root), load_sources(root)

@st.cache_data(show_spinner='Computing retrospective estimates')
def batch_predictions(root, backend):
    df = build_features(root)
    # For reproducibility and to avoid 100 paid network calls, compute chart in local mode.
    records = []
    for row in df.itertuples(index=False):
        obj = df[df.run_id == row.run_id].iloc[0]
        event = sample_event(obj, 'start')
        result = predict(event['features'], stage='start', backend='local')
        records.append({'run_id': row.run_id, 'model_estimate_min': result['prediction_minutes']})
    return pd.DataFrame(records)

root = os.getenv('PRISM_DATA_DIR', 'data')
try:
    runs, tasks, telemetry, sources = load_data(root)
except Exception as error:
    st.error(f'Dataset validation error: {error}')
    st.stop()

st.sidebar.header('Filters and mode')
workflow = st.sidebar.selectbox('Workflow', ['All'] + sorted(runs.workflow_id.unique().tolist()))
backend = st.sidebar.selectbox('Inference backend', ['local', 'azure'], index=0 if os.getenv('PRISM_PREDICTION_BACKEND', 'local') == 'local' else 1)
storage_backend = os.getenv('PRISM_STORAGE_BACKEND', 'local')
st.sidebar.caption(f'Result storage: {storage_backend}. Every recommendation requires manual review.')
filtered = runs if workflow == 'All' else runs[runs.workflow_id == workflow]
joined = filtered.merge(telemetry, on=['workflow_id','run_id'], validate='one_to_one')

pages = st.tabs(['Executive overview', 'Runtime predictions', 'Cluster sizing', 'Spark + tasks',
                 'AI agent reasoning', 'Review / audit', 'Dataset + model quality'])
with pages[0]:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric('Historical runs', len(filtered))
    c2.metric('Workflows', filtered.workflow_id.nunique())
    c3.metric('Mean actual runtime', f'{filtered.runtime_min.mean():.2f} min')
    c4.metric('Successful runs', f'{sum(filtered.result_state == "SUCCESS")}/{len(filtered)}')
    c5.metric('Peak observed CPU', f'{joined.cpu_max.max():.1f}%')
    fig = px.line(filtered.sort_values('start_time'), x='start_time', y='runtime_min', color='workflow_id',
                  markers=True, title='Actual historical workflow runtime')
    st.plotly_chart(fig, use_container_width=True)
    st.info('No actual SLA thresholds or cloud VM rates were provided. SLA breaches and dollar savings are not calculated.')
    try:
        results = read_all('predictions')
        st.subheader(f'Processed event results ({len(results)} stored, maximum 500 shown)')
        if results:
            st.dataframe(pd.DataFrame([{'run_id':r['run_id'],'stage':r['stage'],
                'predicted_min':r['prediction']['prediction_minutes'],
                'flags':', '.join(r.get('flags',[])),
                'review_status':r.get('review_status','PENDING')} for r in results]), use_container_width=True)
        else:
            st.caption('No operational events processed yet. Use Runtime predictions > Process and store.')
    except Exception as exc:
        st.warning(f'Could not read results: {exc}')

with pages[1]:
    st.subheader('Run-level runtime prediction')
    left,right = st.columns([1,2])
    with left:
        run_id = st.selectbox('Historical run', filtered.run_id.tolist())
        stage = st.radio('Prediction point', ['start', 'checkpoint'], horizontal=True)
        st.caption('Start: forecast total runtime. Checkpoint: forecast remaining runtime after first metric sample.')
    chosen = filtered[filtered.run_id == run_id].iloc[0]
    event = sample_event(chosen, stage)
    with right:
        st.json(event, expanded=False)
    if st.button('Score selected run', type='primary'):
        try:
            response = predict(event['features'], stage=stage, backend=backend)
            st.session_state['last_prediction'] = response
            st.session_state['last_event'] = event
        except Exception as exc:
            st.error(f'Prediction call failed: {exc}')
    if st.session_state.get('last_event') == event:
        response = st.session_state['last_prediction']
        if 'error' in response:
            st.error(response['error'])
        else:
            st.metric('Estimated ' + ('total' if stage=='start' else 'remaining') + ' minutes',
                      f"{response['prediction_minutes']:.2f}")
            st.caption('Retrospective example: the selected run may have been used during model training.')
    if st.button('Process event, run agents, and store result'):
        try:
            response = process_event(event, backend=backend, persist=True)
            st.session_state['last_result'] = response
            st.success(f"Stored {response['run_id']} ({response['stage']}); see AI agent reasoning and Review tabs")
        except Exception as exc:
            st.error(f'Processing failed: {exc}')
    st.subheader('Retrospective model estimates vs actual')
    try:
        backtest = batch_predictions(root, 'local')
        chart = filtered[['run_id','start_time','runtime_min','workflow_id']].merge(backtest,on='run_id')
        melted = chart.melt(id_vars=['run_id','workflow_id','start_time'], value_vars=['runtime_min','model_estimate_min'],
                            var_name='series', value_name='minutes')
        st.plotly_chart(px.line(melted.sort_values('start_time'), x='start_time', y='minutes', color='series',
                                hover_data=['run_id','workflow_id'], title='In-sample / mixed retrospective comparison (not holdout validation)'),
                        use_container_width=True)
        st.warning('This chart is not an unbiased performance evaluation: the models were fitted on some displayed runs. Use holdout MAE in Model quality.')
    except Exception as exc:
        st.warning(f'Chart unavailable: {exc}')

with pages[2]:
    st.subheader('Cluster sizing signals (observational)')
    st.plotly_chart(px.scatter(joined, x='cpu_avg', y='runtime_min', size='workers_avg', color='workflow_id',
                               hover_data=['run_id','memory_avg','pending_max'],
                               labels={'cpu_avg':'Mean CPU (%)','runtime_min':'Runtime (min)'},
                               title='CPU, worker count and workflow runtime'), use_container_width=True)
    st.dataframe(joined[['run_id','cluster_id','workers_avg','cpu_avg','cpu_max','memory_avg','memory_max','pending_max']],
                 use_container_width=True)
    st.caption('No hourly VM price or valid controlled before/after experiment exists: savings and optimal sizing cannot be inferred.')
with pages[3]:
    st.subheader('Spark execution and workflow task diagnostics')
    st.dataframe(joined[['run_id','spill_disk_max','shuffle_read_max','spark_failed_max','runtime_min']],
                 use_container_width=True)
    task_rows = tasks.merge(filtered[['workflow_id','run_id']], on=['workflow_id','run_id'])
    st.dataframe(task_rows[['run_id','task_count','task_runtime_sum_min','task_failed_count']], use_container_width=True)
    sparks = sources['spark']
    if workflow != 'All':
        sparks = sparks[sparks.workflow_id==workflow]
    st.plotly_chart(px.scatter(sparks,x='shuffle_read_bytes',y='spill_disk_bytes',color='workflow_id',
                               hover_data=['run_id','stage_id'],title='Recorded Spark shuffle vs disk spill'),use_container_width=True)
with pages[4]:
    st.subheader('Specialized agent outputs')
    result = st.session_state.get('last_result')
    try:
        saved = read_all('predictions')
    except Exception as exc:
        saved = []
        st.warning(f'Stored results unavailable: {exc}')
    if saved:
        ids = [f"{r['run_id']} / {r['stage']}" for r in saved]
        choice = st.selectbox('Open processed record', list(range(len(saved))), format_func=lambda i: ids[i])
        result = saved[choice]
    if result:
        st.metric('Prediction (minutes)', result['prediction']['prediction_minutes'])
        st.write('Detected risk signals:', result.get('flags',[]))
        st.json(result['analysis'])
    else:
        st.info('Process a workflow event first. Without Azure OpenAI credentials the deterministic rules engine is used.')
with pages[5]:
    st.warning('Demo approval register only. Approval never executes an infrastructure change. Restrict dashboard access for real environments.')
    try:
        stored = read_all('predictions')
    except Exception as exc:
        stored=[]
        st.warning(f'Cannot read stored predictions: {exc}')
    if stored:
        selected = st.selectbox('Prediction requiring review', list(range(len(stored))),
                                format_func=lambda i: f"{stored[i]['run_id']} / {stored[i]['stage']}")
        decision = st.radio('Review decision', ['APPROVED','REJECTED'], horizontal=True)
        reviewer = st.text_input('Reviewer name or directory ID')
        note = st.text_area('Decision justification')
        if st.button('Record decision'):
            try:
                entry = record_approval(stored[selected]['run_id'], decision, reviewer, note)
                st.success(f"Review saved: {entry['decision']}; execution status: NOT_EXECUTED")
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info('There are no processed predictions to review yet.')
    try:
        entries = read_all('approvals')
        if entries:
            st.dataframe(pd.DataFrame(entries), use_container_width=True)
    except Exception as exc:
        st.warning(f'Approval history unavailable: {exc}')
with pages[6]:
    st.subheader('Exact dataset statistics')
    st.dataframe(pd.DataFrame([{'dataset': k, 'rows':len(d), 'columns':len(d.columns),
                                'missing_cells':int(d.isna().sum().sum())} for k,d in sources.items()]),use_container_width=True)
    st.write('Actual input feature names')
    from src.features import START_FEATURES,CHECKPOINT_FEATURES
    st.json({'start_features':START_FEATURES, 'checkpoint_features':CHECKPOINT_FEATURES})
    try:
        metrics=json.loads(Path('artifacts/metrics.json').read_text())
        st.json(metrics)
    except Exception as exc:
        st.warning(f'Model report unavailable: {exc}')
    st.caption('All 50 runs are successful; failure risk classifier is intentionally not trained.')
