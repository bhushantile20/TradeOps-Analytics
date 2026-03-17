from django.shortcuts import render
import pandas as pd
import numpy as np
import os
import json

from .utils import calculate_metrics

def landing_page(request):
    return render(request, 'landing_page.html')

def dashboard_overview(request):
    log_path = os.path.join(os.path.dirname(__file__), '../../logs/predictions.parquet')
    drift_report_path = os.path.join(os.path.dirname(__file__), '../../reports/drift/data_drift_report.json')
    model_drift_path = os.path.join(os.path.dirname(__file__), '../../reports/drift/model_drift_report.json')
    
    logs = []
    chart_logs = "[]"
    model_metrics = {}
    top_performer = "LR" # Default
    
    if os.path.exists(log_path):
        df = pd.read_parquet(log_path)
        
        # Calculate performance metrics
        raw_metrics = calculate_metrics(df)
        
        # Convert to template-friendly keys (capitalized or mapping-based)
        model_metrics = {k.lower().replace(' ', '_'): v for k, v in raw_metrics.items()}
        
        # Determine top performer (lowest RMSE)
        best_rmse = float('inf')
        for m, m_data in raw_metrics.items():
            if m_data['rmse'] < best_rmse and m_data['rmse'] > 0:
                best_rmse = m_data['rmse']
                top_performer = m

        # Handle NaN values for JSON and template safety
        df = df.replace({np.nan: None})
        
        # 1. Prepare Chart Logs (Latest 50, ascending)
        chart_df = df.tail(50).copy()
        chart_logs = chart_df.to_json(orient='records', date_format='iso')
        
        # 2. Prepare Table Logs (Latest 10, descending)
        table_df = df.tail(10).copy()
        if 'timestamp' in table_df.columns:
            table_df['timestamp'] = pd.to_datetime(table_df['timestamp']).dt.strftime('%d %b %Y, %I:%M %p')
        logs = table_df.sort_index(ascending=False).to_dict('records')
    
    data_drift = {}
    if os.path.exists(drift_report_path):
        with open(drift_report_path, "r") as f:
            data_drift = json.load(f)
            
    model_drift = {}
    if os.path.exists(model_drift_path):
        with open(model_drift_path, "r") as f:
            model_drift = json.load(f)
    
    return render(request, 'monitoring/dashboard.html', {
        'logs': logs,
        'chart_logs': chart_logs,
        'data_drift': data_drift,
        'model_drift': model_drift,
        'model_metrics': model_metrics,
        'top_performer': top_performer
    })

def drift_monitoring(request):
    import json
    drift_report_path = os.path.join(os.path.dirname(__file__), '../../reports/drift/data_drift_report.json')
    alerts_log_path = os.path.join(os.path.dirname(__file__), '../../logs/alerts.log')
    
    data_drift = {}
    if os.path.exists(drift_report_path):
        with open(drift_report_path, "r") as f:
            data_drift = json.load(f)
            
    alerts = []
    if os.path.exists(alerts_log_path):
        with open(alerts_log_path, "r") as f:
            alerts = f.readlines()
            alerts.reverse()
            alerts = [a.strip() for a in alerts[:20]]

    return render(request, 'monitoring/drift.html', {
        'data_drift': data_drift,
        'alerts': alerts
    })

def prediction_ui(request):
    log_path = os.path.join(os.path.dirname(__file__), '../../logs/predictions.parquet')
    drift_report_path = os.path.join(os.path.dirname(__file__), '../../reports/drift/data_drift_report.json')
    
    history = []
    chart_history = "[]"
    data_drift = {}
    
    if os.path.exists(drift_report_path):
        with open(drift_report_path, "r") as f:
            data_drift = json.load(f)

    if os.path.exists(log_path):
        df = pd.read_parquet(log_path)
        
        # Nullify any values below 1000 as they are likely raw normalized returns
        for col in ['arima_price', 'lr_price', 'lstm_price', 'rnn_price']:
            if col in df.columns:
                df.loc[df[col] < 1000, col] = np.nan
                
        # Handle NaN values for JSON and template safety
        df = df.replace({np.nan: None})
        
        # Prepare chart history (last 50 rows) for the candlestick/line chart
        chart_df = df.tail(50).copy()
        chart_history = chart_df.to_json(orient='records', date_format='iso')

        # Ensure timestamp is datetime and format it
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d %b %Y, %I:%M %p')
        
        # Format forecast_timestamp if it exists
        if 'forecast_timestamp' in df.columns:
            df['forecast_timestamp'] = pd.to_datetime(df['forecast_timestamp']).dt.strftime('%d %b %Y, %I:%M %p')

        # Convert remaining datetime columns to string for JSON safety
        for col in df.select_dtypes(include=['datetime', 'datetimetz']).columns:
            df[col] = df[col].astype(str)
        
        # Sort by index descending (latest first)
        history = df.sort_index(ascending=False).head(10).to_dict('records')
        
    return render(request, 'monitoring/prediction.html', {
        'history': history,
        'chart_history': chart_history,
        'data_drift': data_drift,
        'api_url': 'http://127.0.0.1:8001/trigger'
    })

