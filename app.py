import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.utils
import json
from flask import Flask, request, render_template_string, jsonify, send_file
from werkzeug.utils import secure_filename
import google.generativeai as genai
from datetime import datetime
import io
import re
import traceback
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Gemini Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None

# Config
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variables
cleaned_df = None
original_df = None
current_changes = {}

# ==================== HTML TEMPLATE ====================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Data Analyst Pro</title>
    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            color: white;
            padding: 20px;
        }
        
        .container { max-width: 1400px; margin: 0 auto; }
        
        .header {
            text-align: center;
            padding: 30px;
            background: rgba(0,0,0,0.3);
            border-radius: 20px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }
        
        .stat-value { font-size: 2rem; font-weight: bold; color: #00d2ff; }
        .stat-label { color: #aaa; margin-top: 5px; font-size: 0.9rem; }
        
        .upload-area {
            background: rgba(255,255,255,0.1);
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 30px;
        }
        
        .upload-area:hover {
            border-color: #00d2ff;
            background: rgba(0,210,255,0.1);
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .tab-btn {
            padding: 10px 25px;
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 10px;
            color: white;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .tab-btn.active {
            background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .log-table {
            width: 100%;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            overflow-x: auto;
        }
        
        .log-table th, .log-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .log-table th { background: rgba(0,210,255,0.2); color: #00d2ff; }
        
        .chart-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .chart-btn {
            padding: 8px 20px;
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
        }
        
        .chart-btn:hover { background: #00d2ff; }
        
        .chat-container {
            background: rgba(0,0,0,0.3);
            border-radius: 20px;
            height: 450px;
            display: flex;
            flex-direction: column;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message { margin-bottom: 15px; display: flex; }
        .message.user { justify-content: flex-end; }
        .message.ai { justify-content: flex-start; }
        
        .message-content {
            max-width: 70%;
            padding: 10px 15px;
            border-radius: 15px;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        }
        
        .message.ai .message-content {
            background: rgba(255,255,255,0.1);
        }
        
        .chat-input {
            display: flex;
            padding: 15px;
            gap: 10px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        .chat-input input {
            flex: 1;
            padding: 12px;
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 10px;
            color: white;
        }
        
        .chat-input button {
            padding: 12px 25px;
            background: linear-gradient(135deg, #00d2ff, #3a7bd5);
            border: none;
            border-radius: 10px;
            color: white;
            cursor: pointer;
        }
        
        .progress-container { display: none; margin-bottom: 20px; }
        .progress-bar {
            width: 100%;
            height: 10px;
            background: rgba(255,255,255,0.2);
            border-radius: 5px;
            overflow: hidden;
        }
        .progress-fill {
            width: 0%;
            height: 100%;
            background: linear-gradient(135deg, #00d2ff, #3a7bd5);
            transition: width 0.3s;
        }
        
        .download-btn {
            padding: 10px 25px;
            background: rgba(72, 187, 120, 0.2);
            border: 1px solid #48bb78;
            border-radius: 10px;
            color: #48bb78;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            margin-top: 15px;
        }
        
        @media (max-width: 768px) {
            .header h1 { font-size: 1.5rem; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Data Analyst Pro</h1>
            <p>Upload CSV/Excel • Auto Cleaning • AI Insights • Smart Visualizations</p>
        </div>
        
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card"><div class="stat-value" id="rowCount">-</div><div class="stat-label">Total Rows</div></div>
            <div class="stat-card"><div class="stat-value" id="colCount">-</div><div class="stat-label">Total Columns</div></div>
            <div class="stat-card"><div class="stat-value" id="qualityScore">-</div><div class="stat-label">Quality Score</div></div>
            <div class="stat-card"><div class="stat-value" id="accuracyScore">-</div><div class="stat-label">Accuracy Score</div></div>
        </div>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div style="font-size: 48px;">📁</div>
            <h3>Click to Upload CSV or Excel File</h3>
            <p>Auto-cleaning • Outlier Detection • Missing Value Handling • AI Analysis</p>
            <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display:none">
        </div>
        
        <div class="progress-container" id="progressContainer">
            <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
            <p id="progressText" style="margin-top: 10px;">Processing...</p>
        </div>
        
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('changeLog')">📋 Change Log</button>
            <button class="tab-btn" onclick="showTab('visualizations')">📊 Visualizations</button>
            <button class="tab-btn" onclick="showTab('accuracy')">🎯 Accuracy Metrics</button>
            <button class="tab-btn" onclick="showTab('aiChat')">🤖 AI Assistant</button>
        </div>
        
        <div id="changeLog" class="tab-content active">
            <div id="changeLogContent" style="overflow-x: auto;"></div>
            <div class="download-buttons" id="downloadButtons" style="display: none;">
                <a href="/download/csv" class="download-btn">📥 Download as CSV</a>
                <a href="/download/excel" class="download-btn">📥 Download as Excel</a>
            </div>
        </div>
        
        <div id="visualizations" class="tab-content">
            <div class="chart-selector">
                <button class="chart-btn" onclick="createChart('bar')">📊 Bar Chart</button>
                <button class="chart-btn" onclick="createChart('pie')">🥧 Pie Chart</button>
                <button class="chart-btn" onclick="createChart('line')">📈 Line Chart</button>
                <button class="chart-btn" onclick="createChart('heatmap')">🔥 Heatmap</button>
                <button class="chart-btn" onclick="createChart('scatter')">✨ Scatter Plot</button>
                <button class="chart-btn" onclick="createChart('box')">📦 Box Plot</button>
            </div>
            <div id="chartContainer" style="min-height: 500px;"></div>
        </div>
        
        <div id="accuracy" class="tab-content">
            <div id="accuracyContent"></div>
        </div>
        
        <div id="aiChat" class="tab-content">
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message ai"><div class="message-content">👋 Hello! I'm your AI Data Analyst. Ask me anything about your data!</div></div>
                </div>
                <div class="chat-input">
                    <input type="text" id="chatInput" placeholder="Ask me... e.g., 'Show me summary', 'Any issues in data?'">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentData = null;
        
        document.getElementById('fileInput').addEventListener('change', async function(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            document.getElementById('progressContainer').style.display = 'block';
            document.getElementById('progressFill').style.width = '30%';
            document.getElementById('progressText').innerText = 'Uploading file...';
            
            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const text = await response.text();
                const data = JSON.parse(text);
                
                document.getElementById('progressFill').style.width = '100%';
                document.getElementById('progressText').innerText = 'Analysis complete!';
                setTimeout(() => {
                    document.getElementById('progressContainer').style.display = 'none';
                }, 2000);
                
                currentData = data;
                updateStats(data);
                displayChangeLog(data);
                displayAccuracyMetrics(data);
                document.getElementById('downloadButtons').style.display = 'block';
                
            } catch(error) {
                alert('Error: ' + error.message);
                document.getElementById('progressContainer').style.display = 'none';
            }
        });
        
        function updateStats(data) {
            document.getElementById('rowCount').innerText = data.final_shape?.[0] || '-';
            document.getElementById('colCount').innerText = data.final_shape?.[1] || '-';
            document.getElementById('qualityScore').innerText = data.quality_score || '-';
            document.getElementById('accuracyScore').innerText = data.accuracy_score || '-';
        }
        
        function displayChangeLog(data) {
            let html = '<table class="log-table"><thead><tr><th>Operation</th><th>Details</th><th>Impact</th></tr></thead><tbody>';
            
            html += `<tr><td>📊 Original Data</td><td>${data.original_shape?.[0] || 0} rows, ${data.original_shape?.[1] || 0} columns</td><td>-</td></tr>`;
            html += `<tr><td>🗑️ Duplicates Removed</td><td>${data.duplicates_removed || 0} duplicate rows</td><td>${data.duplicates_removed ? 'Removed' : 'No duplicates'}</td></tr>`;
            
            if (data.missing_values) {
                for (const [col, val] of Object.entries(data.missing_values)) {
                    html += `<tr><td>🔧 Missing Values</td><td>Column '${col}': ${val}</td><td>Filled with median/mode</td></tr>`;
                }
            }
            
            if (data.outliers) {
                for (const [col, val] of Object.entries(data.outliers)) {
                    html += `<tr><td>⚠️ Outliers Removed</td><td>Column '${col}': ${val} outliers</td><td>Removed using IQR</td></tr>`;
                }
            }
            
            html += `<tr><td>✅ Final Data</td><td>${data.final_shape?.[0] || 0} rows, ${data.final_shape?.[1] || 0} columns</td><td>Clean & Ready</td></tr>`;
            html += '</tbody></table>';
            
            document.getElementById('changeLogContent').innerHTML = html;
        }
        
        function displayAccuracyMetrics(data) {
            if (!data.accuracy_metrics || Object.keys(data.accuracy_metrics).length === 0) {
                document.getElementById('accuracyContent').innerHTML = '<div class="stat-card"><p>No accuracy metrics available. Upload data with numeric columns to see ML metrics.</p></div>';
                return;
            }
            
            let html = '<div class="stats-grid">';
            for (const [key, value] of Object.entries(data.accuracy_metrics)) {
                let displayValue = typeof value === 'number' ? value.toFixed(4) : value;
                if (displayValue.length > 30) displayValue = displayValue.substring(0, 30) + '...';
                html += `<div class="stat-card"><div class="stat-value">${displayValue}</div><div class="stat-label">${key}</div></div>`;
            }
            html += '</div>';
            document.getElementById('accuracyContent').innerHTML = html;
        }
        
        async function createChart(type) {
            const response = await fetch('/chart', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chart_type: type })
            });
            const data = await response.json();
            if (data.chart) {
                const chartData = JSON.parse(data.chart);
                Plotly.newPlot('chartContainer', chartData.data, chartData.layout);
            } else if (data.error) {
                alert(data.error);
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            const chatDiv = document.getElementById('chatMessages');
            chatDiv.innerHTML += `<div class="message user"><div class="message-content">${escapeHtml(message)}</div></div>`;
            input.value = '';
            chatDiv.scrollTop = chatDiv.scrollHeight;
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                const data = await response.json();
                chatDiv.innerHTML += `<div class="message ai"><div class="message-content">${escapeHtml(data.response)}</div></div>`;
                chatDiv.scrollTop = chatDiv.scrollHeight;
            } catch(error) {
                chatDiv.innerHTML += `<div class="message ai"><div class="message-content">Error: ${error.message}</div></div>`;
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
'''

# ==================== DATA CLEANING ====================
def clean_data_advanced(df):
    changes = {
        "original_shape": df.shape,
        "duplicates_removed": 0,
        "missing_values": {},
        "outliers": {}
    }
    
    before = len(df)
    df = df.drop_duplicates()
    changes["duplicates_removed"] = before - len(df)
    
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
                changes["missing_values"][col] = f"{missing} (filled with median)"
            else:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                changes["missing_values"][col] = f"{missing} (filled with '{mode_val}')"
    
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if len(outliers) > 0:
            changes["outliers"][col] = len(outliers)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    changes["final_shape"] = df.shape
    
    score = 100
    total_missing = sum(df[col].isnull().sum() for col in df.columns)
    if total_missing > 0:
        score -= min(30, (total_missing / (df.shape[0] * df.shape[1])) * 100)
    if changes["duplicates_removed"] > 0:
        score -= min(10, (changes["duplicates_removed"] / changes["original_shape"][0]) * 50)
    changes["quality_score"] = max(0, int(score))
    
    return df, changes

def calculate_accuracy_metrics(df):
    metrics = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        target = numeric_cols[-1]
        features = numeric_cols[:-1]
        X = df[features].fillna(df[features].mean())
        y = df[target].fillna(df[target].mean())
        
        if len(X) > 10 and len(features) > 0:
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics["R² Score"] = r2_score(y_test, y_pred)
                metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
            except:
                pass
    
    return metrics

def generate_chart(df, chart_type):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if chart_type == 'bar' and categorical_cols and numeric_cols:
        fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
    elif chart_type == 'pie' and categorical_cols:
        fig = px.pie(df, names=categorical_cols[0])
    elif chart_type == 'line' and len(numeric_cols) >= 2:
        fig = px.line(df, x=numeric_cols[0], y=numeric_cols[1])
    elif chart_type == 'heatmap' and len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto")
    elif chart_type == 'scatter' and len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
    elif chart_type == 'box' and numeric_cols:
        fig = px.box(df, y=numeric_cols[0])
    else:
        return None
    
    return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

def gemini_response(df, user_message, changes):
    if not model:
        return "Gemini API key not configured. Please add GEMINI_API_KEY environment variable."
    
    context = f"""
    Dataset: {df.shape[0]} rows, {df.shape[1]} columns
    Columns: {list(df.columns)}
    Cleaning: Removed {changes.get('duplicates_removed',0)} duplicates
    Quality Score: {changes.get('quality_score',0)}/100
    
    User: {user_message}
    Respond helpfully about this data.
    """
    try:
        response = model.generate_content(context)
        return response.text[:1000]
    except:
        return "AI analysis temporarily unavailable."

# ==================== FLASK ROUTES ====================
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload():
    global cleaned_df, original_df, current_changes
    
    try:
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"})
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, nrows=100000)
        else:
            df = pd.read_excel(file, nrows=100000)
        
        original_df = df.copy()
        cleaned_df, changes = clean_data_advanced(df)
        current_changes = changes
        
        # Calculate accuracy
        accuracy_metrics = calculate_accuracy_metrics(cleaned_df)
        changes['accuracy_metrics'] = accuracy_metrics
        
        if accuracy_metrics and 'R² Score' in accuracy_metrics:
            changes['accuracy_score'] = round(accuracy_metrics['R² Score'], 4)
        else:
            changes['accuracy_score'] = changes.get('quality_score', 85)
        
        return jsonify(changes)
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/chart', methods=['POST'])
def chart():
    global cleaned_df
    if cleaned_df is None:
        return jsonify({"error": "No data available"})
    
    data = request.json
    chart_type = data.get('chart_type', 'bar')
    chart_json = generate_chart(cleaned_df, chart_type)
    
    if chart_json:
        return jsonify({"chart": chart_json})
    return jsonify({"error": "Cannot create chart with current data"})

@app.route('/chat', methods=['POST'])
def chat():
    global cleaned_df, current_changes
    if cleaned_df is None:
        return jsonify({"response": "Please upload data first."})
    
    data = request.json
    message = data.get('message', '')
    response = gemini_response(cleaned_df, message, current_changes)
    return jsonify({"response": response})

@app.route('/download/<format>')
def download(format):
    global cleaned_df
    if cleaned_df is None:
        return "No data available", 404
    
    if format == 'csv':
        output = io.BytesIO()
        cleaned_df.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='cleaned_data.csv')
    elif format == 'excel':
        output = io.BytesIO()
        cleaned_df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='cleaned_data.xlsx')
    
    return "Invalid format", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
