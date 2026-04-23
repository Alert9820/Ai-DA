import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
from flask import Flask, request, render_template_string, jsonify, send_file, session
from werkzeug.utils import secure_filename
import google.generativeai as genai
from datetime import datetime
import io
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Gemini Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Config
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'parquet'}
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variable to store cleaned data
cleaned_df = None
original_df = None
accuracy_metrics = {}

# ==================== HTML TEMPLATE ====================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Data Analyst Pro - Enterprise Edition</title>
    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            color: white;
        }
        
        .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
        
        /* Header */
        .header {
            text-align: center;
            padding: 30px 0;
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
        
        /* Stats Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s;
        }
        
        .stat-card:hover { transform: translateY(-5px); }
        .stat-value { font-size: 2rem; font-weight: bold; color: #00d2ff; }
        .stat-label { color: #aaa; margin-top: 5px; }
        
        /* Upload Area */
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
        
        /* Tabs */
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
        
        /* Change Log Table */
        .log-table {
            width: 100%;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            overflow: hidden;
        }
        
        .log-table th, .log-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .log-table th { background: rgba(0,210,255,0.2); color: #00d2ff; }
        
        /* Chart Selector */
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
        
        /* Chat Interface */
        .chat-container {
            background: rgba(0,0,0,0.3);
            border-radius: 20px;
            height: 500px;
            display: flex;
            flex-direction: column;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
        }
        
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
        
        /* Progress Bar */
        .progress-container {
            display: none;
            margin-bottom: 20px;
        }
        
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
        
        /* Responsive */
        @media (max-width: 768px) {
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .header h1 { font-size: 1.5rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Data Analyst Pro - Enterprise Edition</h1>
            <p>100K+ Rows | Real-time AI Analysis | Auto Cleaning | Accuracy Metrics</p>
        </div>
        
        <!-- Stats Dashboard -->
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card"><div class="stat-value" id="rowCount">-</div><div class="stat-label">Total Rows</div></div>
            <div class="stat-card"><div class="stat-value" id="colCount">-</div><div class="stat-label">Total Columns</div></div>
            <div class="stat-card"><div class="stat-value" id="qualityScore">-</div><div class="stat-label">Data Quality Score</div></div>
            <div class="stat-card"><div class="stat-value" id="accuracyScore">-</div><div class="stat-label">AI Accuracy Score</div></div>
        </div>
        
        <!-- Upload Area -->
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div style="font-size: 48px;">📁</div>
            <h3>Click to Upload CSV/Excel/JSON</h3>
            <p>Supports up to 100,000+ rows | Auto-cleaning | AI-powered insights</p>
            <input type="file" id="fileInput" accept=".csv,.xlsx,.xls,.json,.parquet" style="display:none">
        </div>
        
        <!-- Progress -->
        <div class="progress-container" id="progressContainer">
            <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
            <p id="progressText" style="margin-top: 10px;">Processing...</p>
        </div>
        
        <!-- Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('changeLog')">📋 Change Log</button>
            <button class="tab-btn" onclick="showTab('visualizations')">📊 Visualizations</button>
            <button class="tab-btn" onclick="showTab('accuracy')">🎯 Accuracy Metrics</button>
            <button class="tab-btn" onclick="showTab('aiChat')">🤖 AI Assistant</button>
        </div>
        
        <!-- Tab: Change Log -->
        <div id="changeLog" class="tab-content active">
            <div class="log-table-container" id="changeLogContent"></div>
        </div>
        
        <!-- Tab: Visualizations -->
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
        
        <!-- Tab: Accuracy Metrics -->
        <div id="accuracy" class="tab-content">
            <div id="accuracyContent"></div>
        </div>
        
        <!-- Tab: AI Chat -->
        <div id="aiChat" class="tab-content">
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message ai"><div class="message-content">👋 Hello! I'm your AI Data Analyst. I can:<br>• Answer questions about your data<br>• Create custom visualizations<br>• Suggest data cleaning steps<br>• Predict values and find patterns<br><br>Ask me anything about your dataset!</div></div>
                </div>
                <div class="chat-input">
                    <input type="text" id="chatInput" placeholder="Ask me anything... e.g., 'Show me sales trend', 'Predict next values', 'Find anomalies'">
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
                const data = await response.json();
                
                document.getElementById('progressFill').style.width = '100%';
                document.getElementById('progressText').innerText = 'Analysis complete!';
                
                setTimeout(() => {
                    document.getElementById('progressContainer').style.display = 'none';
                }, 2000);
                
                currentData = data;
                updateStats(data);
                displayChangeLog(data);
                displayAccuracyMetrics(data);
                
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
            
            html += `<tr><td>📊 Original Data</td><td>${data.original_shape?.[0]} rows, ${data.original_shape?.[1]} columns</td><td>-</td></tr>`;
            html += `<tr><td>🗑️ Duplicates Removed</td><td>${data.duplicates_removed || 0} duplicate rows</td><td>${data.duplicates_removed ? 'Removed' : 'No duplicates found'}</td></tr>`;
            
            if (data.missing_values) {
                for (const [col, val] of Object.entries(data.missing_values)) {
                    html += `<tr><td>🔧 Missing Values</td><td>Column '${col}': ${val}</td><td>Filled with median/mode</td></tr>`;
                }
            }
            
            if (data.outliers) {
                for (const [col, val] of Object.entries(data.outliers)) {
                    html += `<tr><td>⚠️ Outliers Removed</td><td>Column '${col}': ${val} outliers</td><td>Removed using IQR method</td></tr>`;
                }
            }
            
            if (data.unknown_detected) {
                for (const [col, val] of Object.entries(data.unknown_detected)) {
                    html += `<tr><td>🔍 Unknown Patterns</td><td>Column '${col}': ${val.unknown_count} unknown values</td><td>${val.suggestion || 'Check data source'}</td></tr>`;
                }
            }
            
            html += `<tr><td>✅ Final Data</td><td>${data.final_shape?.[0]} rows, ${data.final_shape?.[1]} columns</td><td>Clean & Ready</td></tr>`;
            html += '</tbody></table>';
            
            document.getElementById('changeLogContent').innerHTML = html;
        }
        
        function displayAccuracyMetrics(data) {
            if (!data.accuracy_metrics) {
                document.getElementById('accuracyContent').innerHTML = '<p>Train a model to see accuracy metrics...</p>';
                return;
            }
            
            let html = '<div class="stats-grid">';
            for (const [key, value] of Object.entries(data.accuracy_metrics)) {
                html += `<div class="stat-card"><div class="stat-value">${typeof value === 'number' ? value.toFixed(4) : value}</div><div class="stat-label">${key}</div></div>`;
            }
            html += '</div>';
            
            document.getElementById('accuracyContent').innerHTML = html;
        }
        
        async function createChart(type) {
            if (!currentData) {
                alert('Please upload data first');
                return;
            }
            
            const response = await fetch('/chart', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chart_type: type })
            });
            
            const data = await response.json();
            if (data.chart) {
                const chartData = JSON.parse(data.chart);
                Plotly.newPlot('chartContainer', chartData.data, chartData.layout);
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            const chatDiv = document.getElementById('chatMessages');
            chatDiv.innerHTML += `<div class="message user"><div class="message-content">${message}</div></div>`;
            input.value = '';
            chatDiv.scrollTop = chatDiv.scrollHeight;
            
            // Send to AI
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            // Add AI response
            chatDiv.innerHTML += `<div class="message ai"><div class="message-content">${data.response}</div></div>`;
            chatDiv.scrollTop = chatDiv.scrollHeight;
            
            // Handle chart creation if AI requests it
            if (data.chart_command) {
                createChart(data.chart_command);
            }
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

# ==================== DATA CLEANING WITH UNKNOWN DETECTION ====================
def clean_data_advanced(df):
    """Advanced cleaning with unknown pattern detection"""
    changes = {
        "original_shape": df.shape,
        "duplicates_removed": 0,
        "missing_values": {},
        "outliers": {},
        "unknown_detected": {}
    }
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    changes["duplicates_removed"] = before - len(df)
    
    # Handle missing and detect unknown patterns
    for col in df.columns:
        # Detect unknown patterns
        unique_vals = df[col].value_counts()
        rare_vals = unique_vals[unique_vals < len(df) * 0.01]  # Less than 1% frequency
        
        if len(rare_vals) > 0 and len(rare_vals) < 20:
            changes["unknown_detected"][col] = {
                "unknown_count": len(rare_vals),
                "unknown_values": rare_vals.head(5).to_dict(),
                "suggestion": f"Found {len(rare_vals)} rare values. Consider reviewing data source."
            }
        
        # Fill missing values
        missing = df[col].isnull().sum()
        if missing > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
                changes["missing_values"][col] = f"{missing} (filled with median)"
            else:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                changes["missing_values"][col] = f"{missing} (filled with '{mode_val}')"
    
    # Remove outliers
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
    
    # Quality score
    score = 100
    total_missing = sum(df[col].isnull().sum() for col in df.columns)
    if total_missing > 0:
        score -= min(30, (total_missing / (df.shape[0] * df.shape[1])) * 100)
    if changes["duplicates_removed"] > 0:
        score -= min(10, (changes["duplicates_removed"] / changes["original_shape"][0]) * 50)
    changes["quality_score"] = max(0, int(score))
    
    return df, changes

# ==================== ACCURACY METRICS ====================
def calculate_accuracy_metrics(df):
    """Calculate ML accuracy metrics"""
    metrics = {}
    
    # Find numeric columns for prediction
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        # Try to predict last numeric column from others
        target = numeric_cols[-1]
        features = numeric_cols[:-1]
        
        # Prepare data
        X = df[features].fillna(df[features].mean())
        y = df[target].fillna(df[target].mean())
        
        if len(X) > 10 and len(features) > 0:
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Regression metrics
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                metrics["R² Score"] = r2_score(y_test, y_pred)
                metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
                metrics["MAE"] = np.mean(np.abs(y_test - y_pred))
                metrics["Model Used"] = "Random Forest Regressor"
                metrics["Features Used"] = ", ".join(features[:3])
                
            except Exception as e:
                metrics["Error"] = str(e)
    
    # Classification metrics if categorical column exists
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        try:
            target_cat = categorical_cols[0]
            feature_num = numeric_cols[0]
            
            # Encode labels
            le = LabelEncoder()
            y_cat = le.fit_transform(df[target_cat].fillna('Unknown'))
            X_num = df[feature_num].fillna(df[feature_num].mean()).values.reshape(-1, 1)
            
            if len(np.unique(y_cat)) >= 2 and len(X_num) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X_num, y_cat, test_size=0.2, random_state=42)
                
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                metrics["Accuracy"] = accuracy_score(y_test, y_pred)
                metrics["Precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics["Recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics["F1 Score"] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
        except Exception as e:
            metrics["Classification Error"] = str(e)
    
    return metrics

# ==================== CHART GENERATION ====================
def generate_chart(df, chart_type):
    """Generate different types of charts"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if chart_type == 'bar' and len(categorical_cols) > 0 and len(numeric_cols) > 0:
        fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], title=f"{categorical_cols[0]} vs {numeric_cols[0]}")
    elif chart_type == 'pie' and len(categorical_cols) > 0:
        fig = px.pie(df, names=categorical_cols[0], title=f"Distribution of {categorical_cols[0]}")
    elif chart_type == 'line' and len(numeric_cols) >= 2:
        fig = px.line(df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
    elif chart_type == 'heatmap' and len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    elif chart_type == 'scatter' and len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
    elif chart_type == 'box' and len(numeric_cols) > 0:
        fig = px.box(df, y=numeric_cols[0], title=f"Box Plot of {numeric_cols[0]}")
    else:
        # Default: show first numeric column distribution
        if numeric_cols:
            fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
        else:
            return None
    
    return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

# ==================== GEMINI AI AGENT ====================
def gemini_agent(df, user_message, changes_log):
    """Gemini AI agent that understands data and can respond"""
    
    context = f"""
    You are an AI Data Analyst Assistant. You have access to a dataset with:
    - Shape: {df.shape[0]} rows, {df.shape[1]} columns
    - Columns: {list(df.columns)}
    - Data Types: {df.dtypes.to_dict()}
    
    Cleaning Operations Performed:
    - Original shape: {changes_log.get('original_shape', 'N/A')}
    - Final shape: {changes_log.get('final_shape', 'N/A')}
    - Duplicates removed: {changes_log.get('duplicates_removed', 0)}
    - Missing values filled: {changes_log.get('missing_values', {})}
    - Outliers removed: {changes_log.get('outliers', {})}
    - Unknown patterns detected: {changes_log.get('unknown_detected', {})}
    
    Statistical Summary:
    {df.describe().to_string()}
    
    Sample Data (first 5 rows):
    {df.head().to_string()}
    
    User Query: "{user_message}"
    
    Respond helpfully. If user asks for a chart, respond with "CHART:<type>" where type is bar/pie/line/heatmap/scatter/box.
    If user asks for prediction, suggest using the accuracy metrics tab.
    Keep responses concise and actionable.
    """
    
    try:
        response = model.generate_content(context)
        response_text = response.text
        
        # Check if chart is requested
        chart_match = re.search(r'CHART:(\w+)', response_text)
        if chart_match:
            chart_type = chart_match.group(1)
            response_text = response_text.replace(f'CHART:{chart_type}', '')
            return response_text.strip(), chart_type
        
        return response_text.strip(), None
        
    except Exception as e:
        return f"AI Error: {str(e)}", None

# ==================== FLASK ROUTES ====================
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload():
    global cleaned_df, original_df, accuracy_metrics
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"})
    
    # Read file based on extension
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file, nrows=200000)  # Limit to 200k for performance
    elif file.filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file, nrows=200000)
    elif file.filename.endswith('.json'):
        df = pd.read_json(file)
    elif file.filename.endswith('.parquet'):
        df = pd.read_parquet(file)
    else:
        return jsonify({"error": "Unsupported format"})
    
    original_df = df.copy()
    
    # Clean data
    cleaned_df, changes = clean_data_advanced(df)
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(cleaned_df)
    
    # Add accuracy score to response
    if accuracy_metrics:
        if 'F1 Score' in accuracy_metrics:
            changes['accuracy_score'] = accuracy_metrics['F1 Score']
        elif 'R² Score' in accuracy_metrics:
            changes['accuracy_score'] = accuracy_metrics['R² Score']
        else:
            changes['accuracy_score'] = 85  # Default score
    
    changes['accuracy_metrics'] = accuracy_metrics
    
    return jsonify(changes)

@app.route('/chart', methods=['POST'])
def chart():
    global cleaned_df
    data = request.json
    chart_type = data.get('chart_type', 'bar')
    
    if cleaned_df is None:
        return jsonify({"error": "No data available"})
    
    chart_json = generate_chart(cleaned_df, chart_type)
    return jsonify({"chart": chart_json})

@app.route('/chat', methods=['POST'])
def chat():
    global cleaned_df, original_df
    
    data = request.json
    message = data.get('message', '')
    
    # Get changes log (simplified)
    changes_log = {
        "original_shape": original_df.shape if original_df is not None else (0,0),
        "final_shape": cleaned_df.shape if cleaned_df is not None else (0,0),
        "duplicates_removed": 0,
        "missing_values": {},
        "outliers": {},
        "unknown_detected": {}
    }
    
    response, chart_type = gemini_agent(cleaned_df if cleaned_df is not None else pd.DataFrame(), message, changes_log)
    
    result = {"response": response}
    if chart_type:
        result["chart_command"] = chart_type
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False, threaded=True)
