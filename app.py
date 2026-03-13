from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.utils
import google.generativeai as genai
import os
import json
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)

# -------- GEMINI SETUP -------- #
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# -------- HTML TEMPLATE -------- #
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Data Analyst Pro</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
            text-align: center;
            font-size: 2.5em;
        }
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            background: #e8eaff;
            border-color: #764ba2;
        }
        .upload-icon {
            font-size: 48px;
            color: #667eea;
            margin-bottom: 10px;
        }
        input[type="file"] {
            display: none;
        }
        .prompt-input {
            width: 100%;
            padding: 15px;
            margin: 20px 0;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        .prompt-input:focus {
            outline: none;
            border-color: #667eea;
        }
        .analyze-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 10px;
            font-size: 18px;
            cursor: pointer;
            transition: transform 0.2s;
            display: block;
            margin: 0 auto;
        }
        .analyze-btn:hover {
            transform: translateY(-2px);
        }
        .results {
            margin-top: 40px;
            padding: 20px;
            border-radius: 10px;
            background: #f8f9ff;
        }
        .chart-container {
            margin-top: 30px;
        }
        .summary-table {
            overflow-x: auto;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            color: #dc3545;
            padding: 10px;
            border-radius: 5px;
            background: #ffe6e6;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Data Analyst Pro</h1>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('file').click()">
                <div class="upload-icon">📁</div>
                <p>Drag & drop or click to upload CSV/Excel file</p>
                <input type="file" id="file" name="file" accept=".csv,.xlsx,.xls" required>
            </div>
            
            <input type="text" 
                   class="prompt-input" 
                   name="prompt" 
                   placeholder="What would you like to analyze? (e.g., 'Show sales trends', 'Correlation analysis', 'Summary statistics')"
                   required>
            
            <button type="submit" class="analyze-btn">Analyze Data</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="loading-spinner"></div>
            <p style="margin-top: 20px;">Analyzing your data...</p>
        </div>
        
        <div class="results" id="results"></div>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('file', document.getElementById('file').files[0]);
            formData.append('prompt', document.querySelector('.prompt-input').value);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    document.getElementById('results').innerHTML = `<div class="error">${data.error}</div>`;
                    return;
                }
                
                let html = '<h2>📊 Analysis Results</h2>';
                
                if (data.summary) {
                    html += '<div class="summary-table"><h3>Statistical Summary</h3>' + data.summary + '</div>';
                }
                
                if (data.chart) {
                    html += '<div class="chart-container"><h3>Visualization</h3>' + data.chart + '</div>';
                }
                
                if (data.insights) {
                    html += '<div class="insights"><h3>💡 Key Insights</h3><p>' + data.insights + '</p></div>';
                }
                
                document.getElementById('results').innerHTML = html;
                
                // Re-execute Plotly scripts
                if (data.chart && data.chart.includes('plotly')) {
                    eval(document.querySelector('#results script').innerHTML);
                }
                
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('results').innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        };
    </script>
</body>
</html>
"""

# -------- ENHANCED DATA CLEANING -------- #
def clean_data(df):
    """Advanced data cleaning with multiple strategies"""
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values intelligently
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # For numeric columns: use median for skewed, mean for normal
            if df[col].skew() > 1:  # skewed data
                df[col] = df[col].fillna(df[col].median())
            else:  # normal distribution
                df[col] = df[col].fillna(df[col].mean())
        else:
            # For categorical: use mode or 'Unknown'
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna('Unknown')
    
    # Convert data types
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    return df

# -------- OUTLIER DETECTION WITH MULTIPLE METHODS -------- #
def detect_outliers(df, method='iqr'):
    """Detect outliers using multiple methods"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_report = {}
    
    if method == 'iqr':
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_report[col] = len(outliers)
    
    elif method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > 3]
            outlier_report[col] = len(outliers)
    
    return outlier_report

# -------- GEMINI PROMPT UNDERSTANDING -------- #
def interpret_prompt(prompt, df_info):
    """Enhanced prompt interpretation with context"""
    
    context = f"""
    Dataset Information:
    - Columns: {', '.join(df_info['columns'])}
    - Total rows: {df_info['rows']}
    - Numeric columns: {', '.join(df_info['numeric_cols'])}
    - Categorical columns: {', '.join(df_info['categorical_cols'])}
    
    User Request: "{prompt}"
    
    Analyze the request and return a JSON with:
    1. analysis_type: chart/summary/correlation/trend/comparison
    2. columns: relevant columns for analysis
    3. chart_type: if chart, specify (bar/line/scatter/histogram/box)
    4. insight_required: brief description of what insights to extract
    """
    
    try:
        response = model.generate_content(context)
        # Try to parse JSON from response
        result = json.loads(response.text)
    except:
        # Fallback to simple parsing
        result = {
            "analysis_type": "summary",
            "columns": [],
            "chart_type": "bar",
            "insight_required": "basic statistics"
        }
    
    return result

# -------- GENERATE INSIGHTS WITH GEMINI -------- #
def generate_insights(df, analysis_result):
    """Generate natural language insights from analysis"""
    
    summary_stats = df.describe().to_string()
    correlation = df.corr(numeric_only=True).to_string()
    
    context = f"""
    Dataset Analysis Results:
    
    Summary Statistics:
    {summary_stats}
    
    Correlations:
    {correlation}
    
    Based on this data, provide:
    1. Key patterns and trends
    2. Important correlations
    3. Anomalies or outliers
    4. Recommendations
    
    Keep it concise and business-friendly.
    """
    
    try:
        response = model.generate_content(context)
        return response.text
    except:
        return "Insights generation temporarily unavailable."

# -------- HOME PAGE -------- #
@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

# -------- ENHANCED ANALYSIS ENDPOINT -------- #
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files["file"]
        prompt = request.form["prompt"]
        
        # Read dataset with encoding detection
        if file.filename.endswith(".csv"):
            # Try different encodings
            try:
                df = pd.read_csv(file)
            except:
                file.seek(0)
                df = pd.read_csv(file, encoding='latin1')
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file type. Please upload CSV or Excel files."})
        
        # Data cleaning
        original_shape = df.shape
        df = clean_data(df)
        cleaned_shape = df.shape
        
        # Detect outliers
        outliers = detect_outliers(df)
        
        # Dataset info for AI
        df_info = {
            "columns": list(df.columns),
            "rows": len(df),
            "numeric_cols": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_cols": list(df.select_dtypes(include=['object']).columns)
        }
        
        # Interpret prompt
        task = interpret_prompt(prompt, df_info)
        analysis_type = task.get("analysis_type", "summary")
        
        response_data = {}
        
        # Generate insights
        insights = generate_insights(df, task)
        response_data["insights"] = insights
        
        # SUMMARY
        if analysis_type in ["summary", "trend"]:
            summary = df.describe(include='all').to_html(classes='summary-table')
            response_data["summary"] = summary
            
            # Add data quality info
            quality_html = f"""
            <div style="margin-top: 20px; padding: 15px; background: #e3f2fd; border-radius: 5px;">
                <h4>📊 Data Quality Report</h4>
                <p>Original rows: {original_shape[0]} | After cleaning: {cleaned_shape[0]}</p>
                <p>Original columns: {original_shape[1]} | After cleaning: {cleaned_shape[1]}</p>
                <p>Outliers detected: {sum(outliers.values())}</p>
            </div>
            """
            response_data["summary"] = quality_html + response_data.get("summary", "")
        
        # CORRELATION
        elif analysis_type == "correlation":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                fig = px.imshow(corr, 
                              title="Correlation Matrix",
                              color_continuous_scale='RdBu',
                              aspect="auto")
                fig.update_layout(height=600)
                response_data["chart"] = fig.to_html()
                
                # Add correlation insights
                strong_corr = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        if abs(corr.iloc[i, j]) > 0.7:
                            strong_corr.append(f"{corr.columns[i]} & {corr.columns[j]}: {corr.iloc[i, j]:.2f}")
                
                if strong_corr:
                    response_data["insights"] += "\n\nStrong Correlations Found:\n" + "\n".join(strong_corr)
            else:
                response_data["error"] = "Need at least 2 numeric columns for correlation"
        
        # CHART
        elif analysis_type == "chart":
            chart_type = task.get("chart_type", "bar")
            cols = task.get("columns", [])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if not cols and len(numeric_cols) >= 2:
                cols = [numeric_cols[0], numeric_cols[1]]
            
            if len(cols) >= 2:
                if chart_type == "line":
                    fig = px.line(df, x=cols[0], y=cols[1:], title=f"{cols[0]} vs {', '.join(cols[1:])}")
                elif chart_type == "scatter":
                    fig = px.scatter(df, x=cols[0], y=cols[1], title=f"{cols[0]} vs {cols[1]}")
                elif chart_type == "histogram":
                    fig = px.histogram(df, x=cols[0], title=f"Distribution of {cols[0]}")
                elif chart_type == "box":
                    fig = px.box(df, y=cols[0], title=f"Box Plot of {cols[0]}")
                else:  # bar chart
                    fig = px.bar(df, x=cols[0], y=cols[1] if len(cols) > 1 else None, 
                               title=f"{cols[0]} Analysis")
                
                fig.update_layout(
                    template="plotly_white",
                    hovermode='x unified',
                    height=500
                )
                response_data["chart"] = fig.to_html()
            else:
                response_data["error"] = "Insufficient columns for chart"
        
        # COMPARISON
        elif analysis_type == "comparison":
            categorical_cols = df_info['categorical_cols']
            numeric_cols = df_info['numeric_cols']
            
            if categorical_cols and numeric_cols:
                group_col = categorical_cols[0]
                value_col = numeric_cols[0]
                
                grouped = df.groupby(group_col)[value_col].agg(['mean', 'sum', 'count']).reset_index()
                
                fig = px.bar(grouped, x=group_col, y='mean', 
                           title=f"Average {value_col} by {group_col}")
                response_data["chart"] = fig.to_html()
                response_data["summary"] = grouped.to_html()
            else:
                response_data["error"] = "Need both categorical and numeric columns for comparison"
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"error": str(e)})

# -------- API ENDPOINT FOR DIRECT ANALYSIS -------- #
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """REST API endpoint for programmatic access"""
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        prompt = data.get('prompt', 'analyze this data')
        
        df = clean_data(df)
        
        # Similar analysis logic as above but returns JSON
        # ... (simplified for API)
        
        return jsonify({"status": "success", "data": df.to_dict()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
