import os
import json
import uuid
import threading
from datetime import datetime
from io import BytesIO, StringIO

from flask import Flask, request, render_template_string, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Gemini AI setup
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Store processing status (in-memory, for demo)
processing_status = {}

# ----------------------------------------------------------------------
# Data Cleaning Functions
# ----------------------------------------------------------------------
def auto_clean_data(df, log):
    """Automatically clean dataframe and record changes in log."""
    original_shape = df.shape
    log['initial_rows'] = original_shape[0]
    log['initial_cols'] = original_shape[1]

    # 1. Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        log['duplicates_removed'] = dup_count
    else:
        log['duplicates_removed'] = 0

    # 2. Missing values
    missing_before = df.isnull().sum().to_dict()
    log['missing_before'] = {k: int(v) for k, v in missing_before.items() if v > 0}
    log['missing_fill_method'] = {}

    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                log['missing_fill_method'][col] = f'filled with median ({median_val:.2f})'
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
                    log['missing_fill_method'][col] = f'filled with mode ({mode_val[0]})'
                else:
                    df[col].fillna('Unknown', inplace=True)
                    log['missing_fill_method'][col] = 'filled with "Unknown"'

    # 3. Outlier capping (IQR)
    outlier_log = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        if outlier_count > 0:
            df[col] = df[col].clip(lower, upper)
            outlier_log[col] = int(outlier_count)
    log['outliers_capped'] = outlier_log

    # 4. Data type conversion (datetime, numeric)
    type_changes = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try datetime
            try:
                df[col] = pd.to_datetime(df[col])
                type_changes[col] = 'datetime'
                continue
            except:
                pass
            # Try numeric
            try:
                df[col] = pd.to_numeric(df[col])
                type_changes[col] = 'numeric'
            except:
                pass
    log['type_changes'] = type_changes

    log['final_rows'] = df.shape[0]
    log['final_cols'] = df.shape[1]
    return df, log

# ----------------------------------------------------------------------
# Visualization Functions
# ----------------------------------------------------------------------
def generate_visualizations(df):
    """Return dict of Plotly HTML snippets."""
    charts = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect='auto',
                        color_continuous_scale='RdBu_r',
                        title='Correlation Heatmap')
        fig.update_layout(height=500, template='plotly_white')
        charts['correlation_heatmap'] = fig.to_html(full_html=False)

    # Distribution + box plots for first 4 numeric columns
    for col in numeric_cols[:4]:
        fig = px.histogram(df, x=col, marginal='box', title=f'Distribution of {col}')
        fig.update_layout(height=450, template='plotly_white')
        charts[f'distribution_{col}'] = fig.to_html(full_html=False)

    # Box plots for outliers (same columns)
    for col in numeric_cols[:4]:
        fig = px.box(df, y=col, title=f'Box Plot - {col}', points='outliers')
        fig.update_layout(height=400, template='plotly_white')
        charts[f'boxplot_{col}'] = fig.to_html(full_html=False)

    # Time series if date column exists
    if date_cols and numeric_cols:
        for date_col in date_cols[:1]:
            for num_col in numeric_cols[:2]:
                fig = px.line(df, x=date_col, y=num_col,
                              title=f'{num_col} over {date_col}')
                fig.update_layout(height=450, template='plotly_white')
                charts[f'timeseries_{date_col}_{num_col}'] = fig.to_html(full_html=False)

    return charts

# ----------------------------------------------------------------------
# Gemini AI Q&A
# ----------------------------------------------------------------------
def ask_gemini_about_data(df, log, user_question):
    """Get AI answer about data quality and cleaning."""
    cleaning_summary = f"""
    Initial: {log['initial_rows']} rows, {log['initial_cols']} cols.
    Duplicates removed: {log['duplicates_removed']}
    Missing before: {log.get('missing_before', {})}
    Missing filled: {log.get('missing_fill_method', {})}
    Outliers capped: {log.get('outliers_capped', {})}
    Type changes: {log.get('type_changes', {})}
    Final: {log['final_rows']} rows, {log['final_cols']} cols.
    """
    sample = df.head(5).to_string()
    prompt = f"""
    You are a data quality expert. Here is the automatic cleaning summary:
    {cleaning_summary}
    Column types: {df.dtypes.to_dict()}
    First 5 rows:
    {sample}
    User question: {user_question}
    Answer concisely and professionally.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI error: {str(e)}"

# ----------------------------------------------------------------------
# Report Generation (HTML & PDF)
# ----------------------------------------------------------------------
def generate_html_report(df, log, charts, ai_insights):
    """Generate complete HTML report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Cleaning Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .chart {{ margin: 30px 0; }}
        </style>
    </head>
    <body>
        <h1>📊 Automated Data Cleaning Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <h2>📋 Change Log</h2>
        <table>
            <tr><th>Metric</th><th>Details</th></tr>
            <tr><td>Initial rows</td><td>{log.get('initial_rows', 'N/A')}</td></tr>
            <tr><td>Initial columns</td><td>{log.get('initial_cols', 'N/A')}</td></tr>
            <tr><td>Duplicates removed</td><td>{log.get('duplicates_removed', 0)}</td></tr>
            <tr><td>Missing values (before)</td><td>{log.get('missing_before', {})}</td></tr>
            <tr><td>Missing values filled</td><td>{log.get('missing_fill_method', {})}</td></tr>
            <tr><td>Outliers capped</td><td>{log.get('outliers_capped', {})}</td></tr>
            <tr><td>Data type changes</td><td>{log.get('type_changes', {})}</td></tr>
            <tr><td>Final rows</td><td>{log.get('final_rows', 'N/A')}</td></tr>
            <tr><td>Final columns</td><td>{log.get('final_cols', 'N/A')}</td></tr>
        </table>
        <h2>🤖 AI Insights</h2>
        <p>{ai_insights}</p>
        <h2>📈 Visualizations</h2>
    """
    for name, chart_html in charts.items():
        html += f'<div class="chart"><h3>{name.replace("_", " ").title()}</h3>{chart_html}</div>'
    html += "</body></html>"
    return html

def generate_pdf_report(df, log, charts, ai_insights, filename='report.pdf'):
    """Generate PDF report using ReportLab."""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Automated Data Cleaning Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    data = [
        ['Metric', 'Details'],
        ['Initial rows', str(log.get('initial_rows', 'N/A'))],
        ['Initial columns', str(log.get('initial_cols', 'N/A'))],
        ['Duplicates removed', str(log.get('duplicates_removed', 0))],
        ['Missing before', str(log.get('missing_before', {}))],
        ['Missing filled', str(log.get('missing_fill_method', {}))],
        ['Outliers capped', str(log.get('outliers_capped', {}))],
        ['Type changes', str(log.get('type_changes', {}))],
        ['Final rows', str(log.get('final_rows', 'N/A'))],
        ['Final columns', str(log.get('final_cols', 'N/A'))],
    ]
    table = Table(data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),
                               ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                               ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                               ('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("AI Insights", styles['Heading2']))
    story.append(Paragraph(ai_insights, styles['Normal']))
    story.append(Paragraph("Note: Interactive charts are available in HTML report.", styles['Italic']))
    doc.build(story)
    return filename

# ----------------------------------------------------------------------
# Flask Routes
# ----------------------------------------------------------------------
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    session_id = str(uuid.uuid4())
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{secure_filename(file.filename)}")
    file.save(temp_path)

    # Start background processing
    thread = threading.Thread(target=process_file, args=(session_id, temp_path))
    thread.daemon = True
    thread.start()

    return jsonify({'session_id': session_id, 'status': 'processing'})

def process_file(session_id, filepath):
    status = {'progress': 0, 'log': None, 'charts': None, 'cleaned_data_csv': None, 'cleaned_data_excel': None, 'df_sample': None, 'error': None}
    processing_status[session_id] = status
    try:
        status['progress'] = 10
        ext = filepath.split('.')[-1].lower()
        if ext == 'csv':
            df = pd.read_csv(filepath, low_memory=False)
        elif ext in ['xls', 'xlsx']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError('Unsupported file type')
        status['progress'] = 30

        log = {}
        df_clean, log = auto_clean_data(df, log)
        status['progress'] = 60
        status['log'] = log

        charts = generate_visualizations(df_clean)
        status['progress'] = 80
        status['charts'] = charts

        # Store cleaned data
        csv_buffer = StringIO()
        df_clean.to_csv(csv_buffer, index=False)
        status['cleaned_data_csv'] = csv_buffer.getvalue()

        excel_buffer = BytesIO()
        df_clean.to_excel(excel_buffer, index=False, engine='openpyxl')
        status['cleaned_data_excel'] = excel_buffer.getvalue()

        status['progress'] = 100
        status['df_sample'] = df_clean.head(10).to_html(classes='dataframe')
    except Exception as e:
        status['error'] = str(e)
    finally:
        try:
            os.remove(filepath)
        except:
            pass
        processing_status[session_id] = status

@app.route('/status/<session_id>')
def get_status(session_id):
    status = processing_status.get(session_id)
    if not status:
        return jsonify({'error': 'Invalid session'}), 404
    result = {
        'progress': status.get('progress', 0),
        'error': status.get('error'),
        'log': status.get('log'),
        'charts': status.get('charts'),
        'df_sample': status.get('df_sample'),
        'ready': status.get('progress') == 100
    }
    return jsonify(result)

@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    data = request.json
    session_id = data.get('session_id')
    question = data.get('question')
    if not session_id or not question:
        return jsonify({'error': 'Missing session_id or question'}), 400
    status = processing_status.get(session_id)
    if not status or status.get('error'):
        return jsonify({'error': 'Data not available'}), 400
    try:
        csv_data = status.get('cleaned_data_csv')
        if not csv_data:
            return jsonify({'error': 'No cleaned data found'}), 400
        df = pd.read_csv(StringIO(csv_data))
        log = status.get('log', {})
        answer = ask_gemini_about_data(df, log, question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<session_id>/<filetype>')
def download_cleaned(session_id, filetype):
    status = processing_status.get(session_id)
    if not status or status.get('progress') != 100:
        return jsonify({'error': 'Data not ready'}), 400
    if filetype == 'csv':
        csv_data = status.get('cleaned_data_csv')
        if not csv_data:
            return jsonify({'error': 'CSV data missing'}), 404
        return send_file(BytesIO(csv_data.encode()), mimetype='text/csv',
                         as_attachment=True, download_name='cleaned_data.csv')
    elif filetype == 'excel':
        excel_data = status.get('cleaned_data_excel')
        if not excel_data:
            return jsonify({'error': 'Excel data missing'}), 404
        return send_file(BytesIO(excel_data), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True, download_name='cleaned_data.xlsx')
    else:
        return jsonify({'error': 'Invalid filetype'}), 400

@app.route('/report/<session_id>/<format>')
def download_report(session_id, format):
    status = processing_status.get(session_id)
    if not status or status.get('progress') != 100:
        return jsonify({'error': 'Data not ready'}), 400
    try:
        csv_data = status.get('cleaned_data_csv')
        df = pd.read_csv(StringIO(csv_data))
        log = status.get('log', {})
        charts = status.get('charts', {})
        ai_summary = ask_gemini_about_data(df, log, "Provide a concise summary of data quality and cleaning effectiveness.")
        if format == 'html':
            html_report = generate_html_report(df, log, charts, ai_summary)
            return send_file(BytesIO(html_report.encode()), mimetype='text/html',
                             as_attachment=True, download_name='data_cleaning_report.html')
        elif format == 'pdf':
            pdf_path = f"/tmp/report_{session_id}.pdf"
            generate_pdf_report(df, log, charts, ai_summary, pdf_path)
            return send_file(pdf_path, mimetype='application/pdf',
                             as_attachment=True, download_name='data_cleaning_report.pdf')
        else:
            return jsonify({'error': 'Invalid format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------------------------------------------------------------
# HTML Template (Modern, Dark/Light, Drag & Drop)
# ----------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Data Cleaning & Analysis Platform</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; transition: background 0.3s, color 0.3s; }
        :root {
            --bg: linear-gradient(135deg, #f5f7fa 0%, #e9edf2 100%);
            --card: rgba(255,255,255,0.95);
            --text: #1e293b;
            --text-sec: #475569;
            --border: #e2e8f0;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
        }
        body.dark {
            --bg: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --card: rgba(30,41,59,0.95);
            --text: #f1f5f9;
            --text-sec: #cbd5e1;
            --border: #334155;
            --accent: #60a5fa;
            --accent-hover: #3b82f6;
        }
        body { font-family: 'Segoe UI', system-ui; background: var(--bg); min-height: 100vh; padding: 2rem; }
        .container { max-width: 1400px; margin: 0 auto; }
        .card { background: var(--card); backdrop-filter: blur(10px); border-radius: 2rem; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 20px 35px -10px rgba(0,0,0,0.1); border: 1px solid var(--border); }
        h1, h2, h3 { color: var(--text); }
        .upload-area { border: 2px dashed var(--accent); border-radius: 1.5rem; padding: 3rem; text-align: center; cursor: pointer; transition: 0.2s; background: var(--card); }
        .upload-area:hover { background: var(--border); border-color: var(--accent-hover); }
        .progress-bar { width: 100%; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin: 1rem 0; }
        .progress-fill { width: 0%; height: 100%; background: var(--accent); transition: width 0.3s; }
        button { background: var(--accent); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 2rem; font-weight: 600; cursor: pointer; margin: 0.5rem; }
        button:hover { background: var(--accent-hover); transform: translateY(-2px); }
        .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 2rem; }
        .chart-container { background: var(--card); border-radius: 1rem; padding: 1rem; margin-bottom: 1rem; border: 1px solid var(--border); }
        .log-table { width: 100%; border-collapse: collapse; color: var(--text); }
        .log-table th, .log-table td { border: 1px solid var(--border); padding: 0.5rem; text-align: left; }
        .log-table th { background: var(--accent); color: white; }
        .theme-toggle { position: fixed; top: 20px; right: 20px; background: var(--card); border-radius: 2rem; padding: 0.5rem 1rem; cursor: pointer; z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        input { padding: 0.75rem; border-radius: 2rem; border: 1px solid var(--border); background: var(--card); color: var(--text); width: 70%; }
        @media (max-width: 768px) { body { padding: 1rem; } .grid-2 { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<div class="theme-toggle" onclick="document.body.classList.toggle('dark')">🌓 Dark/Light</div>
<div class="container">
    <div class="card">
        <h1>🧹 AI Data Cleaning & Analysis Platform</h1>
        <div class="upload-area" id="dropZone">
            <div style="font-size: 3rem;">📂</div>
            <p>Drag & drop or click to upload CSV/Excel</p>
            <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display: none;">
        </div>
        <div id="progressSection" style="display: none;">
            <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
            <p id="statusText">Processing...</p>
        </div>
    </div>
    <div id="results" style="display: none;">
        <div class="card"><h2>📋 Change Log</h2><div id="logTable"></div></div>
        <div class="card"><h2>🤖 Ask Gemini AI</h2><input type="text" id="questionInput" placeholder="e.g., Is this data clean? Any issues in column 'price'?"><button id="askBtn">Ask</button><div id="aiAnswer" style="margin-top:1rem; padding:1rem; background:var(--border); border-radius:1rem;"></div></div>
        <div class="card"><h2>📊 Visualizations</h2><div id="chartsContainer" class="grid-2"></div></div>
        <div class="card"><h2>📥 Downloads</h2><button id="downloadCsv">CSV</button><button id="downloadExcel">Excel</button><button id="downloadHtmlReport">HTML Report</button><button id="downloadPdfReport">PDF Report</button></div>
        <div class="card"><h2>🔍 Data Sample (first 10 rows)</h2><div id="sampleTable"></div></div>
    </div>
</div>
<script>
    let sessionId = null, pollInterval = null;
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const progressSection = document.getElementById('progressSection');
    const progressFill = document.getElementById('progressFill');
    const statusText = document.getElementById('statusText');
    const resultsDiv = document.getElementById('results');

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.opacity = '0.7'; });
    dropZone.addEventListener('dragleave', () => dropZone.style.opacity = '1');
    dropZone.addEventListener('drop', (e) => { e.preventDefault(); dropZone.style.opacity = '1'; const file = e.dataTransfer.files[0]; if(file) handleUpload(file); });
    fileInput.addEventListener('change', (e) => { if(e.target.files[0]) handleUpload(e.target.files[0]); });

    async function handleUpload(file) {
        const formData = new FormData();
        formData.append('file', file);
        progressSection.style.display = 'block';
        resultsDiv.style.display = 'none';
        progressFill.style.width = '0%';
        statusText.innerText = 'Uploading...';
        try {
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            sessionId = data.session_id;
            startPolling();
        } catch(err) { statusText.innerText = 'Upload failed: ' + err.message; }
    }

    function startPolling() {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/status/${sessionId}`);
                const data = await res.json();
                if (data.error) throw new Error(data.error);
                progressFill.style.width = data.progress + '%';
                statusText.innerText = `Processing... ${data.progress}%`;
                if (data.ready === true) {
                    clearInterval(pollInterval);
                    statusText.innerText = 'Complete!';
                    displayResults(data);
                    resultsDiv.style.display = 'block';
                }
            } catch(err) { clearInterval(pollInterval); statusText.innerText = 'Error: ' + err.message; }
        }, 1000);
    }

    function displayResults(data) {
        let logHtml = '<table class="log-table"><tr><th>Metric</th><th>Details</th></tr>';
        for (let [key, val] of Object.entries(data.log)) {
            logHtml += `<tr><td>${key}</td><td>${typeof val === 'object' ? JSON.stringify(val) : val}</td></tr>`;
        }
        logHtml += '</table>';
        document.getElementById('logTable').innerHTML = logHtml;
        const chartsDiv = document.getElementById('chartsContainer');
        chartsDiv.innerHTML = '';
        for (let [name, html] of Object.entries(data.charts)) {
            const div = document.createElement('div');
            div.className = 'chart-container';
            div.innerHTML = `<h3>${name.replace(/_/g, ' ').toUpperCase()}</h3>${html}`;
            chartsDiv.appendChild(div);
            const scripts = div.getElementsByTagName('script');
            for (let s of scripts) eval(s.textContent);
        }
        document.getElementById('sampleTable').innerHTML = data.df_sample || 'No sample';
        document.getElementById('downloadCsv').onclick = () => window.open(`/download/${sessionId}/csv`);
        document.getElementById('downloadExcel').onclick = () => window.open(`/download/${sessionId}/excel`);
        document.getElementById('downloadHtmlReport').onclick = () => window.open(`/report/${sessionId}/html`);
        document.getElementById('downloadPdfReport').onclick = () => window.open(`/report/${sessionId}/pdf`);
        document.getElementById('askBtn').onclick = async () => {
            const q = document.getElementById('questionInput').value;
            if(!q) return;
            const res = await fetch('/ask_gemini', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ session_id:sessionId, question:q }) });
            const ans = await res.json();
            document.getElementById('aiAnswer').innerHTML = ans.answer || ans.error;
        };
    }
</script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
