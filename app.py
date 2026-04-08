import os
import json
import uuid
import threading
from datetime import datetime
from io import BytesIO, StringIO

from flask import Flask, request, render_template_string, jsonify, send_file
from flask_cors import CORS  # ADDED
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # <-- ENABLE CORS

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

processing_status = {}

# ---------- (baaki saare functions auto_clean_data, generate_visualizations, etc. WAISE HI RAHENGE) ----------
# NOTE: maine functions same rakhe hain jaise pehle the. Sirf CORS aur ek extra debug print add kiya hai.

def auto_clean_data(df, log):
    # ... (same as previous code)
    pass

def generate_visualizations(df):
    # ... (same)
    pass

def ask_gemini_about_data(df, log, user_question):
    # ... (same)
    pass

def generate_html_report(df, log, charts, ai_insights):
    # ... (same)
    pass

def generate_pdf_report(df, log, charts, ai_insights, filename='report.pdf'):
    # ... (same)
    pass

# ---------- UPLOAD ENDPOINT (with debug print) ----------
@app.route('/upload', methods=['POST'])
def upload_file():
    print("🔵 Upload endpoint hit")  # DEBUG
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    session_id = str(uuid.uuid4())
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{secure_filename(file.filename)}")
    file.save(temp_path)
    print(f"✅ File saved: {temp_path}")

    thread = threading.Thread(target=process_file, args=(session_id, temp_path))
    thread.daemon = True
    thread.start()

    return jsonify({'session_id': session_id, 'status': 'processing'})

def process_file(session_id, filepath):
    # ... (same as before)
    pass

@app.route('/status/<session_id>')
def get_status(session_id):
    # ... (same)
    pass

@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    # ... (same)
    pass

@app.route('/download/<session_id>/<filetype>')
def download_cleaned(session_id, filetype):
    # ... (same)
    pass

@app.route('/report/<session_id>/<format>')
def download_report(session_id, format):
    # ... (same)
    pass

# ---------- HTML TEMPLATE (slight improvement in fetch error handling) ----------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Data Cleaning & Analysis Platform</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* same as before */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui; background: var(--bg-gradient); min-height: 100vh; padding: 2rem; }
        :root { --bg-gradient: linear-gradient(135deg, #f5f7fa 0%, #e9edf2 100%); --card-bg: rgba(255,255,255,0.95); --text-primary: #1e293b; --border: #e2e8f0; --accent: #3b82f6; }
        body.dark { --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); --card-bg: rgba(30,41,59,0.95); --text-primary: #f1f5f9; --border: #334155; --accent: #60a5fa; }
        .container { max-width: 1400px; margin: 0 auto; }
        .card { background: var(--card-bg); backdrop-filter: blur(10px); border-radius: 2rem; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 20px 35px -10px rgba(0,0,0,0.1); border: 1px solid var(--border); }
        .upload-area { border: 2px dashed var(--accent); border-radius: 1.5rem; padding: 3rem; text-align: center; cursor: pointer; transition: 0.2s; }
        .upload-area:hover { background: var(--border); }
        .progress-bar { width: 100%; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin: 1rem 0; }
        .progress-fill { width: 0%; height: 100%; background: var(--accent); transition: width 0.3s; }
        button { background: var(--accent); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 2rem; cursor: pointer; }
        .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 2rem; }
        .theme-toggle { position: fixed; top: 20px; right: 20px; background: var(--card-bg); border-radius: 2rem; padding: 0.5rem 1rem; cursor: pointer; }
    </style>
</head>
<body>
<div class="theme-toggle" onclick="toggleTheme()">🌓 Dark/Light</div>
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
    <div id="results" style="display: none;">...</div>
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
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.opacity = '1';
        const file = e.dataTransfer.files[0];
        if (file) handleUpload(file);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) handleUpload(e.target.files[0]);
    });

    async function handleUpload(file) {
        const formData = new FormData();
        formData.append('file', file);
        progressSection.style.display = 'block';
        resultsDiv.style.display = 'none';
        progressFill.style.width = '0%';
        statusText.innerText = 'Uploading...';
        try {
            const res = await fetch('/upload', { method: 'POST', body: formData });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            sessionId = data.session_id;
            startPolling();
        } catch (err) {
            statusText.innerText = 'Upload failed: ' + err.message;
            console.error(err);
        }
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
            } catch (err) {
                clearInterval(pollInterval);
                statusText.innerText = 'Error: ' + err.message;
            }
        }, 1000);
    }

    function displayResults(data) {
        // similar to previous code
        let logHtml = '<table class="log-table"><tr><th>Metric</th><th>Details</th></tr>';
        for (let [key, val] of Object.entries(data.log)) {
            logHtml += `<tr><td>${key}</td><td>${JSON.stringify(val)}</td></tr>`;
        }
        logHtml += '</table>';
        document.getElementById('logTable').innerHTML = logHtml;

        const chartsDiv = document.getElementById('chartsContainer');
        chartsDiv.innerHTML = '';
        for (let [name, html] of Object.entries(data.charts)) {
            const div = document.createElement('div');
            div.className = 'chart-container';
            div.innerHTML = `<h3>${name.replace(/_/g, ' ')}</h3>${html}`;
            chartsDiv.appendChild(div);
            const scripts = div.getElementsByTagName('script');
            for (let script of scripts) eval(script.textContent);
        }
        document.getElementById('sampleTable').innerHTML = data.df_sample || 'No sample';
        document.getElementById('downloadCsv').onclick = () => window.open(`/download/${sessionId}/csv`);
        document.getElementById('downloadExcel').onclick = () => window.open(`/download/${sessionId}/excel`);
        document.getElementById('downloadHtmlReport').onclick = () => window.open(`/report/${sessionId}/html`);
        document.getElementById('downloadPdfReport').onclick = () => window.open(`/report/${sessionId}/pdf`);
        document.getElementById('askBtn').onclick = async () => {
            const question = document.getElementById('questionInput').value;
            if (!question) return;
            const res = await fetch('/ask_gemini', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, question: question })
            });
            const ans = await res.json();
            document.getElementById('aiAnswer').innerHTML = ans.answer || ans.error;
        };
    }
    function toggleTheme() { document.body.classList.toggle('dark'); }
</script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
