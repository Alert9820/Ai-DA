import os
import json
import uuid
import threading
import time
from datetime import datetime
from io import BytesIO, StringIO
import base64

from flask import Flask, request, render_template_string, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Gemini AI Setup
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

processing_status = {}
results_cache = {}

def auto_clean_data(df):
    """Complete Automatic Data Cleaning Pipeline"""
    log = {
        'initial_shape': f"{len(df)} rows × {len(df.columns)} columns",
        'duplicates_removed': 0,
        'missing_values': {},
        'outliers_removed': 0,
        'data_types_fixed': [],
        'final_shape': ''
    }
    
    original_len = len(df)
    
    # 1. Remove Duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    log['duplicates_removed'] = initial_len - len(df)
    
    # 2. Handle Missing Values
    missing_info = df.isnull().sum().to_dict()
    log['missing_values'] = {col: int(val) for col, val in missing_info.items() if val > 0}
    
    for col in df.columns:
        if df[col].dtype in ['object', 'string']:
            # Mode for categorical
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
        else:
            # Median for numeric
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # 3. Fix Data Types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    # 4. Detect & Remove Outliers (IQR Method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_count = 0
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_count += len(outliers)
        
        # Replace outliers with median
        median_val = df[col].median()
        df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = median_val
    
    log['outliers_removed'] = outliers_count
    log['final_shape'] = f"{len(df)} rows × {len(df.columns)} columns"
    
    return df, log

def generate_visualizations(df):
    """Generate Interactive Plotly Charts"""
    charts = {}
    
    # 1. Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, 
                       title="Correlation Heatmap",
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        charts['correlation_heatmap'] = fig.to_html(full_html=False, include_plotlyjs=False)
    
    # 2. Distribution Plots
    for col in df.select_dtypes(include=[np.number]).columns[:3]:  # Top 3 numeric columns
        fig = px.histogram(df, x=col, title=f'Distribution of {col}', 
                          marginal="box", nbins=50)
        charts[f'distribution_{col}'] = fig.to_html(full_html=False, include_plotlyjs=False)
    
    # 3. Box Plots for Outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
    if len(numeric_cols) > 0:
        fig = px.box(df, y=numeric_cols.tolist(), title="Box Plots (Outliers Detection)")
        charts['box_plots'] = fig.to_html(full_html=False, include_plotlyjs=False)
    
    return charts

def ask_gemini_about_data(df_sample, log, user_question):
    """Ask Gemini AI about the data"""
    try:
        context = f"""
        Data Summary:
        - Initial shape: {log['initial_shape']}
        - Final shape: {log['final_shape']}
        - Duplicates removed: {log['duplicates_removed']}
        - Outliers handled: {log['outliers_removed']}
        - Sample data (first 5 rows): {df_sample.head().to_json(orient='records')}
        
        User Question: {user_question}
        
        Provide analysis in Hindi/English mix. Be specific about data quality.
        """
        
        response = gemini_model.generate_content(context)
        return response.text
    except Exception as e:
        return f"AI Analysis: {str(e)}. Data preprocessing completed successfully!"

def generate_html_report(df, log, charts, ai_insights):
    """Generate HTML Report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            .header {{ text-align: center; color: #1e40af; }}
            .log-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .log-table th, .log-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            .log-table th {{ background-color: #1e40af; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 AI Data Cleaning Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>🔄 Preprocessing Summary</h2>
        <table class="log-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Initial Shape</td><td>{log['initial_shape']}</td></tr>
            <tr><td>Final Shape</td><td>{log['final_shape']}</td></tr>
            <tr><td>Duplicates Removed</td><td>{log['duplicates_removed']}</td></tr>
            <tr><td>Outliers Handled</td><td>{log['outliers_removed']}</td></tr>
        </table>
        
        <h2>📈 Visualizations</h2>
    """
    
    for chart_name, chart_html in charts.items():
        html_content += f"<h3>{chart_name.replace('_', ' ').title()}</h3>{chart_html}"
    
    html_content += f"""
        <h2>🤖 AI Insights</h2>
        <div style="background: #f3f4f6; padding: 20px; border-radius: 8px;">
            {ai_insights}
        </div>
        </body>
    </html>
    """
    return html_content

def generate_pdf_report(df, log, charts, ai_insights, filename='report.pdf'):
    """Generate PDF Report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], 
                                fontSize=24, spaceAfter=30, alignment=1)
    story.append(Paragraph("📊 AI Data Cleaning Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Summary Table
    summary_data = [
        ['Metric', 'Value'],
        ['Initial Shape', log['initial_shape']],
        ['Final Shape', log['final_shape']],
        ['Duplicates Removed', str(log['duplicates_removed'])],
        ['Outliers Handled', str(log['outliers_removed'])]
    ]
    
    table = Table(summary_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    
    # AI Insights
    story.append(Spacer(1, 20))
    story.append(Paragraph("🤖 AI Insights", styles['Heading2']))
    story.append(Paragraph(ai_insights, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer, filename

@app.route('/upload', methods=['POST'])
def upload_file():
    print("🔵 Upload endpoint hit")
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
    """Process file in background thread"""
    try:
        print(f"🔄 Processing {session_id}")
        processing_status[session_id] = {'progress': 0, 'status': 'loading', 'ready': False}
        
        # Load Data
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        processing_status[session_id]['progress'] = 30
        processing_status[session_id]['status'] = 'cleaning'
        
        # Clean Data
        cleaned_df, log = auto_clean_data(df)
        processing_status[session_id]['progress'] = 70
        processing_status[session_id]['status'] = 'visualizing'
        
        # Generate Visualizations
        charts = generate_visualizations(cleaned_df)
        processing_status[session_id]['progress'] = 90
        processing_status[session_id]['status'] = 'ai_analysis'
        
        # AI Analysis
        df_sample = cleaned_df.head(10)
        ai_insights = ask_gemini_about_data(df_sample, log, "Provide data quality summary")
        
        # Cache Results
        results_cache[session_id] = {
            'df': cleaned_df.to_json(orient='records', date_format='iso'),
            'log': log,
            'charts': charts,
            'ai_insights': ai_insights,
            'df_sample': cleaned_df.head(10).to_html()
        }
        
        processing_status[session_id]['progress'] = 100
        processing_status[session_id]['status'] = 'complete'
        processing_status[session_id]['ready'] = True
        
        print(f"✅ Processing complete for {session_id}")
        
    except Exception as e:
        print(f"❌ Error processing {session_id}: {str(e)}")
        processing_status[session_id] = {
            'progress': 0, 
            'status': f'error: {str(e)}', 
            'ready': False
        }
    
    finally:
        # Cleanup file after 1 hour
        threading.Timer(3600, lambda: os.remove(filepath)).start()

@app.route('/status/<session_id>')
def get_status(session_id):
    status = processing_status.get(session_id, {'progress': 0, 'status': 'not_found', 'ready': False})
    if status.get('ready') and session_id in results_cache:
        status['log'] = results_cache[session_id]['log']
        status['df_sample'] = results_cache[session_id]['df_sample']
    return jsonify(status)

@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    data = request.json
    session_id = data['session_id']
    question = data['question']
    
    if session_id not in results_cache:
        return jsonify({'error': 'Session not found'})
    
    result = results_cache[session_id]
    df = pd.read_json(result['df'])
    ai_response = ask_gemini_about_data(df.head(10), result['log'], question)
    
    return jsonify({'answer': ai_response})

@app.route('/download/<session_id>/<filetype>')
def download_cleaned(session_id, filetype):
    if session_id not in results_cache:
        return jsonify({'error': 'Session expired'}), 404
    
    df_json = results_cache[session_id]['df']
    df = pd.read_json(df_json)
    
    if filetype == 'csv':
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return send_file(BytesIO(buffer.getvalue().encode()), 
                        mimetype='text/csv', 
                        as_attachment=True, 
                        download_name='cleaned_data.csv')
    elif filetype == 'excel':
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Cleaned Data')
        buffer.seek(0)
        return send_file(buffer, 
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        as_attachment=True, 
                        download_name='cleaned_data.xlsx')

@app.route('/report/<session_id>/<report_format>')
def download_report(session_id, report_format):
    if session_id not in results_cache:
        return jsonify({'error': 'Session expired'}), 404
    
    result = results_cache[session_id]
    
    if report_format == 'html':
        html_report = generate_html_report(
            pd.read_json(result['df']),
            result['log'],
            result['charts'],
            result['ai_insights']
        )
        return html_report, 200, {'Content-Type': 'text/html'}
    
    elif report_format == 'pdf':
        buffer, filename = generate_pdf_report(
            pd.read_json(result['df']),
            result['log'],
            result['charts'],
            result['ai_insights']
        )
        return send_file(buffer, 
                        mimetype='application/pdf',
                        as_attachment=True, 
                        download_name=filename)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
