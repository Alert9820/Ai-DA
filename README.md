# 🤖 AI Data Analyst Pro

> An AI-powered data analytics web app that auto-cleans your CSV/Excel data, generates smart visualizations, runs ML accuracy metrics, and lets you chat with an AI assistant about your data — all in one place.

🔗 **Live Demo:** [your-app.onrender.com](https://ai-da-1.onrender.com/)

---

## 📸 Preview

```
Upload CSV/Excel → Auto Clean → Visualize → ML Metrics → Chat with AI
```

---

## ✨ Features

### 📁 File Upload
- Supports **CSV** and **Excel** (.xlsx, .xls) files
- Handles files up to **500MB**
- Up to **100,000 rows** processed automatically

### 🧹 Auto Data Cleaning
- Duplicate row removal
- Missing value imputation — median for numeric, mode for categorical
- Outlier detection and removal using **IQR method**
- **Data Quality Score** — 0 to 100 rating after cleaning

### 📊 Smart Visualizations (Plotly)
- Bar Chart
- Pie Chart
- Line Chart
- Heatmap (Correlation Matrix)
- Scatter Plot
- Box Plot

### 🎯 ML Accuracy Metrics
- Auto-trains **Random Forest** model on numeric columns
- Shows **R² Score** and **RMSE**
- No manual configuration needed

### 🤖 AI Chat Assistant (Gemini)
- Powered by **Google Gemini 1.5 Flash**
- Ask anything about your data in natural language
- Get instant insights, summaries, and analysis

### 📥 Download Cleaned Data
- Download as **CSV**
- Download as **Excel**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest) |
| Visualizations | Plotly |
| AI Assistant | Google Gemini 1.5 Flash |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render.com |

---

## 📁 Project Structure

```
Ai-DA/
├── app.py           # Flask backend — ETL + ML + AI + API routes
├── requirements.txt # Python dependencies
└── README.md
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.11+
- Google Gemini API Key ([Get here](https://makersuite.google.com/app/apikey))

### Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Alert9820/Ai-DA.git
cd Ai-DA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export GEMINI_API_KEY="your_gemini_api_key_here"
export SECRET_KEY="your_secret_key_here"

# 4. Run the app
python app.py

# 5. Open browser
http://localhost:10000
```

### Windows Users
```bash
set GEMINI_API_KEY=your_gemini_api_key_here
set SECRET_KEY=your_secret_key_here
python app.py
```

---

## 🌐 Deploy on Render

1. Fork this repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set environment variables:
   - `GEMINI_API_KEY` → your Gemini API key
   - `SECRET_KEY` → any random string
5. Set **Start Command:**
   ```
   python app.py
   ```
6. Click **Deploy** ✅

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes (for AI chat) | Google Gemini API key |
| `SECRET_KEY` | Yes | Flask session secret key |

> Without `GEMINI_API_KEY`, all features work except the AI chat assistant.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve dashboard UI |
| `POST` | `/upload` | Upload CSV/Excel file |
| `POST` | `/chart` | Generate chart by type |
| `POST` | `/chat` | Chat with AI assistant |
| `GET` | `/download/csv` | Download cleaned CSV |
| `GET` | `/download/excel` | Download cleaned Excel |

---

## 📊 Supported File Formats

| Format | Extension | Max Size |
|---|---|---|
| CSV | `.csv` | 500MB |
| Excel | `.xlsx` | 500MB |
| Excel (old) | `.xls` | 500MB |

---

## 💡 Use Cases

- **Data Analysts** — Quick data cleaning + visual exploration
- **Business Teams** — Upload sales/HR/finance data, get instant insights
- **Students** — Learn data analysis concepts interactively
- **Non-technical Users** — Chat with AI to understand data without coding
- **Researchers** — Clean and visualize datasets instantly

---

## 🐛 Known Limitations

- Session data lost on server restart (Render free tier spins down)
- AI chat requires Gemini API key
- Very large files may slow down on free tier hosting
- ML metrics only available when 2+ numeric columns present

---

## 👨‍💻 Author

**Sunny Chaurasiya**
- Built as an AI-powered Data Analytics portfolio project
- Demonstrates: Data cleaning, ML automation, AI integration, Flask API, interactive visualizations

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  <strong>🤖 AI Data Analyst Pro — Your data, understood instantly.</strong>
</div>

