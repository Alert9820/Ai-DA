from flask import Flask, request
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
import os

app = Flask(__name__)

# -------- GEMINI SETUP -------- #

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")


# -------- DATA CLEANING -------- #

def clean_data(df):

    df = df.drop_duplicates()

    # numeric missing
    for col in df.select_dtypes(include=np.number):
        df[col] = df[col].fillna(df[col].median())

    # categorical missing
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# -------- OUTLIER REMOVAL -------- #

def remove_outliers(df):

    numeric_cols = df.select_dtypes(include=np.number)

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


# -------- GEMINI PROMPT UNDERSTANDING -------- #

def interpret_prompt(prompt, columns):

    context = f"""
You are a professional data analyst.

Dataset columns:
{columns}

User request:
{prompt}

Return ONLY one word from this list:
chart
summary
correlation
"""

    try:
        response = model.generate_content(context)
        result = response.text.strip().lower()
    except:
        result = "summary"

    return result


# -------- HOME PAGE -------- #

@app.route("/")
def home():

    return """
    <html>
    <head>

    <title>AI Data Analyst</title>

    <style>

    body{
        font-family: Arial;
        text-align:center;
        background:#f2f2f2;
        padding:40px;
    }

    h1{
        color:#333;
    }

    input,button{
        padding:10px;
        margin:10px;
    }

    </style>

    </head>

    <body>

    <h1>AI Data Analyst</h1>

    <form action="/analyze" method="post" enctype="multipart/form-data">

    <input type="file" name="file" required><br>

    <input type="text" name="prompt" placeholder="example: show sales chart"><br>

    <button type="submit">Analyze</button>

    </form>

    </body>
    </html>
    """


# -------- ANALYSIS -------- #

@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files["file"]
    prompt = request.form["prompt"]

    # read dataset
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)

    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(file)

    else:
        return "Unsupported file type"

    # data cleaning
    df = clean_data(df)

    # outlier removal
    df = remove_outliers(df)

    columns = list(df.columns)

    # prompt understanding
    task = interpret_prompt(prompt, columns)

    # SUMMARY
    if "summary" in task:

        return """
        <h2>Dataset Summary</h2>
        """ + df.describe().to_html()

    # CORRELATION
    if "correlation" in task:

        corr = df.corr(numeric_only=True)

        fig = px.imshow(corr)

        return fig.to_html()

    # CHART
    if "chart" in task:

        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) < 2:
            return "Not enough numeric columns for chart"

        fig = px.bar(df, x=numeric_cols[0], y=numeric_cols[1])

        return fig.to_html()

    return df.head().to_html()


# -------- RUN SERVER -------- #

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
