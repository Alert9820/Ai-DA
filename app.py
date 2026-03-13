from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import plotly.express as px
import io

app = Flask(__name__)

# -------- DATA CLEANING -------- #

def clean_data(df):

    # remove duplicates
    df = df.drop_duplicates()

    # fill missing numeric values
    for col in df.select_dtypes(include=np.number):
        df[col] = df[col].fillna(df[col].median())

    # fill missing categorical values
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# -------- OUTLIER REMOVAL -------- #

def remove_outliers(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


# -------- PROMPT INTERPRETER -------- #

def interpret_prompt(prompt):

    prompt = prompt.lower()

    if "chart" in prompt or "plot" in prompt:
        return "chart"

    if "summary" in prompt or "describe" in prompt:
        return "summary"

    if "clean" in prompt:
        return "clean"

    return "summary"


# -------- CHART GENERATOR -------- #

def create_chart(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) >= 2:
        x = numeric_cols[0]
        y = numeric_cols[1]
    else:
        return "Not enough numeric columns"

    fig = px.bar(df, x=x, y=y)

    return fig.to_html()


# -------- MAIN API -------- #

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
        return jsonify({"error": "Unsupported file format"})

    # clean data
    df = clean_data(df)

    # remove outliers
    df = remove_outliers(df)

    # understand prompt
    task = interpret_prompt(prompt)

    if task == "summary":

        result = df.describe().to_html()

        return result

    if task == "chart":

        chart = create_chart(df)

        return chart

    if task == "clean":

        return df.head().to_html()


# -------- RUN SERVER -------- #

if __name__ == "__main__":
    app.run(debug=True)
