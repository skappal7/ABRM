import taipy as tp
from taipy import Gui, Config, Core, Scope
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Global variables
data = pd.DataFrame()
selected_model = "Random Forest"
models = {}
X_train, X_test, y_train, y_test = None, None, None, None

# Data preprocessing function
def preprocess_data(data):
    # Drop 'Agent ID' and 'Attrition Flag' columns
    data = data.drop(['Agent ID', 'Attrition Flag'], axis=1, errors='ignore')
    
    # Convert categorical variables to numeric if any
    data = pd.get_dummies(data, drop_first=True)
    
    # Separate features and target
    X = data.drop('Risk Indicator', axis=1)
    y = data['Risk Indicator']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Model training function
def train_model(X_train, y_train, X_test, y_test, model_type="Random Forest"):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    
    if model_type not in models:
        raise ValueError("Unsupported model type")
    
    model = models[model_type]
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, cm

# Prediction function
def predict(model, input_data):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    return prediction, probability

# Utility functions
def create_chart(df, x_column, y_column, chart_type="scatter"):
    plt.figure(figsize=(10, 6))
    if chart_type == "scatter":
        sns.scatterplot(data=df, x=x_column, y=y_column, hue="Risk Indicator")
    elif chart_type == "histogram":
        sns.histplot(data=df, x=x_column, hue="Risk Indicator", kde=True)
    plt.title(f"{chart_type.capitalize()} plot of {x_column} vs {y_column}")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Page content and functions
def on_change_x(state):
    state.chart = create_chart(state.data, state.x_column, state.y_column, state.chart_type)

def on_change_y(state):
    state.chart = create_chart(state.data, state.x_column, state.y_column, state.chart_type)

def on_change_chart_type(state):
    state.chart = create_chart(state.data, state.x_column, state.y_column, state.chart_type)

def on_file_upload(state, file):
    if file:
        state.data = pd.read_csv(file)
        state.data_summary = state.data.describe().to_html()
        state.x_column = state.data.columns[1]  # Skip 'Agent ID'
        state.y_column = state.data.columns[2]
        state.chart = create_chart(state.data, state.x_column, state.y_column, state.chart_type)
        return state.data_summary

def prepare_data(state):
    global X_train, X_test, y_train, y_test
    X, y = preprocess_data(state.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    state.data_prep_message = "Data prepared for model training."

def train_model_gui(state):
    global models, X_train, X_test, y_train, y_test
    if X_train is None:
        state.train_message = "Please prepare the data first."
        return
    
    model, accuracy, cm = train_model(X_train, y_train, X_test, y_test, model_type=state.selected_model)
    models[state.selected_model] = model
    
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    state.confusion_matrix = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    state.train_message = f"Model trained. Accuracy: {accuracy:.2f}"

def make_prediction(state):
    if not models or state.data.empty:
        state.prediction_result = "Please upload data and train a model first."
        return
    
    model = models[state.selected_model]
    X, _ = preprocess_data(state.data)
    predictions, probabilities = predict(model, X)
    
    state.data['Predicted Risk'] = predictions
    state.data['Risk Probability'] = probabilities[:, 1]  # Assuming binary classification
    
    state.prediction_result = "Predictions made. You can now download the results."
    state.prediction_table = state.data.to_html(index=False)

# Taipy pages
root_md = """
<|toggle|theme|>

# Agent Risk Indicator Prediction

Welcome to the Agent Risk Indicator Prediction application. This tool helps predict risk indicators for agents based on various factors.

## Navigation

<|layout|columns=1 1 1 1|
<|card|
### Data Upload
<|navigate|data_upload|Upload Data|button|>
|>

<|card|
### Data Visualization
<|navigate|data_visualization|Visualize Data|button|>
|>

<|card|
### Model Training
<|navigate|train|Train Model|button|>
|>

<|card|
### Make Predictions
<|navigate|predict|Predict|button|>
|>
|>
"""

data_upload_md = """
<|toggle|theme|>

# Data Upload

<|layout|columns=1 1|
<|card|
## Upload your CSV file
<|{file}|file_selector|label=Upload CSV|on_change=on_file_upload|>
|>

<|card|
## Data Summary
<|{data_summary}|raw|>
|>
|>

<|navigate|data_visualization|Visualize Data|button|>
"""

data_viz_md = """
<|toggle|theme|>

# Data Visualization

<|layout|columns=1 1 1|
<|{x_column}|selector|label=Select X|lov={data.columns}|on_change=on_change_x|>
<|{y_column}|selector|label=Select Y|lov={data.columns}|on_change=on_change_y|>
<|{chart_type}|selector|label=Chart Type|lov=["scatter", "histogram"]|on_change=on_change_chart_type|>
|>

<|{chart}|image|>

<|navigate|train|Train Model|button|>
"""

train_md = """
<|toggle|theme|>

# Model Training

<|layout|columns=1 1 1|
<|{selected_model}|selector|label=Select Model|lov=["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]|>
<|Prepare Data|button|on_action=prepare_data|>
<|Train Model|button|on_action=train_model_gui|>
|>

<|{data_prep_message}|>
<|{train_message}|>

## Confusion Matrix

<|{confusion_matrix}|image|>

<|navigate|predict|Make Predictions|button|>
"""

predict_md = """
<|toggle|theme|>

# Prediction

<|Make Predictions|button|on_action=make_prediction|>

<|{prediction_result}|>

## Prediction Results

<|{prediction_table}|raw|>

<|Download Results|button|on_action=lambda state: state.data.to_csv("predictions.csv")|>

<|navigate|/|Back to Home|button|>
"""

# Taipy app configuration
pages = {
    "/": root_md,
    "data_upload": data_upload_md,
    "data_visualization": data_viz_md,
    "train": train_md,
    "predict": predict_md,
}

# Initial state
initial_state = {
    "data": data,
    "x_column": "",
    "y_column": "",
    "chart_type": "scatter",
    "chart": "",
    "selected_model": "Random Forest",
    "data_summary": "",
    "data_prep_message": "",
    "train_message": "",
    "prediction_result": "",
    "prediction_table": "",
    "confusion_matrix": "",
}

# Create the Gui object
gui = Gui(pages=pages)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    gui.run(host="0.0.0.0", port=port, use_reloader=False)
