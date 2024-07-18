import taipy as tp
from taipy import Gui, Config, Core, Scope
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from data_preprocessing import preprocess_data
from model_training import train_model, predict

# Initial data loading
data = pd.read_csv("ABRMData.csv")

# Global variables
selected_model = "Random Forest"
models = {}
X_train, X_test, y_train, y_test = None, None, None, None

# Utility functions
def create_chart(df, x_column, y_column, chart_type="scatter"):
    plt.figure(figsize=(10, 6))
    if chart_type == "scatter":
        sns.scatterplot(data=df, x=x_column, y=y_column, hue="EXITED")
    elif chart_type == "histogram":
        sns.histplot(data=df, x=x_column, hue="EXITED", kde=True)
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
    if not models:
        state.prediction_result = "Please train a model first."
        return
    
    input_data = pd.DataFrame({
        'CREDITSCORE': [state.creditscore],
        'TENURE': [state.tenure],
        # Add other features here
    })
    
    model = models[state.selected_model]
    prediction, probability = predict(model, input_data)
    
    state.prediction_result = f"Prediction: {'High' if prediction[0] == 1 else 'Low'} burnout risk"
    state.prediction_probability = f"Probability: {probability[0][1]:.2f}"

# Taipy pages
index_md = """
# Agent Burnout Risk Prediction

Welcome to the Agent Burnout Risk Prediction application. This tool helps identify agents at risk of burnout using machine learning techniques.

## Features:
- Data visualization
- Model training
- Burnout risk prediction

<|Navigate to Data Visualization|button|on_action=lambda state: state.navigate("data_visualization")|>
"""

data_viz_md = """
# Data Visualization

<|{x_column}|selector|label=Select X|lov={data.columns}|on_change=on_change_x|>
<|{y_column}|selector|label=Select Y|lov={data.columns}|on_change=on_change_y|>
<|{chart_type}|selector|label=Chart Type|lov=["scatter", "histogram"]|on_change=on_change_chart_type|>

<|{chart}|image|>

<|Navigate to Model Training|button|on_action=lambda state: state.navigate("train")|>
"""

train_md = """
# Model Training

<|{selected_model}|selector|label=Select Model|lov=["Random Forest"]|>
<|Prepare Data|button|on_action=prepare_data|>
<|Train Model|button|on_action=train_model_gui|>

<|{data_prep_message}|>
<|{train_message}|>

## Confusion Matrix

<|{confusion_matrix}|image|>

<|Navigate to Prediction|button|on_action=lambda state: state.navigate("predict")|>
"""

predict_md = """
# Prediction

<|{creditscore}|number|label=Credit Score|>
<|{tenure}|number|label=Tenure|>
# Add other input fields here

<|Make Prediction|button|on_action=make_prediction|>

<|{prediction_result}|>
<|{prediction_probability}|>

<|Navigate to Home|button|on_action=lambda state: state.navigate("/")|>
"""

# Taipy app configuration
config = Config()

# Define the pages
pages = {
    "/": index_md,
    "data_visualization": data_viz_md,
    "train": train_md,
    "predict": predict_md,
}

# Initial state
initial_state = {
    "data": data,
    "x_column": data.columns[0],
    "y_column": data.columns[1],
    "chart_type": "scatter",
    "chart": create_chart(data, data.columns[0], data.columns[1]),
    "selected_model": "Random Forest",
    "creditscore": 700,
    "tenure": 5,
    "data_prep_message": "",
    "train_message": "",
    "prediction_result": "",
    "prediction_probability": "",
}

# Create the Gui object
gui = Gui(pages=pages)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    gui.run(host="0.0.0.0", port=port)
