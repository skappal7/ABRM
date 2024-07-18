import os
import pandas as pd
import joblib
from flask import Flask, request, redirect
from data_preprocessing import preprocess_data
from model_training import train_models
from taipy.gui import Gui, Markdown

app = Flask(__name__)

FILE_PATH = 'ABRMData.csv'
MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']

gui = Gui(page_name="Agent Burnout Prediction")

# Markdown content for each page
introduction_md = """
# Agent Burnout Prediction

Burnout risk is a significant issue in workplaces. This application helps predict the risk of burnout for agents based on historical data and various metrics.
"""

visualization_md = """
## Data Visualization

Visualize the dataset with scatter plots and histograms.
"""

upload_md = """
## Data Upload

Upload your dataset here.
<form action="/upload" method="post" enctype="multipart/form-data">
    <input type="file" name="file">
    <input type="submit" value="Upload">
</form>
"""

preparation_md = """
## Data Preparation

Preprocess and prepare your data for model training.
"""

training_md = """
## Model Training

Train machine learning models and evaluate their performance.
"""

prediction_md = """
## Prediction

Input new data to predict burnout risk.
<form action="/predict" method="post">
    <label for="aht">Average of AHT (seconds):</label>
    <input type="text" id="aht" name="Average of AHT (seconds)"><br><br>
    <label for="attendance">Average of Attendance:</label>
    <input type="text" id="attendance" name="Average of Attendance"><br><br>
    <label for="csat">Average of CSAT (%):</label>
    <input type="text" id="csat" name="Average of CSAT (%)"><br><br>
    <label for="attrition">Attrition Flag:</label>
    <input type="text" id="attrition" name="Attrition Flag"><br><br>
    <input type="submit" value="Predict">
</form>
"""

database_md = """
## Database

View and download predictions and probabilities.
"""

@app.route('/')
def index():
    return gui.show(page_content=Markdown(introduction_md))

@app.route('/visualization')
def visualization():
    return gui.show(page_content=Markdown(visualization_md))

@app.route('/upload')
def upload():
    return gui.show(page_content=Markdown(upload_md))

@app.route('/prepare')
def prepare():
    return gui.show(page_content=Markdown(preparation_md))

@app.route('/train')
def train():
    return gui.show(page_content=Markdown(training_md))

@app.route('/predict')
def predict():
    return gui.show(page_content=Markdown(prediction_md))

@app.route('/database')
def database():
    return gui.show(page_content=Markdown(database_md))

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save(FILE_PATH)
    print(f"File saved to {FILE_PATH}")
    return redirect('/prepare')

@app.route('/prepare', methods=['POST'])
def prepare_data():
    try:
        X, y, scaler = preprocess_data(FILE_PATH)
        joblib.dump(scaler, 'scaler.joblib')
        print("Data prepared successfully.")
        return redirect('/train')
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return f"Error: {e}"

@app.route('/train', methods=['POST'])
def train_models_route():
    try:
        X, y, scaler = preprocess_data(FILE_PATH)
        models = train_models(X, y)
        for model_name in MODEL_NAMES:
            model = joblib.load(f'{model_name}.joblib')
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            report = classification_report(y, y_pred, target_names=['Low Risk', 'Medium Risk', 'High Risk'])
            print(f"{model_name} model accuracy: {accuracy}")
            print(report)
        return redirect('/database')
    except Exception as e:
        print(f"Error during model training: {e}")
        return f"Error: {e}"

@app.route('/predict', methods=['POST'])
def predict_route():
    input_data = request.form.to_dict()
    input_df = pd.DataFrame([input_data])
    scaler = joblib.load('scaler.joblib')
    input_scaled = scaler.transform(input_df)
    predictions = {}
    for model_name in MODEL_NAMES:
        model = joblib.load(f'{model_name}.joblib')
        prediction = model.predict(input_scaled)[0]
        predictions[model_name] = ['Low Risk', 'Medium Risk', 'High Risk'][prediction]
    table_rows = "".join([f"<tr><td>{model_name}</td><td>{predictions[model_name]}</td></tr>" for model_name in MODEL_NAMES])
    return f"""
    <h2>Predictions</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Prediction</th>
        </tr>
        {table_rows}
    </table>
    <a href="/">Go Back</a>
    """

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
