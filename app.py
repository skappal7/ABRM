import pandas as pd
import joblib
from flask import Flask, request, redirect, render_template_string
from data_preprocessing import preprocess_data
from model_training import train_models
from sklearn.metrics import accuracy_score, classification_report

# Initialize Flask app
app = Flask(__name__)

# Define file paths and model names
FILE_PATH = 'ABRMData.csv'
MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']

@app.route('/')
def index():
    return """
    <h1>Agent Burnout Prediction</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    """

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save(FILE_PATH)
    return redirect('/train')

@app.route('/train')
def train():
    X, y, scaler = preprocess_data(FILE_PATH)
    models = train_models(X, y)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.joblib')
    
    model_reports = {}
    for model_name in MODEL_NAMES:
        model = joblib.load(f'{model_name}.joblib')
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=['Low Risk', 'Medium Risk', 'High Risk'])
        model_reports[model_name] = {'Accuracy': accuracy, 'Report': report}

    table_rows = "".join([f"<tr><td>{model_name}</td><td>{model_reports[model_name]['Accuracy']}</td><td><pre>{model_reports[model_name]['Report']}</pre></td></tr>" for model_name in MODEL_NAMES])
    return f"""
    <h2>Model Training Complete</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Report</th>
        </tr>
        {table_rows}
    </table>
    """

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form.to_dict()
        input_df = pd.DataFrame([input_data])

        # Load the scaler and models
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
        """
    else:
        return """
        <h2>Enter Agent Data for Prediction</h2>
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
