import pandas as pd
import joblib
from data_preprocessing import preprocess_data
from model_training import train_models
from sklearn.metrics import accuracy_score, classification_report
from taipy.gui import Gui

# Define file paths and model names
FILE_PATH = 'ABRMData.csv'
MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']

# Define the GUI
gui = Gui()

@gui.page("/")
def index():
    return """
    <h1>Agent Burnout Prediction</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    """

@gui.page("/upload", methods=['POST'])
def upload_file():
    file = gui.request.files['file']
    file.save(FILE_PATH)
    return gui.redirect('/train')

@gui.page("/train")
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

    return f"""
    <h2>Model Training Complete</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Report</th>
        </tr>
        {" ".join([f"<tr><td>{model_name}</td><td>{model_reports[model_name]['Accuracy']}</td><td><pre>{model_reports[model_name]['Report']}</pre></td></tr>" for model_name in MODEL_NAMES])}
    </table>
    """

@gui.page("/predict", methods=['GET', 'POST'])
def predict():
    if gui.request.method == 'POST':
        input_data = gui.request.form.to_dict()
        input_df = pd.DataFrame([input_data])

        # Load the scaler and models
        scaler = joblib.load('scaler.joblib')
        input_scaled = scaler.transform(input_df)

        predictions = {}
        for model_name in MODEL_NAMES:
            model = joblib.load(f'{model_name}.joblib')
            prediction = model.predict(input_scaled)[0]
            predictions[model_name] = ['Low Risk', 'Medium Risk', 'High Risk'][prediction]

        return f"""
        <h2>Predictions</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Prediction</th>
            </tr>
            {" ".join([f"<tr><td>{model_name}</td><td>{predictions[model_name]}</td></tr>" for model_name in MODEL_NAMES])}
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
    gui.run()
