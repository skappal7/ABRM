import taipy as tp
import pandas as pd
import joblib
from data_preprocessing import preprocess_data
from model_training import train_models
from sklearn.metrics import accuracy_score, classification_report

# Define file paths and model names
FILE_PATH = 'ABRMData.xlsx'
MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']

# Taipy UI configuration
page = tp.Page()

@page.route('/')
def index():
    return tp.render_template('index.html')

@page.route('/upload', methods=['POST'])
def upload_file():
    file = tp.request.files['file']
    file.save(FILE_PATH)
    return tp.redirect('/train')

@page.route('/train')
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

    return tp.render_template('train.html', model_reports=model_reports)

@page.route('/predict', methods=['GET', 'POST'])
def predict():
    if tp.request.method == 'POST':
        input_data = tp.request.form.to_dict()
        input_df = pd.DataFrame([input_data])

        # Load the scaler and models
        scaler = joblib.load('scaler.joblib')
        input_scaled = scaler.transform(input_df)

        predictions = {}
        for model_name in MODEL_NAMES:
            model = joblib.load(f'{model_name}.joblib')
            prediction = model.predict(input_scaled)[0]
            predictions[model_name] = ['Low Risk', 'Medium Risk', 'High Risk'][prediction]

        return tp.render_template('predict.html', predictions=predictions)
    else:
        return tp.render_template('predict.html')

if __name__ == '__main__':
    tp.run(page, host='0.0.0.0', port=5000)
