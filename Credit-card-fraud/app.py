from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
import tempfile
import shutil
import uuid

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'fraud_detection_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
try:
    model = joblib.load('fraud_model.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Only CSV files are allowed', 'error')
        return redirect(url_for('index'))

    try:
        # Create unique filename to prevent collisions
        filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        # Read CSV
        try:
            df = pd.read_csv(temp_path)
        except Exception as e:
            flash(f'Error reading CSV: {str(e)}', 'error')
            return redirect(url_for('index'))

        # Check required columns
        required_columns = {'Time', 'Amount'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            flash(f'Missing required columns: {", ".join(missing)}', 'error')
            return redirect(url_for('index'))

        # Process data
        original_df = df.copy()
        scaler = StandardScaler()
        
        if 'Amount' in df.columns:
            df['Amount'] = scaler.fit_transform(df[['Amount']])
        if 'Time' in df.columns:
            df['Time'] = scaler.fit_transform(df[['Time']])
        if 'Class' in df.columns:
            df.drop('Class', axis=1, inplace=True)

        # Make predictions
        predictions = model.predict(df)
        original_df['Fraud Prediction'] = ['Yes' if x == 1 else 'No' for x in predictions]
        original_df['Risk Score'] = (model.predict_proba(df)[:, 1] * 100).round(2)

        # Save results with unique filename
        result_filename = f"results_{uuid.uuid4().hex}.csv"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        original_df.to_csv(result_path, index=False)

        # Prepare response
        sample_df = original_df.head(10).copy()
        if len(sample_df) > 0:
            sample_df['Time'] = pd.to_datetime(sample_df['Time'], unit='s').dt.strftime('%H:%M:%S')
            sample_df['Amount'] = sample_df['Amount'].apply(lambda x: f"${x:.2f}")

        fraud_count = sum(predictions)
        total_transactions = len(predictions)
        fraud_percentage = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0

        return render_template(
            'index.html',
            tables=[sample_df.to_html(classes='data', escape=False)],
            frauds=fraud_count,
            total=total_transactions,
            percentage=f"{fraud_percentage:.2f}%",
            result_file=result_filename
        )

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download(filename):
    try:
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(path):
            flash('File not found', 'error')
            return redirect(url_for('index'))
        return send_file(path, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Download error: {str(e)}")
        flash('Error downloading file', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)