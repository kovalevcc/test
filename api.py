from flask import Flask, request, jsonify, send_file
import pandas as pd
import inference
import tempfile
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file and file.filename.endswith('.xlsx'):
            # Load the XLSX file into a pandas DataFrame
            df = pd.read_excel(file, engine='openpyxl')

            prediction1, prediction2 = inference.prediction(df)
            df['Категория'] = prediction1[0]
            df['Уровень рейтинга'] = prediction2[1]

            # Create a temporary file to store the modified DataFrame
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
                df.to_excel(temp_file, index=False, engine='openpyxl')

            # Send the temporary file as a response
            return send_file(temp_file.name, as_attachment=True)

        return jsonify({'error': 'Invalid file format. Please upload an XLSX file.'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            data = request.get_json(force=True)
            text = data.get('text')

            if text is None:
                return jsonify({'error': 'Text data is missing in the request.'})

            prediction1, prediction2 = inference.text_prediction(text)

            return jsonify({'prediction1': prediction1, 'prediction2': prediction2})
        else:
            return 'Content-Type not supported!'

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)
