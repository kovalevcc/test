from flask import Flask, request, jsonify, send_file, json
import pandas as pd
import inference
import tempfile
import os
import numpy as np
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

@app.route('/test', methods=['POST'])
def test():
    response = jsonify({'test': 'lorem ipsum'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        #content_type = request.headers.get('Content-Type')
        #if content_type == 'application/json':
            data = request.get_json(force=True)
            text = data.get('text')

            if text is None:
                response = jsonify({'error': 'Text data is missing in the request.'})
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response

            prediction1, prediction2 = inference.text_prediction(text)
            listed_pred1 = prediction1.tolist()
            listed_pred2 = prediction2.tolist()
            response = jsonify({'prediction1': listed_pred1, 'prediction2': listed_pred2})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        # else:
        #     return 'Content-Type not supported!'
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
