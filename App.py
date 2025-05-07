from flask import Flask, request, jsonify
import joblib
import main

app = Flask(__name__)
model, vectorizer = main.load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    result = "Fake" if prediction == 1 else "Real"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
