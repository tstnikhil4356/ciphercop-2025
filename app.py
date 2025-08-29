from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained ML model
# Make sure you saved your model as 'model.pkl'
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Example: assume you pass features as a list
    features = data.get("features", [])

    # Reshape input for model
    prediction = model.predict([features])[0]

    return jsonify({"prediction": str(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
