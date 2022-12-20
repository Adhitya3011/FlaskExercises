from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

#loading model from pickle
model = pickle.load(open("trained_model.pkl", "rb"))

#defining home page 
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    preds = model.predict(features)
    output = preds[0]
    print(output)
    return render_template("index.html", prediciton_text = "Flower Species: {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)