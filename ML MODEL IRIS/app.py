#Imports
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#init app
app = Flask(__name__)
model = pickle.load(open('trained_model.pkl', 'rb'))

#homepage
@app.route('/')
def home():
    return render_template('index.html')

#predict page
@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
   
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output =prediction[0]

    return render_template('index.html', prediction_text='The Flower is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)