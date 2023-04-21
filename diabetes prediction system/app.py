import numpy as np
from flask import Flask , request , render_template
import joblib
input_features =[]
app = Flask(__name__)
model = joblib.load("model_4.pkl")


@app.route('/')
def home():
    return render_template("index1.html")


@app.route('/predict', methods =['POST'])
def predict():
  if request.method=="POST":
        x1 = float(request.form["value1"])
        x2 = float(request.form["value2"])
        x3 = float(request.form["value3"])
        x4 = float(request.form["value4"])
        x5 = float(request.form["value5"])
        x6 = float(request.form["value6"])
        x7 = float(request.form["value7"])
        x8 = float(request.form["value8"])
        input_features.append(x1)
        input_features.append(x2)
        input_features.append(x3)
        input_features.append(x4)
        input_features.append(x5)
        input_features.append(x6)
        input_features.append(x7)
        input_features.append(x8)
        features= [np.array(input_features)]
        prediction = model.predict(features)  
        output = prediction[0]
        if output == 0:
          return render_template('index2.html',prediction_text1 = x1,
                                              prediction_text2 = x2,
                                              prediction_text3 = x3,
                                              prediction_text4 = x4,
                                              prediction_text5 = x5,
                                              prediction_text6 = x6,
                                              prediction_text7 = x7,
                                              prediction_text8 = x8,
                                              prediction_text9 = "Diabetic")
        else:
          return render_template('index2.html',prediction_text1 = x1,
                                              prediction_text2 = x2,
                                              prediction_text3 = x3,
                                              prediction_text4 = x4,
                                              prediction_text5 = x5,
                                              prediction_text6 = x6,
                                              prediction_text7 = x7,
                                              prediction_text8 = x8,
                                              prediction_text9 = "Non-Diabetic")


if __name__=="__main__":
    app.run(debug=True)
