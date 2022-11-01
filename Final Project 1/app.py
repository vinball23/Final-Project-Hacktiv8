from flask import Flask, request, render_template
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

model = joblib.load(open('model/linear_regression.pkl', 'rb'))

app = Flask(__name__, template_folder='templates')
    
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='The cab price is ${}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)