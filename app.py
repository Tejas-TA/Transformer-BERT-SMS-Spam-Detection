from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np

# Load the Multinomial Naive Bayes' pkl model and TF|IDF Vectorizer using joblib
classifier = joblib.load('SMS_Detection_Model.pkl')
cv = joblib.load('TF_IDF.pkl')

app = Flask(__name__, template_folder='templates')

#Default routing when server starts
@app.route('/')
def home():
	return render_template('index.html')

#Navigating to predict. At this step, our SMS is transformed to array using TF|IDF vectorizer
#and predicted using Multinomial Bayes' Model(classifier). After that, the result.html displays the output
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	SMS = request.form['SMS']
    	data = [SMS]
    	vectorized = cv.transform(data).toarray()
    	my_prediction = classifier.predict(vectorized)
    	return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    # Use below for local flask deployment
    app.run(debug=True)
    
    #Use below for AWS EC2 deployment
    #app.run(host='0.0.0.0',port=8080)