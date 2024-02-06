from flask import Flask,render_template


import pandas as pd
import pickle
from requests import request

app = Flask(__name__)
car = pd.read_csv('cleaned_car.csv')
model = pickle.load(open("LinearRegressionModel.pkl",'rb'))
@app.route('/')
def index():
  companies = sorted(car['company'].unique())
  car_model = sorted(car['name'].unique())
  year = sorted(car['year'].unique(), reverse = True)
  fuel_type = car['fuel_type'].unique()
  millage =sorted(car['kms_driven'].unique())
  return render_template('index.html',companies = companies, car_model = car_model, year=year,fuel_type = fuel_type)


@app.route('/predict',methods=['POST'])
def predict():
  company = request.form.get('company')
  car_model = request.form.get('car_model')
  year = int(request.form.get('year'))
  fuel_type = request.form.get('fuel_type')
  kms_driven = request.form.get('kilo_driven')
  
  
  prediction = model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]], columns = ['name','company','year','kms_driven','fuel_type'])) 
  return str(prediction[0])

if __name__=="__main__":
    app.run(debug = True)