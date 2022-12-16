from flask import Flask, render_template,request
import pandas as pd
import joblib



# create the object of Flask
app = Flask(__name__)


# creating our routes
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():

    
    if request.method == 'GET':
        return render_template('predict.html')

    if request.method == 'POST':
        fuel = request.form['fuel']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = request.form['owner']
        year = request.form['year']
        km_driven = request.form['km_driven']
        

        model = request.form['models']

        res = pd.DataFrame(data = 
            {'fuel':[fuel],'year':[year],
             'km_driven':[km_driven],'seller_type':[seller_type],
              'transmission':[transmission],'owner':[owner]})
    
        
        if model == "1":
            dt = joblib.load("Models/cars_decision_tree_model.pkl")
            sonuc = str(int(dt.predict(res))).strip('[]')

        if model == "2":
            rf = joblib.load("Models/cars_random_forest_model.pkl")
            sonuc = str(int(rf.predict(res))).strip('[]')

        if model == "3":
            lr = joblib.load("Models/cars_linear_regression_model.pkl")
            sonuc = str(int(lr.predict(res))).strip('[]')

        return render_template('predict.html', sonuc = sonuc )

# run flask app
if __name__ == "__main__":
    app.run(debug=True)
