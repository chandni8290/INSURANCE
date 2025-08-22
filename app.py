from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = joblib.load('model.lb')

# Home page with input form
@app.route('/')
def home():
    return render_template('index.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact form
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(f"📩 Contact: {name} <{email}> — {message}")
        return render_template('contact.html', success=True)
    return render_template('contact.html')

# Prediction logic
@app.route('/project', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Encode categorical values
        sex_code = 1 if sex.lower() == 'male' else 0
        smoker_code = 1 if smoker.lower() == 'yes' else 0
        region_map = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
        region_code = region_map.get(region.lower(), 0)

        # Prepare input
        features = pd.DataFrame([[age, sex_code, bmi, children, smoker_code, region_code]],
                                columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

        # Predict
        prediction = round(model.predict(features)[0], 2)

        # Save to history.csv
        features['Predicted_Cost'] = prediction
        features['Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not os.path.exists("history.csv"):
            features.to_csv("history.csv", index=False)
        else:
            features.to_csv("history.csv", mode='a', header=False, index=False)

        return render_template('project.html', prediction=prediction)
    except Exception as e:
        print("❌ Prediction Error:", e)
        return render_template('project.html', prediction=None)

# Prediction history
@app.route('/history')
def history():
    try:
        df = pd.read_csv("history.csv")
        return render_template("history.html", data=df.to_dict(orient="records"))
    except:
        return render_template("history.html", data=[])

# Run app
if __name__ == '__main__':
    app.run(debug=True)
