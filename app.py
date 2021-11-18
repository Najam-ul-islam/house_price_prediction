from flask import Flask, request, jsonify
import pickle
import pandas as pd
from math import floor
app = Flask(__name__)
model = pickle.load(open('regression_model_updated.pkl', 'rb'))


# @app.route('/')
# def home():
#     return render_template('index.html', prediction_text='Please enter the required fields')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        property_type = request.form['property_type']
        city = request.form['city']
        province_name = request.form['province_name']
        baths = request.form['baths']
        purpose = request.form['purpose']
        bedrooms = request.form['bedrooms']
        Area_Size = request.form['Area_Size']
        df = pd.DataFrame([[property_type, city, province_name, baths, purpose, bedrooms, Area_Size]], columns=[
                          'property_type', 'city', 'province_name', 'baths', 'purpose', 'bedrooms', 'Area_Size'], index=['Input'], dtype='float')
        prediction = model.predict(df)[0]
        original_input = {'property_type': property_type, 'city': city, 'province_name': province_name,
                          'baths': baths, 'purpose': purpose, 'bedrooms': bedrooms, 'Area_Sizes': Area_Size, 'Predicted_Price':  floor(prediction)}
    return jsonify(original_input)


if __name__ == '__main__':
    app.run(debug=True)
