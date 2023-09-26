import pandas as pd
import numpy as np
import joblib
import json
import pickle
from flask import Flask, request, jsonify
from flask import request,app,jsonify,url_for,render_template

app = Flask(__name__)

# Load your model, dictionaries, and preprocessor
preprocessor = pickle.load(open('preprocessor_base.pkl', 'rb'))
model = joblib.load('xgb_model.pkl')
sales_per_day_dict = json.load(open('sales_per_day_dict_no_zeros', 'r'))
customers_per_day_dict = json.load(open('customers_per_day_dict_no_zeros', 'r'))
sales_per_customers_per_day_dict = json.load(open('sales_per_customers_per_day_dict_no_zeros', 'r'))
# Load other dictionaries as needed

sales_per_day_dict = json.load(open(r'C:\Users\91987\Documents\HandsOnML\projects\Sales_time_series\rossmann-store-sales\sales_per_day_dict_no_zeros','r'))
customers_per_day_dict = json.load(open(r'C:\Users\91987\Documents\HandsOnML\projects\Sales_time_series\rossmann-store-sales\customers_per_day_dict_no_zeros','r'))
sales_per_customers_per_day_dict = json.load(open(r'C:\Users\91987\Documents\HandsOnML\projects\Sales_time_series\rossmann-store-sales\sales_per_customers_per_day_dict_no_zeros','r'))

acceleration_dict = json.load(open(r'C:\Users\91987\Documents\HandsOnML\projects\Sales_time_series\rossmann-store-sales\acceleration_dict_no_zeros','rb'))

freq2_dict = json.load(open(r'C:\Users\91987\Documents\HandsOnML\projects\Sales_time_series\rossmann-store-sales\freq2_dict_no_zeros','r'))
freq3_dict = json.load(open(r'C:\Users\91987\Documents\HandsOnML\projects\Sales_time_series\rossmann-store-sales\freq3_dict_no_zeros','r'))
amp2_dict = json.load(open(r'C:\Users\91987\Documents\HandsOnML\projects\Sales_time_series\rossmann-store-sales\amp2_dict_no_zeros','r'))
amp3_dict = json.load(open(r'C:\Users\91987\Documents\HandsOnML\projects\Sales_time_series\rossmann-store-sales\amp3_dict_no_zeros','r'))


@app.route('/')
def home():
    return render_template('home.html')

def final_func_1(X):
    '''
    This function takes raw X as input and returns the predicitons.

    All the preprocessing will be done in this function using the pipeline
    that we created to preprocess the data.
    '''
    if X['Open']==1:
        global preprocessor
        global model
        global holidays_dict
        global sales_per_day_dict
        global customers_per_day_dict
        global sales_per_customers_per_day_dict
        global acceleration_dict
        global freq2_dict
        global freq3_dict
        global amp2_dict
        global amp3_dict
        global store_state_dict

        X = pd.DataFrame(data=[X.values], columns=X.index)
        store = X['Store'][0]

        #Adding date features.
        X['Date'] = pd.to_datetime(X['Date'])
        X['Year'] = X['Date'].dt.year
        X['Month'] = X['Date'].dt.month
        X['Day'] = X['Date'].dt.day
        X['WeekOfYear'] = X['Date'].dt.isocalendar().week
        X['DayOfYear'] = X['Date'].dt.dayofyear

        X['SalesPerDay'] = sales_per_day_dict[f'{store}']
        X['Customers_per_day'] = customers_per_day_dict[f'{store}']
        X['Sales_Per_Customers_Per_Day'] = sales_per_customers_per_day_dict[f'{store}']

        #Splitting PromoInterval into parts. For ex: (Jan,March,May) --> (Jan), (March), (May).
        s = X['PromoInterval'].str.split(',').apply(pd.Series, 1)
        s.columns = ['PromoInterval0', 'PromoInterval1', 'PromoInterval2', 'PromoInterval3']
        X = X.join(s)

        #Converting Promointerval columns to numerical.
        month_to_num_dict = {
                            'Jan' : 1,
                            'Feb' : 2,
                            'Mar' : 3,
                            'Apr' : 4,
                            'May' : 5,
                            'Jun' : 6,
                            'Jul' : 7,
                            'Aug' : 8,
                            'Sept' : 9, 
                            'Oct' : 10,
                            'Nov' : 11,
                            'Dec' : 12,
                            'nan' : np.nan
                            }

        X['PromoInterval0'] = X['PromoInterval0'].map(month_to_num_dict)
        X['PromoInterval1'] = X['PromoInterval1'].map(month_to_num_dict)
        X['PromoInterval2'] = X['PromoInterval2'].map(month_to_num_dict)
        X['PromoInterval3'] = X['PromoInterval3'].map(month_to_num_dict)

        #Removing PromoInterval feature as no further use now.
        del X['PromoInterval']

        X['Acceleration'] = acceleration_dict[f'{store}']
        #X['State'] = store_state_dict[store]

        X['Promo_before_days'] = np.nan
        X['Promo_after_days'] = np.nan
        
        X['Frequency_2'] = freq2_dict[f'{store}']
        X['Frequency_3'] = freq3_dict[f'{store}']
        X['Amplitude_2'] = amp2_dict[f'{store}']
        X['Amplitude_3'] = amp3_dict[f'{store}']

        if X['Promo2SinceWeek'][0] == np.nan:  
            X['Promo2SinceWeek'][0] = -1
        
        if X['Promo2SinceYear'][0] == np.nan:
            X['Promo2SinceYear'][0] = -1

        if X['PromoInterval0'][0] == np.nan:
            X['PromoInterval0'][0] = -1

#         state = X['State'][0]
#         temp_df = pd.read_csv(f'/content/drive/My Drive/Rossmann Sales Forecasting/weather data/{state}.csv',
#                     sep=';', parse_dates=['Date'],
#                         date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')))
#         X = pd.merge(X, temp_df, how='left', on='Date')

        del X['Date']
        del X['Open']

        X = preprocessor.transform(X)

        prediction = model.predict(X)
    
    else:
        predictiaon = 0

    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("HII")
        
        # Define a dictionary that maps field names to their data types
        field_data_types = {
            'Store': int,
            'DayOfWeek': int,
            'Date': str,  # Keep date as a string for now
            'Customers': int,
            'Open': int,
            'Promo': int,
            'StateHoliday': str,
            'SchoolHoliday': int,
            'StoreType': str,
            'Assortment': str,
            'CompetitionDistance': float,
            'CompetitionOpenSinceMonth': float,
            'CompetitionOpenSinceYear': float,
            'Promo2': int,
            'Promo2SinceWeek': float,
            'Promo2SinceYear': float,
            'PromoInterval': str
        }

        # Convert values based on field names
        data = [field_data_types[field](value) if field in field_data_types else value for field, value in request.form.items()]

        # Create a DataFrame from the list
        test_df = pd.DataFrame([data], columns=request.form.keys())

        print(test_df)
        output = final_func_1(test_df.iloc[0])
        return render_template("home.html",prediction_text="The predicted value of Sales is {}".format(output))
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)

    
