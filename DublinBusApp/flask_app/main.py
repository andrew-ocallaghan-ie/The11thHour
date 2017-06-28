
from flask import Flask, render_template, request
from flask_cors import CORS
from busData import BusDB
from sklearn.externals import joblib
import requests
import traceback
import datetime
import csv

# View site @ http://localhost:5000/
# --------------------------------------------------------------------------#
# Creating Flask App
app = Flask(__name__)
# Enable Cross Origin Resource Sharing
CORS(app)

#--------------------------------------------------------------------------#
def scrape_weather():
    '''Returns a summary of the current weather from the Wunderground API'''
    # API URI
    api = 'http://api.wunderground.com/api'
    # API Parameters
    city = '/IE/Dublin'
    app_id = '/0d675ef957ce972d'
    api_type = '/hourly'

    URI = api + app_id + api_type + '/q' + city + '.json'

    # Loading Data
    try:
        req = requests.get(URI)
        data = req.json()

    except:
        data = []
        print(traceback.format_exc())

    # Temperature
    current_temp = data['hourly_forecast'][0]['temp']['metric']
    # Rainfall
    current_precipitation = data['hourly_forecast'][0]['qpf']['metric']

    # Returning Summary
    return (current_temp, current_precipitation)


#--------------------------------------------------------------------------#
def extract_holidays():
    '''Returns a list of school holidays'''
    stop_info = open('static/school_holidays_2017.csv', 'r')
    reader = csv.reader(stop_info)
    next(reader)
    holidays = []

    for row in reader:
        holidays += row

    holidays = [datetime.datetime.strptime(x, '%d/%m/%Y').date() for x in holidays]

    return(holidays)


# --------------------------------------------------------------------------#
# Index Page
@app.route('/', methods=['GET', 'POST'])
def index():
    predictor = joblib.load('static/rf_regressor.pkl')

    if request.method == 'POST':
        origin_stop_id = request.form['origin']
        desination_stop_id = request.form['destination']
        users_route = request.form['user_route']
        # weather = scrape_weather()
        # current_temp = weather[0]
        # current_rain = weather[1]
        current_hour = datetime.datetime.now().hour

        current_date = datetime.datetime.now().date()
        if current_date in extract_holidays():
            is_school_holiday = 1
        else:
            is_school_holiday = 0

        current_weekday = datetime.datetime.now().weekday()


    # x will be a list of inputs that we give to the predictor: time, rain etc.
    # we then run this through the predictor model to get a predicted delay
    # to send to flask and display to the user

    # x = [1, current_temp, current_rain, current_hour, 0]
    # delay = predictor.predict(x)

    # this needs to be changed to only return the delay value
    return render_template('index.html', **locals())


# =================================== API ==================================#
# An API is used to allow the website to dynamically query the DB without
# having to be refreshed.
#   - /api/routes/routenum     -> returns all stops associated with route
# --------------------------------------------------------------------------#
# --------------------------------------------------------------------------#

# API - Returns JSON file with stop info for bus route.
@app.route('/api/routes/<string:routenum>', methods=['GET'])
def get_stop_info(routenum):
    return BusDB().bus_stop_info(routenum)

# --------------------------------------------------------------------------#


@app.route('/api/all_routes/', methods=['GET'])
def get_route_info():
    return BusDB().bus_route_info()

# --------------------------------------------------------------------------#
# Setting app to run only if this file is run directly.
if __name__ == '__main__':
    app.run(debug=True)
