from flask import Flask, render_template, request, g, jsonify
from flask_cors import CORS
from busData import BusDB
from sklearn.externals import joblib
import requests
import traceback
import datetime
import csv
from sqlalchemy import create_engine

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
        if current_weekday < 5:
            is_weekend = 0
        else:
            is_weekend = 1


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
def get_route_info(routenum):
    return BusDB().route_info(routenum)


# =================================== EC2 ==================================#
URI="bikesdb.cvaowyzhojfp.eu-west-1.rds.amazonaws.com"
PORT = "3306"
DB = "dbikes"
USER = "teamgosky"
PASSWORD = "teamgosky"
# =================================== EC2 ==================================#


def connect_to_database():
    db_str = "mysql+mysqldb://{}:{}@{}:{}/{}"
    engine = create_engine(db_str.format(USER, PASSWORD, URI, PORT, DB), echo=True)
    return engine
#    db = MySQLdb.connect(host="localhost",user="teamgosky",passwd="teamgosky",db="dbikes")
#    return db

def get_db():
    engine = getattr(g, 'engine', None)
    if engine is None:
        engine = g.engine = connect_to_database()
    return engine

@app.route("/all")
#@functools.lru_cache(maxsize=128)
def get_station():
    engine=get_db()
    sql="select * from station;"
    rows = engine.execute(sql).fetchall()
    print('#found{}stations',len(rows))
    return jsonify(stations=[dict(row.items()) for row in rows])
    

# --------------------------------------------------------------------------#
# Setting app to run only if this file is run directly.
if __name__ == '__main__':
    app.run(debug=True)
