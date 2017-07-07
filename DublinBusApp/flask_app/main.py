# import MySQLdb
from flask import Flask, render_template, request, g, jsonify, flash, redirect, url_for, session, logging
from flask_cors import CORS
from busData import BusDB
from db_luas_dart import alldublinstation
from sklearn.externals import joblib
import requests
import traceback
import datetime
import csv
from sqlalchemy import create_engine
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
import pymysql
pymysql.install_as_MySQLdb()

# View site @ http://localhost:5000/
# --------------------------------------------------------------------------#
# Creating Flask App
app = Flask(__name__)
# Enable Cross Origin Resource Sharing
CORS(app)

# --------------------------------------------------------------------------#


# Class for form
class RegisterForm(Form):
    name = StringField('Name', [validators.length(min=1, max=50)])
    username  = StringField('Username', [validators.length(min=4, max=25)])
    email = StringField('Email', [validators.length(min=4, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message = 'Passwords do not match')
        ])
    confirm = PasswordField('Confirm Password')

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
        origin = request.form['origin']
        destination = request.form['destination']
        # weather = scrape_weather()
        # current_temp = weather[0]
        # current_rain = weather[1]
        current_time = datetime.datetime.now()
        current_hour = current_time.hour

        html1 = origin + " to " + destination
        html2 = "Estimated journey time is: "
        html3 = "You will arrive in " + destination + " at:"

        # current_date = datetime.datetime.now().date()
        # if current_date in extract_holidays():
        #     is_school_holiday = 1
        # else:
        #     is_school_holiday = 0
        #
        current_weekday = datetime.datetime.now().weekday()


    # this needs to be changed to return the delay value
    return render_template('home.html', **locals())

# --------------------------------------------------------------------------#
# Search for Route Page
@app.route('/route_search', methods=['GET', 'POST'])
def route_search():
    """Takes the input from the user for route number and direction"""

    if request.method == 'POST':
        users_route = request.form['user_route']
        if request.form.get('direction') == 'on':
            direction = "southbound"
        else:
            direction = "northbound"

    return render_template('route_search.html', **locals())


# --------------------------------------------------------------------------#
# Search for Stop Page
@app.route('/stop_search', methods=['GET', 'POST'])
def stop_search():
    """Initially this loads the stop_search page. If there is a POST request i.e. the user inputs something
    it will open the stop page of the requested stop"""

    if request.method == 'POST':
        stop_num = request.form['user_stop']
        return render_template('bus_stop.html', **locals())

    return render_template('stop_search.html', **locals())


# --------------------------------------------------------------------------#
# Stop Info Page
@app.route('/stop/<string:stopnum>', methods=['GET', 'POST'])
def stop_info(stopnum):
    """Displays the stop info page. It is activated from the links on the route_search page."""
    stop_num = stopnum

    return render_template('bus_stop.html', **locals())


# --------------------------------------------------------------------------#
# User Registration Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data))

        engine = get_db()
        sql = "INSERT INTO users(name, email, username, password) VALUES(%s, %s, %s, %s)"
        engine.execute(sql, (name, email, username, password))
        flash('You are successfully registered, now you can log in', 'success')
    return render_template('register.html', form=form)


# =================================== API ==================================#
# An API is used to allow the website to dynamically query the DB without
# having to be refreshed.

#   - /api/routes/all_routes             -> returns all routes
#   - /api/routes/routenum/direction     -> returns all stops associated with route in a given direction
#   - /api/routes/stations               -> returns all stations (Luas, Dart, Bike)
#   - /api/routes/stops                  -> returns all stops for the stop_search autocomplete
# --------------------------------------------------------------------------#


@app.route('/api/all_routes/', methods=['GET'])
def get_route_info():
    return BusDB().bus_route_info()

# --------------------------------------------------------------------------#


@app.route('/api/routes/<string:routenum>/<string:direction>/', methods=['GET'])
def get_stop_info(routenum, direction):
    return BusDB().bus_stop_info_for_route(routenum, direction)

# --------------------------------------------------------------------------#


@app.route('/api/stations/', methods=['GET'])
def get_all_info():
    return alldublinstation().all_stop_info()

# --------------------------------------------------------------------------#


@app.route('/api/stops/', methods=['GET'])
def get_all_stop_info():
    return BusDB().all_bus_stop_info()


# =================================== DB ==================================#

URI="bikesdb.cvaowyzhojfp.eu-west-1.rds.amazonaws.com"
PORT = "3306"
DB = "All_routes"
USER = "teamgosky"
PASSWORD = "teamgosky"


def connect_to_database():
    db_str = "mysql+mysqldb://{}:{}@{}:{}/{}"
    engine = create_engine(db_str.format(USER, PASSWORD, URI, PORT, DB), echo=True)
    return engine

def get_db():
    engine = getattr(g, 'engine', None)
    if engine is None:
        engine = g.engine = connect_to_database()
    return engine

# =================================== DB ==================================#

# Setting app to run only if this file is run directly.
if __name__ == '__main__':
    app.secret_key='secret123'
    app.run(debug=True)