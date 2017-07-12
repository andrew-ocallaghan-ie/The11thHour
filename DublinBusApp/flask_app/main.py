from flask import Flask, render_template, request, g, flash, redirect, url_for, session
from flask_cors import CORS
from busData import InfoDB
import datetime
from sqlalchemy import create_engine
from wtforms import Form, StringField, PasswordField, validators
from passlib.hash import sha256_crypt
import pymysql
from functools import wraps
import requests
from sklearn.externals import joblib
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


# --------------------------------------------------------------------------#
def get_common_routes(src_stop_num, dest_stop_num):
    """Finds common routes between two bus stops
    Returns a dictionary of routes, easier to loop through"""
    route_options = {}

    engine = get_db()
    sql = "SELECT Route, Direction FROM All_routes.all_routes WHERE Stop_ID = %s AND Route IN (SELECT Route FROM All_routes.all_routes WHERE Stop_ID = %s);"
    result = engine.execute(sql, (src_stop_num, dest_stop_num))
    all_data = result.fetchall()

    for row in all_data:
        route_options[row[0].strip('\n')] = (row[1])

    return route_options

# --------------------------------------------------------------------------#
def stops_between_src_dest(src_stop_num, dest_stop_num, route):
    """Finds out how many stops are between two stops on a given route"""
    engine = get_db()
    sql = "SELECT Stop_sequence FROM All_routes.all_routes WHERE (Stop_ID = %s AND Route = %s) OR (Stop_ID = %s AND Route = %s);"
    result = engine.execute(sql, (src_stop_num, int(route), dest_stop_num, int(route)))
    all_data = result.fetchall()

    src_stop_sequence = all_data[0][0]
    dest_stop_sequence = all_data[1][0]

    stops_travelled = dest_stop_sequence - src_stop_sequence

    return stops_travelled

# --------------------------------------------------------------------------#
# Index Page
@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        origin = request.form['origin']
        destination = request.form['destination']

        # This will be taken out once we take in addresses rather than bus stop IDs
        source_num = int(request.form['origin'])
        dest_num = int(request.form['destination'])

        current_time = datetime.datetime.now()
        current_weekday = datetime.datetime.now().weekday()

        route_options = get_common_routes(source_num, dest_num)

        for route, direction in route_options.items():

            predictor = joblib.load('static/pkls/beta' + route + '.csvrf_regressor.pkl')
            direction = direction
            timestamp = 1357044910000000
            stops_travelled = stops_between_src_dest(source_num, dest_num, route)

            time_pred = predictor.predict([1, current_weekday, direction, timestamp, stops_travelled])

            arrival_time = current_time + datetime.timedelta(minutes = float(time_pred[0]))
            arrival_time_hours = arrival_time.hour
            arrival_time_minutes = arrival_time.minute

            # These values are returned from here so that the HTML page doesn't show info until
            # the user has inputted values
            html1 = origin + " to " + destination
            html2 = "Estimated journey time is: " + str(time_pred[0]) + " minutes."
            html3 = "You will arrive in " + destination + " at: " + str(arrival_time.hour) + ":" + str(arrival_time_minutes)

    return render_template('home.html', **locals())

# --------------------------------------------------------------------------#
# Search for Route Page
@app.route('/route_search', methods=['GET', 'POST'])
def route_search():
    """Takes the input from the user for route number and direction"""

    if request.method == 'POST':
        users_route = request.form['user_route']
        # if request.form.get('direction') == 'on':
        #     direction = "southbound"
        # else:
        #     direction = "northbound"

        if request.form.get('direction') == 'on':
            direction = 1
        else:
            direction = 0

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


# --------------------------------------------------------------------------#
# User login page
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        #Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']
        
        engine = get_db()
        sql = "SELECT * FROM users WHERE username = %s"
        result = engine.execute(sql, [username])
        all_data=result.fetchall()
        
        if len(all_data) > 0:
            # Get stored hash
            data = all_data[0]
            password = data['password']
            
        
            #Compare passwords
            if sha256_crypt.verify(password_candidate, password):
                #Passed
                session['logged_in'] = True
                session['username'] = username
                flash('Your are now logged in','success')
                return redirect(url_for('myroutes'))
            else:
                error = 'Invalid login'
                return render_template('login.html', error=error)
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)
    return render_template('login.html')

# --------------------------------------------------------------------------#
# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap

# --------------------------------------------------------------------------#
# Dashboard
@app.route('/myroutes')
@is_logged_in
def myroutes():
    return render_template('myroutes.html')

# --------------------------------------------------------------------------#
# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You are now logged out','success')
    return redirect(url_for('login'))
# --------------------------------------------------------------------------#


# =================================== API ==================================#
# An API is used to allow the website to dynamically query the DB without
# having to be refreshed.

#   - /api/routes/all_routes             -> returns all route information
#   - /api/routes/routenum/direction     -> returns all stops associated with route in a given direction
#   - /api/routes/stations               -> returns all stations (Luas, Dart, Bike)
#   - /api/routes/stops                  -> returns all stops for the stop_search autocomplete
# --------------------------------------------------------------------------#


@app.route('/api/all_routes/', methods=['GET'])
def get_route_info():
    return InfoDB().bus_route_info()

# --------------------------------------------------------------------------#


@app.route('/api/routes/<string:routenum>/<string:direction>/', methods=['GET'])
def get_stop_info(routenum, direction):
    return InfoDB().bus_stop_info_for_route(routenum, direction)

# --------------------------------------------------------------------------#


@app.route('/api/stations/', methods=['GET'])
def get_all_info():
    return InfoDB().all_stop_info()

# --------------------------------------------------------------------------#


@app.route('/api/stops/', methods=['GET'])
def get_all_stop_info():
    return InfoDB().all_bus_stop_info()


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