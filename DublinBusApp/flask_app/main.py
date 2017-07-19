#http://flask.pocoo.org/
from flask import Flask, render_template, request, g, flash, redirect, url_for, session

#https://flask-cors.readthedocs.io/en/latest/
from flask_cors import CORS

from busData import api, dbi

#https://docs.python.org/3/library/datetime.html
import datetime

#https://www.sqlalchemy.org/
from sqlalchemy import create_engine

#https://wtforms.readthedocs.io/en/latest/
from wtforms import Form, StringField, PasswordField, validators

#https://passlib.readthedocs.io/en/stable/
from passlib.hash import sha256_crypt

#http://pymysql.readthedocs.io/en/latest/index.html
import pymysql

#https://docs.python.org/3/library/functools.html
from functools import wraps

#http://www.pythonforbeginners.com/requests/using-requests-in-python
import requests

#http://scikit-learn.org/stable/
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
def location_from_address(address):
    """Gets latitude & longitude from an address
    Returns a tuple of lat/long
    Currently only takes the first search result
    Should make a way to take many"""
    address = address.replace(" ", "+")
    key = "AIzaSyBVaetyYe44_Ay4Oi5Ljxu83jKLnMKEtBc"
    url = "https://maps.googleapis.com/maps/api/geocode/json?"
    params = {'address': address, 'region': 'IE', 'components': 'locality:dublin|country:IE', 'key': key}

    r = requests.get(url, params=params)
    data = r.json()

    lat = data['results'][0]['geometry']['location']['lat']
    long = data['results'][0]['geometry']['location']['lng']

    location = (lat, long)
    return location

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

        route_options = dbi().get_common_routes(source_num, dest_num)

        for route, direction in route_options.items():

            predictor = joblib.load('static/pkls/beta' + route + '.csvrf_regressor.pkl')
            direction = direction
            timestamp = 1357044910000000
            stops_travelled = dbi().stops_between_src_dest(source_num, dest_num, route)

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
        print(stop_num)
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
        print(username)
        
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
                return render_template('home.html', error=error)
        else:
            error = 'Username not found'
            return render_template('home.html', error=error)
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
    username = session['username']
    engine = get_db()
    sql = "SELECT * FROM like_stop WHERE username = %s"
    result = engine.execute(sql, [username])
    all_data=result.fetchall()
    
    stopnamelist=[0] * len(all_data)
    stopidlist=[0] * len(all_data)
    #print(stopnamelist)
    for a in range(0,len(all_data)):
        stopidlist[a]=all_data[a][1]
    Length=len(all_data)
    for a in range(0,len(all_data)):
        print(all_data[a]['stop_id'])
        #stopnamelist[a]=all_data[a]['stop_id']
        sql = "SELECT Stop_name FROM Stops WHERE Stop_ID = %s"
        stopnamelist[a] = engine.execute(sql, [all_data[a]['stop_id']]).fetchall()[0][0]
    print(stopnamelist)
    print(stopidlist)
    return render_template('myroutes.html',**locals())

# --------------------------------------------------------------------------#
# delete the stop
@app.route('/delete', methods=['POST'])
@is_logged_in
def delete():
    if request.method == 'POST':
        username = session['username']
        print(username) 
        stop_id = request.form['user_delete']
        
        print(stop_id,username)
        engine = get_db()
        
        sql = "DELETE FROM like_stop WHERE username = %s AND stop_id = %s"
        result = engine.execute(sql, [username,stop_id])
            
        return redirect(url_for('myroutes'))

    return redirect(url_for('myroutes'))

# --------------------------------------------------------------------------#
# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You are now logged out','success')
    return redirect(url_for('index'))

# --------------------------------------------------------------------------#
#user like function
@app.route('/likestop', methods=['POST'])
@is_logged_in
def likestop():
    
    if request.method == 'POST':
        stop_id = request.form['stopnum']
    
        username = session['username'] 
        
        print(stop_id,username)
        engine = get_db()
        
        sql = "SELECT * FROM like_stop WHERE username = %s AND stop_id = %s"
        result = engine.execute(sql, [username,stop_id])
        all_data=result.fetchall()
        
        if len(all_data) >0:
            flash('You have already added the stop into Myroutes','danger')
        else:
            sql = "INSERT INTO like_stop(username, stop_id) VALUES( %s, %s)"
            engine.execute(sql, (username, stop_id))
            flash('Congrats! you have added this stop into Myroutes','success')
            
        stop_num = stop_id
        return render_template('bus_stop.html', **locals())

    return render_template('stop_search.html', **locals())

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
    return api().bus_route_info()

# --------------------------------------------------------------------------#


@app.route('/api/routes/<string:routenum>/<string:direction>/', methods=['GET'])
def get_stop_info(routenum, direction):
    return api().bus_stop_info_for_route(routenum, direction)

# --------------------------------------------------------------------------#


@app.route('/api/stations/', methods=['GET'])
def get_all_info():
    return api().all_stop_info()

# --------------------------------------------------------------------------#


@app.route('/api/stops/', methods=['GET'])
def get_all_stop_info():
    return api().all_bus_stop_info()


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
    app.run(debug=True,host='0.0.0.0')