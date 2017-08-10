# http://flask.pocoo.org/
from flask import Flask, render_template, request, g, flash, redirect, url_for, session, Markup

# https://flask-cors.readthedocs.io/en/latest/
from flask_cors import CORS

from busData import api, dbi, everything

# https://docs.python.org/3/library/datetime.html
from _datetime import datetime

# https://www.sqlalchemy.org/
from sqlalchemy import create_engine

# https://wtforms.readthedocs.io/en/latest/
from wtforms import Form, StringField, PasswordField, validators

# https://passlib.readthedocs.io/en/stable/
from passlib.hash import sha256_crypt

# http://pymysql.readthedocs.io/en/latest/index.html
import pymysql

# https://docs.python.org/3/library/functools.html
from functools import wraps

# http://www.pythonforbeginners.com/requests/using-requests-in-python
import requests

# http://pandas.pydata.org/
import pandas as pd

#pymysql.install_as_MySQLdb()

from flask import  jsonify
import json

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
    username = StringField('Username', [validators.length(min=4, max=25)])
    email = StringField('Email', [validators.length(min=4, max=50)])
    work = StringField('Work')
    home = StringField('home')
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField('Confirm Password')


# --------------------------------------------------------------------------#
# Index Page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        src = request.form['origin']
        dest = request.form['destination']
        now_arrive_depart_selection = request.form['now_arrive_depart']
        if now_arrive_depart_selection == '0':
            time = datetime.now()
            date = time.date()
            weekday = time.weekday()
            hour = time.hour
            min = time.minute
            # This needs to be edited to be able to handle the two different use cases
            # i.e. Arrive By & Depart At. This is a good base for taking in the code though
        else:
            time = request.form['time'].split(":")
            date = request.form['date'].split("-")
            year = int(date[0])
            month = int(date[1])
            day = int(date[2])
            weekday = datetime(year, month, day).weekday()
            hour = int(time[0])
            min = int(time[1])
            time = datetime(year, month, day, hour, min)
        
        #THE DICTIONARY!
        #take googleplaces api call from everything and keep it here.
        route_options = everything(src, dest, time)

        return render_template('route_options.html', **locals())

    return render_template('home.html', **locals())


# --------------------------------------------------------------------------#
# Search for Route Page
@app.route('/route_search', methods=['GET', 'POST'])
def route_search():
    """Takes the input from the user for route number and direction"""

    if request.method == 'POST':
        users_route = request.form['user_route']

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
        work = form.work.data
        home = form.home.data
        password = sha256_crypt.encrypt(str(form.password.data))

        engine = get_db()
        sql = "INSERT INTO users(name, email, username, home, work, password) VALUES(%s, %s, %s, %s, %s, %s)"
        engine.execute(sql, (name, email, username, work, home, password))
        flash('You are successfully registered, now you can log in', 'success')
    return render_template('register.html', form=form)


# --------------------------------------------------------------------------#
# User login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']
        print(username)

        engine = get_db()
        sql = "SELECT * FROM users WHERE username = %s"
        result = engine.execute(sql, [username])
        all_data = result.fetchall()

        if len(all_data) > 0:
            # Get stored hash
            data = all_data[0]
            password = data['password']

            # Compare passwords
            if sha256_crypt.verify(password_candidate, password):
                # Passed
                session['logged_in'] = True
                session['username'] = username
                flash('Your are now logged in', 'success')
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
    all_data = result.fetchall()

    stopnamelist = [0] * len(all_data)
    stopidlist = [0] * len(all_data)
    # print(stopnamelist)
    for a in range(0, len(all_data)):
        stopidlist[a] = all_data[a][1]
    Length = len(all_data)
    for a in range(0, len(all_data)):
        print(all_data[a]['stop_id'])
        # stopnamelist[a]=all_data[a]['stop_id']
        sql = "SELECT Stop_name FROM Stops WHERE Stop_ID = %s"
        stopnamelist[a] = engine.execute(sql, [all_data[a]['stop_id']]).fetchall()[0][0]
    print(stopnamelist)
    print(stopidlist)
    return render_template('myroutes.html', **locals())


# --------------------------------------------------------------------------#
# delete the stop
@app.route('/delete', methods=['POST'])
@is_logged_in
def delete():
    if request.method == 'POST':
        username = session['username']
        print(username)
        stop_id = request.form['user_delete']

        print(stop_id, username)
        engine = get_db()

        sql = "DELETE FROM like_stop WHERE username = %s AND stop_id = %s"
        result = engine.execute(sql, [username, stop_id])

        return redirect(url_for('myroutes'))

    return redirect(url_for('myroutes'))


# --------------------------------------------------------------------------#
# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('index'))


# --------------------------------------------------------------------------#
# user like function
@app.route('/likestop', methods=['POST'])
@is_logged_in
def likestop():
    if request.method == 'POST':
        stop_id = request.form['stopnum']

        username = session['username']

        print(stop_id, username)
        engine = get_db()

        sql = "SELECT * FROM like_stop WHERE username = %s AND stop_id = %s"
        result = engine.execute(sql, [username, stop_id])
        all_data = result.fetchall()

        if len(all_data) > 0:
            flash('You have already added the stop into Myroutes', 'danger')
        else:
            sql = "INSERT INTO like_stop(username, stop_id) VALUES( %s, %s)"
            engine.execute(sql, (username, stop_id))
            flash('Congrats! you have added this stop into Myroutes', 'success')

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


# --------------------------------------------------------------------------#
@app.route("/api/bike_data")
def get_bikes():
    bikes = dbi().bikes()
    return json.dumps(bikes)



# =================================== DB ==================================#



URI = "bikesdb.cvaowyzhojfp.eu-west-1.rds.amazonaws.com"
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
app.secret_key='secret123'
app.config['SESSION_TYPE'] = 'filesystem'
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')