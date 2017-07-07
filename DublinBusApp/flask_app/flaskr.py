import simplejson as json
import os
import MySQLdb
from sqlalchemy import create_engine
from flask import Flask, jsonify, request, session, g, redirect, url_for, abort, \
     render_template, flash, session, logging
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
     
URI="bikesdb.cvaowyzhojfp.eu-west-1.rds.amazonaws.com"
PORT = "3306"
DB = "All_routes"
USER = "teamgosky"
PASSWORD = "teamgosky"
app = Flask(__name__)
app.config['MYSQL_CURSORCLASS']='DictCursor'

def connect_to_database():
    db_str = "mysql+mysqldb://{}:{}@{}:{}/{}"
    engine = create_engine(db_str.format(USER, PASSWORD, URI, PORT, DB), echo=True)
    return engine

def get_db():
    engine = getattr(g, 'engine', None)
    if engine is None:
        engine = g.engine = connect_to_database()
    return engine

@app.route('/')
def show_entries():
    engine=get_db()
    sql="select title, text from entries order by id desc;"
    rows = engine.execute(sql).fetchall()
    entries = [dict(row.items()) for row in rows]
    return render_template('show_entries.html', entries=entries)

    #entries = [dict(title=row[0], text=row[1]) for row in cur.fetchall()]
    #return render_template('show_entries.html', entries=entries)
    
@app.route('/add', methods=['POST'])
def add_entry():
    if not session.get('logged_in'):
        abort(401)
    engine=get_db()
    sql="insert into User (title, text) values (?, ?);"
    engine.execute(sql,[request.form['title'], request.form['text']])
    engine.commit()
    flash('New entry was successfully posted')
    return redirect(url_for('show_entries'))


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out')
    return redirect(url_for('show_entries'))

class RegisterForm(Form):
    name = StringField('Name', [validators.length(min=1, max=50)])
    username  = StringField('Username', [validators.length(min=4, max=25)])
    email = StringField('Email', [validators.length(min=4, max=50)])
    password = PasswordField('Email', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message = 'Passwords do not match')
        ])
    confirm = PasswordField('Confirm Password')
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        return render_template('register.html')
    return render_template('register.html', form=form)

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
            print("Hi",all_data)
            password = data['password']
            print(password)
        
            #Compare passwords
            if sha256_crypt.verify(password_candidate, password):
                #app.logger.info('PASSWORD MATCHED')
                print('matched')
            else:
                #app.logger.info('PASSWORD not MATCHED')
                print('not matched')
        else:
            print('no user')
    return render_template('login.html')


if __name__ == '__main__':
    app.run()