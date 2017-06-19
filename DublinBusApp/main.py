from flask import Flask, render_template
from flask_cors import CORS
from busData import BusDB

# View site @ http://localhost:5000/
# --------------------------------------------------------------------------#
# Creating Flask App
app = Flask(__name__)
# Enable Cross Origin Resource Sharing
CORS(app)


# --------------------------------------------------------------------------#
# Index Page
@app.route('/')
def index():

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

# --------------------------------------------------------------------------#
# Setting app to run only if this file is run directly.
if __name__ == '__main__':
    app.run(debug=True)
