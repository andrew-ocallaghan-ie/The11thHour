from flask import jsonify
from operator import itemgetter
from sqlalchemy import create_engine
from flask import g
import pandas as pd
import datetime
import requests
import traceback
import numpy as np
import csv
import json
import datetime
from sklearn.externals import joblib
from tkinter.constants import CURRENT

# --------------------------------------------------------------------------#

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


# --------------------------------------------------------------------------#

class api:
    '''Class that deals with querying DB for API construction.'''

    def __init__(self):
        pass

        # ----------------------------------------------------------------------------------#

    def bus_route_info(self):
        """Returns a JSON list of all the bus routes"""
        routes = []

        engine = get_db()
        sql = "SELECT Route, Origin, Destination\
              FROM All_routes.Routes \
              WHERE Direction = 0 ORDER BY abs(Route);"
        result = engine.execute(sql)
        all_data = result.fetchall()

        for row in all_data:
            route = {
                'route': row[0],
                'origin': row[1],
                'destination': row[2]
            }
            routes.append(route)

        # Returning info
        return jsonify({'route': routes})

        # --------------------------------------------------------------------------#

    def bus_stop_info_for_route(self, routenum, direction):
        """Returns a JSON list with info about a given stop in a given direction"""

        engine = get_db()
        sql = "SELECT st.Stop_ID, Stop_name, Lat, Lon, Stop_sequence, Routes_serviced, Direction \
               FROM All_routes.Stops st, All_routes.Sequence sq \
               WHERE st.Stop_ID = sq.Stop_ID AND Route = %s AND Direction = %s;"
        result = engine.execute(sql, (routenum, direction))
        all_data = result.fetchall()

        stops = []

        for row in all_data:
            stop = {
                'id': int(row[0]),
                'name': row[1],
                'lat': row[2],
                'lon': row[3],
                'order': int(row[4]),
                'other_routes': row[5]
            }
            stops.append(stop)

        stops = sorted(stops, key=itemgetter('order'))

        # Returning info
        return jsonify({'stops': stops})



        # --------------------------------------------------------------------------#

    def all_bus_stop_info(self):
        """Returns the stop information as a JSON dictionary. This is to make it faster to lookup individual stops."""

        engine = get_db()
        sql = "SELECT Stop_ID, Stop_name, Lat, Lon, Routes_serviced FROM All_routes.Stops ORDER BY abs(Stop_ID);"
        result = engine.execute(sql)
        all_data = result.fetchall()
        stops = {}

        for row in all_data:
            stop = {
                int(row[0]): (row[1], row[2], row[3], row[4])
            }
            stops.update(stop)

        # Returning info
        return jsonify({'stops': stops})

    # --------------------------------------------------------------------------#

    def all_stop_info(self):
        engine = get_db()
        sql = "SELECT Address, Lat, Lon, Category, Colour FROM All_routes.dublinbike_dart_luas;"
        result = engine.execute(sql)
        all_data = result.fetchall()
        stops = []

        for row in all_data:
            stop = {
                'name': row[0],
                'latitude': float(row[1]),
                'longitude': float(row[2]),
                'category': row[3],
                'color': row[4]
            }
            stops.append(stop)

        # Returning info
        return jsonify({'stops': stops})

        # --------------------------------------------------------------------------#


class dbi:
    '''Class that deals with querying DB, for nonAPI purposes'''

    def __init__(self):
        pass

        # ----------------------------------------------------------------------------------#

    def location_from_address(self, address):
        """Gets latitude & longitude from an address Returns a tuple of lat/long
        Currently only takes the first search result"""
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

    def find_nearby_stops(self, src, dest):
        """
        Finds out the nearest stops to a given point
        Returns route, stop IDs, direction, seq in a list
        Todo include the stop names and max stop seq and scheduled number of stops to return the scheduled speed
        """
        src_lat, src_lon = self.location_from_address(src)
        dest_lat, dest_lon = self.location_from_address(dest)
        engine = get_db()
        radius = 0.3

        """use 6371 as constant and drop degree conv."""
        sql = "SELECT  start_route, start_direction, start_stop, end_stop, start_stop_seq, end_stop_seq, distance_in_km_start, distance_in_km_end\
        FROM(SELECT DISTINCT s.Route as start_route, s.Direction as start_direction, s.Stop_ID as start_stop, s.Stop_sequence as start_stop_seq, 111.111 *\
                DEGREES(ACOS(COS(RADIANS(%s))\
        * COS(RADIANS(s.Lat))\
        * COS(RADIANS(%s - s.Lon))\
        + SIN(RADIANS(%s))\
        * SIN(RADIANS(s.Lat))))  AS 'distance_in_km_start'\
                    FROM All_routes.new_all_routes s\
                    HAVING distance_in_km_start< %s) As Start\
                    JOIN\
        (SELECT DISTINCT e.Route as end_route, e.Direction as end_direction , e.Stop_ID as end_stop, e.Stop_sequence as end_stop_seq, 111.111 * \
        DEGREES(ACOS(COS(RADIANS(%s))\
        * COS(RADIANS(e.Lat))\
        * COS(RADIANS(%s - e.Lon))\
        + SIN(RADIANS(%s))\
        * SIN(RADIANS(e.Lat))))  AS 'distance_in_km_end'\
                    FROM All_routes.new_all_routes e\
                    HAVING distance_in_km_end< %s) as end\
        WHERE start_route=end_route AND start_stop_seq<end_stop_seq AND start_direction=end_direction"

        result = engine.execute(sql, (src_lat, src_lon, src_lat, radius, dest_lat, dest_lon, dest_lat, radius))
        all_data = result.fetchall()

        dataframe = pd.DataFrame(all_data,
                                 columns=["Route", "Direction", "Start_Stop_ID", "End_Stop_ID", "Start_Stop_Sequence",
                                          "End_Stop_Sequence", "Distance_in_km_from_start", "Distance_in_km_from_end"])
        print(dataframe)
        return self.priority_options(dataframe)

    def priority_options(self, dataframe):
        #this is here to select the nearest of the subset returned by sql to the user - minimises total user walking distance
        dataframe['low_score'] = dataframe["Distance_in_km_from_start"]+dataframe["Distance_in_km_from_end"]
        dataframe = dataframe.loc[dataframe.groupby('Route').low_score.idxmin()]
        print('priority dataframe', dataframe)
        return self.dataframe_to_dict(dataframe)

    def dataframe_to_dict(self, dataframe):
        #this converts the interesting routes to a dictionary
        route_options = dataframe.transpose().to_dict()
        i = 1
        break_at = len(route_options)
        for option in route_options:
            if i > break_at:
                break
            route_options["Option " + str(i)] = route_options.pop(option)
            i += 1
        print (route_options)
        return route_options

    # --------------------------------------------------------------------------#

    def scrape_weather(self):
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
        # current_rain = data['hourly_forecast'][0]['qpf']['metric']
        # Windspeed
        current_wind = data['hourly_forecast'][0]['wspd']['metric']

        # Returning Summary
        return (current_temp, current_wind)

    # --------------------------------------------------------------------------#
    def bikes(self):
        APIKEY = 'a360b2a061d254a3a5891e4415511251899f6df1'
        NAME = "Dublin"
        STATIONS_URI = "https://api.jcdecaux.com/vls/v1/stations"
        r = requests.get(STATIONS_URI, params={"apiKey": APIKEY,
                                               "contract": NAME})
        data = (json.loads(r.text))

        return(data)

    # --------------------------------------------------------------------------#
    def extract_holidays(self):
        '''Returns a list of school holidays'''
        holidays = []
        engine = get_db()

        sql = "SELECT * FROM All_routes.School_Holidays"

        result = engine.execute(sql)
        all_data = result.fetchall()

        for row in all_data:
            holidays += row

        holidays = [datetime.datetime.strptime(x, '%d/%m/%Y').date() for x in holidays]

        return (holidays)

    # --------------------------------------------------------------------------#
    def get_max_sequence(self, route, direction):
        """Return the max stop sequence for a route in a direction"""
        engine = get_db()

        sql = "SELECT MAX(Stop_sequence) FROM All_routes.Sequence WHERE Route = %s AND Direction = %s;"

        result = engine.execute(sql, (route, direction))
        all_data = result.fetchone()

        max_seq = all_data[0]
        return max_seq

    # --------------------------------------------------------------------------#
    def get_sched_time(self, route, direction):
        """Return the max stop sequence for a route in a direction"""
        engine = get_db()

        sql = "SELECT Scheduled_Overall_Time FROM All_routes.routes_to_add_time WHERE Route = %s AND Direction = %s;"

        result = engine.execute(sql, (route, direction))
        all_data = result.fetchone()

        time = all_data[0]
        return time


def everything(src, dest):
    """Determines the journey time for each viable route option
    Returns a list of options ordered by their journey time"""

    route_options = dbi().find_nearby_stops(src, dest)
    current_time = datetime.datetime.now()
    current_weekday = current_time.weekday()
    current_hour = current_time.hour
    current_min = current_time.minute
    bit1 = "1" if (current_min > 15) else "0"
    bit2 = "1" if (current_min > 30) else "0"
    bit3 = "1" if (current_min > 45) else "0"
    time_bin = str(current_hour) + bit1 + bit2 + bit3

    current_date = current_time.date()
    if current_date in dbi().extract_holidays():
        is_school_holiday = 1
    else:
        is_school_holiday = 0

    weather = dbi().scrape_weather()
    current_temp, current_wind = weather

    for option in route_options.keys():
        print(option)
        '''todo add stop name to dict'''
        direction = int(route_options[option]['Direction'])
        route = route_options[option]['Route']
        src_stop_seq = route_options[option]['Start_Stop_Sequence']
        dest_stop_seq = route_options[option]['End_Stop_Sequence']
        stops_to_travel = dest_stop_seq - src_stop_seq
        '''consider mergine this with bigger call, overhead issue?'''
        max_stop_seq = dbi().get_max_sequence(route, direction)
        scheduled_time = dbi().get_sched_time(route, direction)
        sched_speed_per_stop = scheduled_time / max_stop_seq
        predictor = joblib.load('static/pkls/xbeta' + route + '.csvrf_regressor.pkl')
        '''use lambda or zip to int all inputs, its cleaner'''
        params = [
            [int(current_weekday), int(time_bin), int(current_wind), int(current_temp), int(is_school_holiday),
             int(sched_speed_per_stop), int(stops_to_travel), int(src_stop_seq)]]

        time_pred = predictor.predict(params)[0]
        print(time_pred, 'time_pred on main page')
        route_options[option]['html'] = "<div data-toggle='collapse' data-target='#map'><div class='option_route' onclick='boxclick(this, 1)'>" + route + "</div><div class='option_src_dest'>" + str(
            'i should be a name') + " to " + 'i too should be a name' + "</div><div class='option_journey_time'>" + str(
            int(time_pred)) + "</div></div>"

    return route_options