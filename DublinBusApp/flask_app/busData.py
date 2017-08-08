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
from bs4 import BeautifulSoup
import urllib.request

# http://scikit-learn.org/stable/
from sklearn.externals import joblib

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
               FROM All_routes.new_all_routes st, All_routes.Sequence sq \
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
        sql = "SELECT Stop_ID, Stop_name, Lat, Lon, Routes_serviced FROM All_routes.new_all_routes ORDER BY abs(Stop_ID);"
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
        sql = 'SELECT Address, Lat, Lon, Category, Colour,Stop_Id FROM All_routes.dublinbike_dart_luas;'
        result = engine.execute(sql)
        all_data = result.fetchall()
        stops = []

        for row in all_data:
            stop = {
                'name': row[0],
                'latitude': float(row[1]),
                'longitude': float(row[2]),
                'category': row[3],
                'color': row[4],
                'stop_id': row[5]
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

    def find_darts(self, src, dest):
        """
        Finds darts
        """
        src_lat, src_lon = self.location_from_address(src)
        dest_lat, dest_lon = self.location_from_address(dest)
        engine = get_db()
        radius = 0.8
        """use 6371 as constant and drop degree conv."""
        sql = "SELECT  distance_in_km_start_dart, distance_in_km_end_dart, start_dart_name, end_dart_name, start_cat, end_cat\
         FROM(SELECT  s.Address as start_dart_name, s.category as start_cat, 111.111 *\
                 DEGREES(ACOS(COS(RADIANS(%s))\
         * COS(RADIANS(s.Lat))\
         * COS(RADIANS(%s - s.Lon))\
         + SIN(RADIANS(%s))\
         * SIN(RADIANS(s.Lat))))  AS 'distance_in_km_start_dart'\
                     FROM All_routes.dublinbike_dart_luas s\
                     HAVING distance_in_km_start_dart< %s) As start_dart\
                     JOIN\
         (SELECT DISTINCT e.Address as end_dart_name, e.category as end_cat,111.111 * \
         DEGREES(ACOS(COS(RADIANS(%s))\
         * COS(RADIANS(e.Lat))\
         * COS(RADIANS(%s - e.Lon))\
         + SIN(RADIANS(%s))\
         * SIN(RADIANS(e.Lat))))  AS 'distance_in_km_end_dart'\
                     FROM All_routes.dublinbike_dart_luas e\
                     HAVING distance_in_km_end_dart< %s) As end_dart\
         WHERE start_cat='luas'  OR start_cat='dart' and start_cat = end_cat"
        dart_result = engine.execute(sql, (src_lat, src_lon, src_lat, radius, dest_lat, dest_lon, dest_lat, radius))
        dart_all_data = dart_result.fetchall()
        dart_dataframe = pd.DataFrame(dart_all_data,
                                 columns=["dist_dart_start", "dist_dart_end", "start_dart_name", "end_dart_name", "start_cat", "end_cat"])
        dart_dataframe['low_score_dart'] = dart_dataframe["dist_dart_start"] + dart_dataframe["dist_dart_end"]
        dart_dataframe['walking_dist_dart'] = int(dart_dataframe.low_score_dart.values[0] / 0.05)
        dart_dataframe = dart_dataframe.loc[dart_dataframe.low_score_dart.idxmin()]
        dart_dataframe.drop(['low_score_dart','dist_dart_start','dist_dart_end', 'end_cat'])
        print ("dart_dataframe",dart_dataframe)
        return dart_dataframe


    def find_nearby_stops(self, src, dest):
        """
        Finds out the nearest stops to a given point
        Returns route, stop IDs, direction, seq in a list
        Todo include the stop names and max stop seq and scheduled number of stops to return the scheduled speed
        """

        src_lat, src_lon = self.location_from_address(src)
        dest_lat, dest_lon = self.location_from_address(dest)
        self.mid_point_lat = (src_lat + dest_lat) / 2
        self.mid_point_lon = (src_lon + dest_lon) / 2
        engine = get_db()
        radius = 0.3

        """use 6371 as constant and drop degree conv."""
        sql = "SELECT  start_route, start_direction, start_stop, end_stop, start_stop_seq, end_stop_seq, distance_in_km_start, distance_in_km_end, start_stop_name, end_stop_name\
        FROM(SELECT DISTINCT s.Route as start_route, s.Direction as start_direction, s.Stop_ID as start_stop, s.Stop_sequence as start_stop_seq, s.Stop_name as start_stop_name, 111.111 *\
                DEGREES(ACOS(COS(RADIANS(%s))\
        * COS(RADIANS(s.Lat))\
        * COS(RADIANS(%s - s.Lon))\
        + SIN(RADIANS(%s))\
        * SIN(RADIANS(s.Lat))))  AS 'distance_in_km_start'\
                    FROM All_routes.new_all_routes s\
                    HAVING distance_in_km_start< %s) As Start\
                    JOIN\
        (SELECT DISTINCT e.Route as end_route, e.Direction as end_direction , e.Stop_ID as end_stop, e.Stop_sequence as end_stop_seq, e.Stop_name as end_stop_name, 111.111 * \
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
                                          "End_Stop_Sequence", "Distance_in_km_from_start", "Distance_in_km_from_end",
                                          "Start_Stop_Name", "End_Stop_Name"])
        dataframe['mid_point_lat'] = self.mid_point_lat
        dataframe['mid_point_lon'] = self.mid_point_lon
        return self.priority_options(dataframe)[0]

    def priority_options(self, dataframe):
        # this is here to select the nearest of the subset returned by sql to the user - minimises total user walking distance
        dataframe['low_score'] = dataframe["Distance_in_km_from_start"] + dataframe["Distance_in_km_from_end"]
        dataframe = dataframe.loc[dataframe.groupby('Route').low_score.idxmin()]
        # return self.dataframe_to_dict(dataframe)
        self.start_stop_seq = dataframe.Start_Stop_Sequence
        self.end_stop_seq = dataframe.End_Stop_Sequence
        self.start_route = dataframe.Route
        self.direction = dataframe.Direction

        return (dataframe, self.extract_lat_lon_stops())

    # --------------------------------------------------------------------------#

    def extract_lat_lon_stops(self):
        '''Returns lat lon of all stops on route'''
        co_ords = []
        engine = get_db()

        sql = 'SELECT lat, lon FROM All_routes.new_all_routes\
        WHERE Stop_sequence >= %s AND Stop_sequence<= %s AND Route = %s AND Direction = %s;'

        # result = engine.execute(sql, (self.start_stop_seq, self.end_stop_seq, self.start_route, self.direction))
        result = engine.execute(sql, (
        int(self.start_stop_seq.values[0]), int(self.end_stop_seq.values[0]), int(self.start_route.values[0]),
        int(self.direction.values[0])))
        all_co_ord_data = result.fetchall()

        for row in all_co_ord_data:
            co_ords.append({"lat": row[0], "lng": row[1]})
        print ('co_ords', co_ords)

        return (co_ords)

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

        return (data)


    #--------------------------------------------------------------------------#
    def fares(self):


        with urllib.request.urlopen("http://www.dublinbus.ie/en/Fare-Calculator/Fare-Calculator-Results/?routeNumber=46a&direction=O&board=31&alight=38") as response:
            data = response.read()
            soup = BeautifulSoup(data, 'html.parser')
            r = soup.find('span', {'id': 'ctl00_FullRegion_MainRegion_ContentColumns_holder_FareListingControl_lblFare'})
        return (r.text)

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

        sql = "SELECT MAX(Stop_sequence) FROM All_routes.new_all_routes WHERE Route = %s AND Direction = %s;"

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


def dataframe_to_dict(dataframe):
    # this converts the interesting routes to a dictionary
    route_options = dataframe.transpose().to_dict()
    i = 1
    break_at = len(route_options)
    for option in route_options:
        if i > break_at:
            break
        route_options["Option " + str(i)] = route_options.pop(option)
        i += 1
    return route_options


def everything(src, dest, time):
    """Determines the journey time for each viable route option
    Returns a list of options ordered by their journey time"""

    weekday = time.weekday()
    hour = time.hour
    min = time.minute
    bit1 = "1" if (min > 15) else "0"
    bit2 = "1" if (min > 30) else "0"
    bit3 = "1" if (min > 45) else "0"
    time_bin = str(hour) + bit1 + bit2 + bit3

    date = time.date()
    if date in dbi().extract_holidays():
        is_school_holiday = 1
    else:
        is_school_holiday = 0

    weather = dbi().scrape_weather()
    current_temp, current_wind = weather

    route_options = dbi().find_nearby_stops(src, dest)
    darts = dbi().find_darts(src, dest)
    route_options.Direction = route_options.Direction.astype(int)
    route_options['Stops_To_Travel'] = route_options.End_Stop_Sequence - route_options.Start_Stop_Sequence
    route_options['temp'] = current_temp
    route_options['wind'] = current_wind
    route_options['day'] = weekday
    route_options['holiday'] = is_school_holiday
    route_options['time_bin'] = time_bin
    route_options['time'] = time

    def sched_speed(dataframe):
        direction = int(list(dataframe.Direction.unique())[0])
        route = list(dataframe.Route.unique())[0]
        dataframe['max_stop_sequence'] = dbi().get_max_sequence(route, direction)
        dataframe['scheduled_time'] = dbi().get_sched_time(route, direction)
        dataframe['sched_speed'] = dataframe.scheduled_time / dataframe.max_stop_sequence
        dataframe['walking_mins'] =int(dataframe.low_score / 0.05)
        dataframe.drop(['max_stop_sequence', 'scheduled_time'], axis=1)
        return dataframe

    route_options = route_options.groupby(['Route', 'Direction']).apply(sched_speed)

    def everything_else(df):

        def extra_time_bins(time):
            times = []
            time_bins = []
            times_for_chart = []
            for i in range(8):
                times.append(time + datetime.timedelta(minutes=15 * i))

            for time in times:
                # makes time pretty
                times_for_chart.append(str(time.hour) + ":" + str(time.minute))
                # makes time bin for time
                bit1 = "1" if (time.minute > 15) else "0"
                bit2 = "1" if (time.minute > 30) else "0"
                bit3 = "1" if (time.minute > 45) else "0"
                '''str .join this later'''
                time_bins.append(str(time.hour) + bit1 + bit2 + bit3)
            return (time_bins, times_for_chart)

        time_bins, pretty_times = extra_time_bins(time)
        df = pd.concat([df] * 8, axis=0, ignore_index=True)

        df['time_bin'] = time_bins
        df['pretty_times'] = pretty_times

        return df

    route_options = route_options.groupby(['Route', 'Direction']).apply(everything_else)

    def make_predictions(df):
        try:
            route = list(df.Route.unique())[0]
            predictor = joblib.load('static/pkls/' + str(route) + 'rf.pkl')
            columns = ['day', 'time_bin', 'wind', 'temp', 'holiday', 'sched_speed', 'Stops_To_Travel',
                       'Start_Stop_Sequence']
            df['Predictions'] = predictor.predict(df[columns])
            '''add stop name to df,'''
            pred = str(df.Predictions.values[0])

            # df['html'] = "<div data-toggle='collapse' data-target='#map'><div class='option_route' onclick='boxclick(this, 1)'>" + route + "</div><div class='option_src_dest'>" +str(df.Start_Stop_Name) + " to " + str(df.End_Stop_Name) + "</div><div class='option_journey_time'>" + pred + "</div></div>"
            return df
        except:
            pass
    route_options = route_options.groupby(['Route', 'Direction']).apply(make_predictions)

    return dataframe_to_dict(route_options)
