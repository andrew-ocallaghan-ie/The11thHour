from flask import jsonify
from operator import itemgetter
from sqlalchemy import create_engine
from flask import g
import pandas as pd
import datetime
import requests
import traceback
import csv

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

        # --------------------------------------------------------------------------#

    # def get_common_routes(self, src_stop_num, dest_stop_num):
    #     """Finds common routes between two bus stops
    #     Returns a dictionary of routes, easier to loop through"""
    #     route_options = {}
    #
    #     engine = get_db()
    #     sql = "SELECT * \
    #              FROM All_routes.Sequence \
    #              WHERE Stop_ID = %s AND Route IN (\
    #                  SELECT Route \
    #                  FROM All_routes.Sequence \
    #                  WHERE Stop_ID = %s);"
    #
    #     result = engine.execute(sql, (src_stop_num, dest_stop_num))
    #     all_data = result.fetchall()
    #
    #     for row in all_data:
    #         route_options[row[0].strip('\n')] = (row[1])
    #
    #     return route_options

        # ---------------------------------------------------------------------------#

    def stops_between_src_dest(self, src_stop_num, dest_stop_num, route):
        """Finds out how many stops are between two stops on a given route"""
        engine = get_db()
        sql = "SELECT Stop_sequence \
                FROM All_routes.Sequence \
                WHERE (Stop_ID = %s AND Route = %s) OR \
                      (Stop_ID = %s AND Route = %s);"

        result = engine.execute(sql, (src_stop_num,
                                      int(route),
                                      dest_stop_num,
                                      int(route)))
        all_data = result.fetchall()

        src_stop_sequence = int(all_data[0][0])
        dest_stop_sequence = int(all_data[1][0])

        stops_travelled = dest_stop_sequence - src_stop_sequence

        return stops_travelled

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
        """
        src_lat, src_lon = self.location_from_address(src)
        dest_lat, dest_lon = self.location_from_address(dest)
        engine = get_db()
        radius = 0.3

        """use 6371 as constant and drop degree conv."""
        sql = "SELECT  start_route, start_direction, start_stop, end_stop, start_stop_seq, end_stop_seq\
        FROM(SELECT DISTINCT s.Route as start_route, s.Direction as start_direction, s.Stop_ID as start_stop, s.Stop_sequence as start_stop_seq, 111.111 *\
                DEGREES(ACOS(COS(RADIANS(%s))\
        * COS(RADIANS(s.Lat))\
        * COS(RADIANS(%s - s.Lon))\
        + SIN(RADIANS(%s))\
        * SIN(RADIANS(s.Lat))))  AS 'distance_in_km_start' \
                    FROM All_routes.new_all_routes s\
                    HAVING distance_in_km_start< %s) As Start\
                    JOIN \
        (SELECT DISTINCT e.Route as end_route, e.Direction as end_direction , e.Stop_ID as end_stop, e.Stop_sequence as end_stop_seq, 111.111 * \
        DEGREES(ACOS(COS(RADIANS(%s))\
        * COS(RADIANS(e.Lat))\
        * COS(RADIANS(%s - e.Lon))\
        + SIN(RADIANS(%s))\
        * SIN(RADIANS(e.Lat))))  AS 'distance_in_km_end' \
                    FROM All_routes.new_all_routes e\
                    HAVING distance_in_km_end< %s) as end\
        WHERE start_route=end_route AND start_stop_seq<end_stop_seq AND start_direction=end_direction"

        result = engine.execute(sql, (src_lat, src_lon, src_lat, radius, dest_lat, dest_lon, dest_lat, radius))
        all_data = result.fetchall()

        dataframe = pd.DataFrame(all_data, columns=["Route", "Direction", "Start_Stop_ID", "End_Stop_ID", "Start_Stop_Sequence", "End_Stop_Sequence"])
        print (dataframe)
        return dataframe

    # def route_overlap(self, stop_ids):
    #     """gets overlap of routes between two stops"""
    #     engine = get_db()
    #
    #     sql = "SELECT Stop_ID, Routes_serviced \
    #            FROM All_routes.Stops\
    #            WHERE Stop_ID in (%s)" % ",".join(map(str, stop_ids))
    #
    #     stop_route = {}
    #     result = engine.execute(sql)
    #     all_data = result.fetchall()
    #
    #     for row in all_data:
    #         stop_route[row[0]] = set(map(str.strip, row[1].split(" - ")))
    #
    #     return stop_route

        # ---------------------------------------------------------------------#

    # def route_plan(self, routes, src_stops, dest_stops):
    #     """gets table of routes, dir, stop_id"""
    #     engine = get_db()
    #     all_stops = src_stops
    #     if src_stops != dest_stops:
    #         all_stops = src_stops.union(dest_stops)
    #
    #     # ',' is important for getting sql to read correctly '%s'
    #     sql = "SELECT *, Stop_ID in (%s) as 'src' \
    #            FROM All_routes.Sequence\
    #            WHERE \
    #                Route in ('%s') AND\
    #                Stop_ID in (%s)\
    #             ORDER BY Route, Direction, Stop_Sequence" % \
    #           (",".join(map(str, src_stops)),
    #            "','".join(map(str, routes)),
    #            ','.join(map(str, all_stops)))
    #
    #     stops = []
    #     result = engine.execute(sql)
    #     all_data = result.fetchall()
    #     dataframe = pd.DataFrame(all_data, columns=["Route", "Direction", "Stop_ID", "Stop_Sequence", "Src"])
    #
    #     return dataframe

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
