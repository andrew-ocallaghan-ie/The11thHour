
from flask import jsonify
from operator import itemgetter
from sqlalchemy import create_engine
from flask import g

# --------------------------------------------------------------------------#

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


# --------------------------------------------------------------------------#

class api:
    '''Class that deals with querying DB for API construction.'''
    

    def __init__(self):
        pass

        #----------------------------------------------------------------------------------#

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
        
    def get_common_routes(self, src_stop_num, dest_stop_num):
        """Finds common routes between two bus stops
        Returns a dictionary of routes, easier to loop through"""
        route_options = {}
    
        engine = get_db()
        sql =   "SELECT * \
                 FROM All_routes.Sequence \
                 WHERE Stop_ID = %s AND Route IN (\
                     SELECT Route \
                     FROM All_routes.Sequence \
                     WHERE Stop_ID = %s);"
        
        result = engine.execute(sql, (src_stop_num, dest_stop_num))
        all_data = result.fetchall()
    
        for row in all_data:
            route_options[row[0].strip('\n')] = (row[1])
    
        return route_options

        # ---------------------------------------------------------------------------#
        
        
    def stops_between_src_dest(self, src_stop_num, dest_stop_num, route):
        """Finds out how many stops are between two stops on a given route"""
        engine = get_db()
        sql =   "SELECT Stop_sequence \
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

        #----------------------------------------------------------------------------------#


    def nearby_stops(self, lat,long):
        """
        Finds out the nearest stops to a given point
        Returns stop IDs in a list
        """
        stops = []
        engine = get_db()
        radius = 0.3
        
        """use 6371 as constant and drop degree conv."""
        sql =   "SELECT Stop_ID, 111.111 * \
                DEGREES( acos (\
                    cos ( radians(%s) ) * cos( radians(Lat) ) * cos( radians(Lon) - radians(%s) ) +\
                    sin ( radians(%s) ) * sin( radians( Lat ) ) ) ) \
                AS `distance_in_km` \
                FROM All_routes.Stops \
                HAVING (distance_in_km < %s) \
                ORDER BY distance_in_km;"
        result = engine.execute(sql, (lat, long, lat, radius))
        all_data = result.fetchall()
    
        for row in all_data:
            stops.append(row[0])
    
        return stops
    
    
