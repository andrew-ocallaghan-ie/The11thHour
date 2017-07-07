from flask import jsonify
import csv
from operator import itemgetter

class BusDB:
    '''Class that deals with querying DB.'''

    def __init__(self):
        pass

    def bus_route_info(self):
        stop_info = open('static/all_route_names.csv', 'r')
        reader = csv.reader(stop_info)
        routes = []

        headings = next(reader)

        for row in reader:

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
        route_info = open('static/' + direction + '_stops.csv', 'r', encoding = "ISO-8859-1")
        reader = csv.reader(route_info)
        stops = []

        headings = next(reader)

        for row in reader:

            if row[0] == routenum:
                stop = {
                    'id': int(row[1]),
                    'name': row[2],
                    'lat': row[3],
                    'lon': row[4],
                    'order': int(row[5]),
                    'other_routes': row[7]
                }
                stops.append(stop)

        stops = sorted(stops, key=itemgetter('order'))

        # Returning info
        return jsonify({'stops': stops})

        # --------------------------------------------------------------------------#


    def all_bus_stop_info(self):
        all_stops = open('static/2012_stop_info.csv', 'r', encoding = "ISO-8859-1")
        reader = csv.reader(all_stops)
        stops = []

        headings = next(reader)

        for row in reader:

            stop = {
                'id': int(row[0]),
                'name': row[1],
                'lat': row[2],
                'lon': row[3],
                'routes': row[4]
            }
            stops.append(stop)

        stops = sorted(stops, key=itemgetter('id'))

        # Returning info
        return jsonify({'stops': stops})