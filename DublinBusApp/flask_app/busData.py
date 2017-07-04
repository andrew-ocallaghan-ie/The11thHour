from flask import jsonify
import csv

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
                'name': row[1]
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
                    'id': row[1],
                    'name': row[2],
                    'lat': row[3],
                    'lon': row[4],
                    'order': row[5],
                    'other_routes': row[7]
                }
                stops.append(stop)

        # Returning info
        return jsonify({'stops': stops})