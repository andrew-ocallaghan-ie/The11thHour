from flask import jsonify
import csv

class dart:
    '''Class that deals with querying DB.'''

    def __init__(self):
        pass

    def dart_stop_info(self):
        stop_info = open('static/dart.csv', 'r')
        reader = csv.reader(stop_info)
        stops = []

        headings = next(reader)
        for row in reader:

            stop = {
                'stop_id': row[5],
                'name': row[0],
                'latitude': float(row[2]),
                'longitude': float(row[3]),
            }
            stops.append(stop)

        # Returning info
        return jsonify({'stops': stops})

    # --------------------------------------------------------------------------#


