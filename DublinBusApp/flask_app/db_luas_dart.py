from flask import jsonify
import csv

class alldublinstation:
    '''Class that deals with querying DB.'''

    def __init__(self):
        pass

    def all_stop_info(self):
        stop_info = open('static/dublinbike_dart_luas.csv', 'r', errors='ignore')
        reader = csv.reader(stop_info)
        stops = []

        headings = next(reader)
        for row in reader:

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