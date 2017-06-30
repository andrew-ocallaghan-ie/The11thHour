from flask import jsonify
import csv

class luas:
    '''Class that deals with querying DB.'''

    def luas_info(self):
        stop_info = open('static/luas.csv', 'r', encoding='UTF-8', errors='ignore')
        reader = csv.reader(stop_info)
        stops = []
        headings = next(reader)
        for row in reader:

            stop = {
                'name': row[0],
                'latitude': float(row[1]),
                'longitude': float(row[2]),
            }
            stops.append(stop)

        # Returning info
        return jsonify({'stops': stops})

    # --------------------------------------------------------------------------#