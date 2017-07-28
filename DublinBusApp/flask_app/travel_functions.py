'''
Created on 20 Jul 2017

@author: Andy
'''


from busData import dbi

#https://docs.python.org/3/library/datetime.html
import datetime

#http://pymysql.readthedocs.io/en/latest/index.html
import pymysql

#http://www.pythonforbeginners.com/requests/using-requests-in-python
import requests

#http://pandas.pydata.org/
import pandas as pd

#http://scikit-learn.org/stable/
from sklearn.externals import joblib

pymysql.install_as_MySQLdb()

# --------------------------------------------------------------------------#
# def location_from_address(self, address):
#     """Gets latitude & longitude from an address Returns a tuple of lat/long
#     Currently only takes the first search result"""
#     address = address.replace(" ", "+")
#     key = "AIzaSyBVaetyYe44_Ay4Oi5Ljxu83jKLnMKEtBc"
#     url = "https://maps.googleapis.com/maps/api/geocode/json?"
#     params = {'address': address, 'region': 'IE', 'components': 'locality:dublin|country:IE', 'key': key}
#
#     r = requests.get(url, params=params)
#     data = r.json()
#
#     lat = data['results'][0]['geometry']['location']['lat']
#     long = data['results'][0]['geometry']['location']['lng']
#
#     location = (lat, long)
#     return location

#----------------------------------------------------------------------#

# def possible_routes(area):
#     """gets stop and route info of area
#     returns dict with key for each stop, and 'all_routes'"""
#     lat, lon = location_from_address(area)
#     possible_stops = dbi().find_nearby_stops(lat, lon)
#
#     stop_route = dbi().route_overlap(possible_stops)
#     all_routes = set()
#
#     for stop, routes in stop_route.items():
#         if not all_routes == routes:
#             all_routes = all_routes ^ routes
#
#     stop_route["all_routes"] = all_routes
#     return stop_route

#-----------------------------------------------------------------------#
# """TODO: The following TWO functions can be dropped in place of a 1 function making a good sql call"""
# def find_viable_routes(source_area, dest_area):
#     """performs manipulation on info to get viable routes"""
#
#     info = {"source_info":possible_routes(source_area),
#             "dest_info":possible_routes(dest_area)}
#
#     source_info = info["source_info"]
#     dest_info = info["dest_info"]
#     info["viable_routes"] = source_info["all_routes"].intersection(dest_info["all_routes"])
#
#     return info
    
# --------------------------------------------------------------------------#

# def find_viable_stops(info):
#     """refines src_dest info to stops with viable routes only"""
#     viable_routes = info["viable_routes"]
   
    # def popper(some_info):
    #     """pops excess stops"""
    #     stops_to_pop = []
    #     for stop, routes in some_info.items():
    #         if  routes.isdisjoint(viable_routes):
    #             stops_to_pop.append(stop)
    #         elif routes != viable_routes:
    #             some_info[stop] = routes.intersection(viable_routes)
    #         else:
    #             pass
    #     some_info = dict([(k,v) for k,v in some_info.items() if k not in stops_to_pop])
    #     return some_info
    #
    # info["source_info"] = popper( info["source_info"] )
    # info["dest_info"]  = popper(  info["dest_info"]  )
    #
    # return info
    
#-----------------------------------------------------------------------#

# def route_planner(info):
#     """returns table of viable source and dest stops"""
#     viable_routes = info["viable_routes"]
#
#     viable_src_stops = set(info["source_info"].keys())
#     viable_src_stops.discard("all_routes")
#
#     viable_dest_stops = set(info["dest_info"].keys())
#     viable_dest_stops.discard("all_routes")
#
#     dataframe = dbi().route_plan(viable_routes, viable_src_stops, viable_dest_stops)
#
#     return determine_travel_options(dataframe)

#-----------------------------------------------------------------------#


# def determine_travel_options(dataframe):
#     src_df = dataframe[dataframe.Src == 1]
#     src_idx = src_df.groupby(['Route', "Direction"])['Stop_Sequence'].transform(max) == src_df.Stop_Sequence
#
#     dest_df = dataframe[dataframe.Src == 0]
#     dest_idx = dest_df.groupby(['Route', "Direction"])['Stop_Sequence'].transform(min) == dest_df.Stop_Sequence
#
#     dataframe = pd.concat([src_df[src_idx], dest_df[dest_idx]], axis=0)\
#                         .sort_values(by=["Route",
#                                          "Direction",
#                                          "Stop_Sequence"])
#
    # def validate_direction(dataframe):
    #     """Rows in groups are marked "valid" if direction of travel
    #     is correct for user's journey"""
    #     value1= dataframe[dataframe.Src ==1].Stop_Sequence.values[0]
    #     value2= dataframe[dataframe.Src ==0].Stop_Sequence.values[0]
    #     dataframe.Valid = (value1 < value2)
    #     return dataframe
    
    
    # def valid_route_options(dataframe):
    #     """takes all valid rows, and modifies the frame
    #     such that each row represents a single route option"""
    #     src_df = dataframe[dataframe.Src == 1]
    #     src_df = src_df.rename(columns = {"Stop_Sequence":"Src_Stop_Sequence",
    #                                       "Stop_ID":"Src_Stop_ID"})
    #
    #     dest_df = dataframe[dataframe.Src == 0]
    #     dest_df = dest_df[["Route", "Direction", "Stop_ID", "Stop_Sequence"]]
    #     dest_df = dest_df.rename(columns={"Stop_Sequence":"Dest_Stop_Sequence",
    #                                       "Stop_ID":"Dest_Stop_ID"})
    #
    #     result = pd.merge(src_df, dest_df, on=["Route","Direction"], how="inner" )
    #     result = result.drop(["Valid", "Src"],axis=1)
    #
    #     return result
    #
    # dataframe["Valid"]= 123
    # dataframe = dataframe.groupby(["Route", "Direction"])
    # dataframe = dataframe.apply(validate_direction)
    # dataframe = dataframe[dataframe.Valid == True]
    # route_options_dataframe = valid_route_options(dataframe)
    #
    #
    # def determine_travel_options(self, dataframe):
    #     src_df = dataframe[dataframe.Src == 1]
    #     src_idx = src_df.groupby(['Route', "Direction"])['Stop_Sequence'].transform(max) == src_df.Stop_Sequence
    #
    #     dest_df = dataframe[dataframe.Src == 0]
    #     dest_idx = dest_df.groupby(['Route', "Direction"])['Stop_Sequence'].transform(min) == dest_df.Stop_Sequence
    #
    #     dataframe = pd.concat([src_df[src_idx], dest_df[dest_idx]], axis=0) \
    #         .sort_values(by=["Route",
    #                          "Direction",
    #                          "Stop_Sequence"])
    
#     def dataframe_to_dict(dataframe):
#         route_options = dataframe.transpose().to_dict()
#         i=1
#         break_at = len(route_options)
#         for option in route_options:
#             if i > break_at:
#                 break
#             route_options["Option "+str(i)] = route_options.pop(option)
#             i+=1
#         return route_options
#
#     return dataframe_to_dict(route_options_dataframe)
#
#
# # --------------------------------------------------------------------------#
#
# def get_option_times(route_options):
#     """Determines the journey time for each viable route option
#     Returns a list of options ordered by their journey time"""
#
#     options = []
#     timestamp = datetime.datetime.now()
#     current_weekday = timestamp.weekday()
# #     current_hour = timestamp.
#     for option, info in route_options.items():
#
#         route = info['Route']
#         direction = info['Direction']
# #         info["time_bin"] = timestamp
#         timestamp = info['Time_Bin']
#         src_stop = info['Src_Stop_ID']
#         dest_stop = info['Dest_Stop_ID']
#         stops_travelled = info["Src_Stop_Sequence"] - info["Dest_Stop_Sequence"]
# #         stops_travelled = dbi().stops_between_src_dest(src_stop, dest_stop, route)
# #         stops_traveled = difference between sourece and end stop in info
#
#
#         predictor = joblib.load('static/pkls/xbeta' + route + '.csvrf_regressor.pkl')
#
#         time_pred = predictor.predict([1, current_weekday, direction, timestamp, stops_travelled])
#
#         options.append([time_pred, route, src_stop, dest_stop])
#
#     options = options.sort(key=lambda x: x[0])
#     return options


# --------------------------------------------------------------------------#

