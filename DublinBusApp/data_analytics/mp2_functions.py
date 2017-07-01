import pandas as pd
import numpy as np

import sys
import os
import glob

import time
import multiprocessing
from multiprocessing import Pool
import threading

def extract_route(route, month):
    #each thread extracts half the route and return the concat of their accumulators
    
    path=os.getcwd() + "\\re_constructed_"+month
    
    #should help loop run a little faster.
    read = pd.read_csv
    concat = pd.concat
    
    #stores contents of a folder as a list.
    contents = os.listdir(path)
    
    #small performance improvement.
    content_length = len(contents)
    
    columns   =    ["Timestamp",
                    "LineID", 
                    "Direction",
                    "Journey_Pattern_ID", 
                    "Timeframe", 
                    "Vehicle_Journey_ID", 
                    "Operator", 
                    "Congestion", 
                    "Lon",
                    "Lat", 
                    "Delay", 
                    "Block_ID",
                    "Vehicle_ID",
                    "Stop_ID",
                    "At_Stop"]
    
    
    #reads csv from data folder in cwd    
    #This line may need to be changed to use "/" instead of "\\" to run on mac or linux.
    accumulator1 = read(path+"\\"+contents[0], index_col=None, header=0, encoding="utf-8", converters = {"Journey_Pattern_ID":str, "LineID":str, "Direction":str })
    
    
    #Sets columns names for dataframe (so it can operated easily and readably)
    accumulator.columns = columns
    
    
    for file in range(contents):
        print("extracting", route, "from file", file)
        #This line may need to be changed to use "/" instead of "\\" to run on mac or linux.
        next_df = pd.read_csv(path+"\\"+file, index_col=None, header=0, converters = {"Journey_Pattern_ID":str, "LineID":str, "Direction":str })
        next_df.columns = columns

        #Line Continuation char is used for readability
        accumulator = concat([accumulator[( accumulator["LineID"] == route)], \
                                 next_df [(   next_df["LineID"]   == route)]] \
                                 , axis=0)
        
        print(accumulator.shape, "acc") #(use this to track concats are happening correctly; debugging)
        
    
    return accumulator



def find_routes(month):
    
    path=os.getcwd() + "\\re_constructed_"+month
    
    #should help loop run a little faster.
    read = pd.read_csv
    
    #stores contents of a folder as a list.
    contents = os.listdir(path)
    
    #small performance improvement.
    content_length = len(contents)
    
    columns   =    ["Timestamp",
                    "LineID", 
                    "Direction",
                    "Journey_Pattern_ID", 
                    "Timeframe", 
                    "Vehicle_Journey_ID", 
                    "Operator", 
                    "Congestion", 
                    "Lon",
                    "Lat", 
                    "Delay", 
                    "Block_ID",
                    "Vehicle_ID",
                    "Stop_ID",
                    "At_Stop"]
    
    
    #reads csv from data folder in cwd    
    #This line may need to be changed to use "/" instead of "\\" to run on mac or linux.
    accumulator = read(path+"\\"+contents[0], index_col=None, header=0, encoding="utf-8", converters = {"Journey_Pattern_ID":str, "LineID":str, "Direction":str } )
    
    #Sets columns names for dataframe (so it can operated easily and readably)
    accumulator.columns = columns
    
    set_of_routes = set(accumulator.LineID.unique())
    
    print("Constructing Route Set... Current File:", 0)
    print(len(set_of_routes))
#     comparison_set = set_of_routes #this is used to test the necesity of looping through all files
    
    #Should help loop run a bit faster
    unite = set_of_routes.union
    
    #read through all files once and determine maximum set of routes
    for i in range(1,content_length):
        
        #if you change the folder name or need to make this run on mac or linux, this is the line you change.
        next_df = read(path+"\\"+contents[i], index_col=None, header=0, encoding ="utf-8", converters = {"Journey_Pattern_ID":str, "LineID":str, "Direction":str })
        next_df.columns = columns
        accumulator.LineID.astype("str")
        
        next_set_of_routes = set(next_df.LineID.unique())
        set_of_routes = set_of_routes.union(next_set_of_routes)
        print("Constructing Route Set... Current File:", i)
        print(len(set_of_routes))
    
    # This block of code can be used to show that extracting all routes from a single file does not work, its nexesary to go through each file exhaustively
#
#     print("total set of routes\n", set(set_of_routes)==set(comparison_set))
#     print(comparison_set - set_of_routes)
#     print(set_of_routes - comparison_set)
    
    floatset = {str(x) for x in set_of_routes if isinstance(x, float) or isinstance(x, int)}
    print("intset",len(floatset))
    strset = {x for x in set_of_routes if isinstance(x, str)}
    print("strset",len(strset))

#     set_of_routes = sorted(intset.union(strset), key=str)
    print("no dupes",len(set_of_routes))
    print(set_of_routes)
    print("removed literal duplicates and converted each set to string")
    
    return set_of_routes




#multiprocess for each month
def wrap_route_extract(routes, month):
    for route in routes:
        dataframe = extract_route(route, month)
        dataframe.to_csv(month+"_Routes\\alpha-"+route+".csv", encoding ="utf-8")


        
def complete_extraction(month):
    
    path = os.getcwd() + "\\re_constructed_"+month
    all_routes = list(find_routes(month))
    
    
    iter1 = set(all_routes[:len(all_routes)//2])
    iter2 = set(all_routes[len(all_routes)//2:])
    print("routes found")
    
    #Instantiate Threads
    thread1 = threading.Thread(target = wrap_route_extract, kwargs=dict(routes=iter1, month=month))
    thread2 = threading.Thread(target = wrap_route_extract, kwargs=dict(routes=iter2, month=month))
        
        
    #Start each in parallel
    thread1.start()
    thread2.start()
        
        
    #threads wait for eachother to complete then merge.
    thread1.join()
    thread2.join()
    print("Multi-Threading Complete")
                               

        
        
        
        
       
        


    

    
    
    
    







