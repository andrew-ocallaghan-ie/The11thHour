import os
import glob

import time
import multiprocessing
from multiprocessing import Pool
import threading

import pandas as pd
import numpy as np



def fix_JPID(vjids, day, dataframe):
    loc = dataframe.loc
    total = len(vjids)
    current = 0
    for run in vjids:
        current+=100
        print(current//total,"% Reconstructed for ", day)
        vids = set(dataframe[ (dataframe    [ "Timeframe" ]     ==  day   ) &\
                          (dataframe ["Journey_Pattern_ID"] == "null" ) &\
                          (dataframe ["Vehicle_Journey_ID"] ==  run   ) ]\
                          .Vehicle_ID.unique())

        for vehicle in vids:
            re_construct = list(loc[ (dataframe    ["Timeframe"]       ==   day )  &\
                                     (dataframe ["Vehicle_Journey_ID"] ==   run )  &\
                                     (dataframe    ["Vehicle_ID"]      == vehicle ),\
                                     "Journey_Pattern_ID" ].unique() )  

            if len(re_construct) == 2:
                if re_construct[0] != "null":
                    loc[ (dataframe    ["Timeframe"]       ==  day    ) &\
                         (dataframe ["Vehicle_Journey_ID"] ==  run    ) &\
                         (dataframe    ["Vehicle_ID"]      == vehicle ), \
                         "Journey_Pattern_ID" ] = re_construct[0]

                else:
                    loc[ (dataframe    ["Timeframe"]        == day     ) &\
                         (dataframe ["Vehicle_Journey_ID"]  == run     ) &\
                         (dataframe    ["Vehicle_ID"]       == vehicle ), \
                         "Journey_Pattern_ID" ] = re_construct[1]
                    
    return dataframe
                


    
def wrap_JPID(dataframe):
    #every day
    time_frames = set(dataframe[ dataframe["Journey_Pattern_ID"] == "null" ].Timeframe.unique())
    total1=len(time_frames)
    
    for day in time_frames:
        vjid = list( dataframe[ (dataframe   [ "Timeframe" ]      ==  day  ) &\
                                (dataframe ["Journey_Pattern_ID"] == "null") ]\
                                .Vehicle_Journey_ID.unique())                     
        
    return fix_JPID(vjid, day, dataframe)




def fix_LID_and_Dir(dataframe):
    modified_frame = dataframe
    modified_frame["Journey_Pattern_ID"] = dataframe["Journey_Pattern_ID"].astype("str")
    modified_frame["LineID"] =  dataframe["Journey_Pattern_ID"].str[:4]
    modified_frame["Direction"] = dataframe["Journey_Pattern_ID"].str[4]
    
    return modified_frame




def drop_columns(dataframe):
    modified_frame = dataframe
    modified_frame = modified_frame.drop("Congestion", axis=1)
    modified_frame = modified_frame.drop("Delay", axis=1)
    modified_frame = modified_frame.drop("Block_ID", axis=1)
    modified_frame = modified_frame.drop("Operator", axis=1)
    
    return modified_frame




def drop_it_like_its_stop(dataframe):
    modified_frame = dataframe.loc[(dataframe.At_Stop == 1)]

    return modified_frame




def drop_null_JPIDS(dataframe):

    return dataframe[dataframe["Journey_Pattern_ID"] != "null" ]




def remove_idle_at_stop(df):
    df = df.drop_duplicates(subset='Stop_ID', keep='first')
    
    return df



def stops_made(df):
    for i in range(df['Timestamp'].size):
        df['Stops_Made'].values[i] = i+1
    
    return df



def make_stops_made(dataframe):
    modified_frame = dataframe
    modified_frame["Stops_Made"] = 0
    modified_frame = modified_frame.groupby(['Vehicle_Journey_ID', 'Timeframe'])
    modified_frame = modified_frame.apply(remove_idle_at_stop)
    modified_frame = modified_frame.groupby(['Vehicle_Journey_ID', 'Timeframe'])
    modified_frame = modified_frame.apply(stops_made)
    
    return modified_frame    




def drop_middle(dataframe):
    concat = pd.concat
    stops = pd.read_csv("stops.csv", encoding="utf-8", converters={"First_1":str, "First_2":str, "First_3":str, "Last_1":str, "Last_2":str, "Last_3":str})
    col1 = stops.First_1.unique()
    col2 = stops.First_2.unique()
    col3 = stops.First_3.unique()
    col4 = stops.Last_1.unique()
    col5 = stops.Last_2.unique()
    col6 = stops.Last_3.unique()
    mysets = [col1, col2, col3, col4, col5, col6]
    arbitrary_element = col1[0]
    
    full_set = set(frozenset().union(*mysets))
    
    
    numset = {str(x) for x in full_set if ( isinstance(x, int) or isinstance(x, float) ) }
    strset = {x for x in full_set if isinstance(x, str)}
    set_of_routes = sorted(numset.union(strset), key=str)
    
    modified_frame = dataframe
    refined_frame  = dataframe.loc[(dataframe.Stop_ID == arbitrary_element)]
    for element in full_set:
        next_frame = dataframe.loc[(dataframe.Stop_ID == element)]
        
        refined_frame, next_frame = refined_frame.align(next_frame, axis=1)
        
        refined_frame = concat([refined_frame, next_frame], axis = 0)
    
    return refined_frame




def re_construct(path, files, month, columns):
    read = pd.read_csv
    fix_jpid = wrap_JPID
    fix_lidir = fix_LID_and_Dir
    
    for file in files:
        print("Reconstructing", file)
        modify_me = read(path+"\\"+file, index_col=None, header=0, encoding="utf-8", converters = {"Journey_Pattern_ID":str, "LineID":str, "Direction":str, "Stop_ID":str })
        
        modify_me.columns = columns
        modify_me = drop_columns(modify_me)
        print("\tDropped Columns")
        
        modify_me = drop_it_like_its_stop(modify_me)
        print("\tDropped Not Stop data")
        
#         modify_me = fix_jpid(modify_me)
        modify_me = drop_null_JPIDS(modify_me)
#         print("\tFixed JPIDs")
              
        modify_me = fix_lidir(modify_me)
        print("\tDirection & LineID Re-Derived")
    
        
        modify_me = make_stops_made(modify_me)
        print("\t idling removed and stops_made column created")
        
        modify_me = modify_me.drop_duplicates(keep="first")
        print("\tDropped Dupes")
        
        modify_me = drop_middle(modify_me)
        print("Dropped Middle of each journey")
        
        modify_me.to_csv("re_constructed_"+month+"\\re_con_"+file, encoding = "utf-8", index=False)
        print("\tSaved!")
        
        

        
def wrap_re_construct(month):
    path = os.getcwd()
    path = path +"\\"+month+"DayRawCopy"
    contents = os.listdir(path)
    
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
    
    total2 = len(contents)
    iter1 = set(contents[:total2//2])
    iter2 = set(contents[total2//2:])
        
    thread1 = threading.Thread(target = re_construct, kwargs=dict(path=path, files=iter1, month=month, columns=columns))
    thread2 = threading.Thread(target = re_construct, kwargs=dict(path=path, files=iter2, month=month, columns=columns))
        
    thread1.start()
    thread2.start()
        
    thread1.join()
    thread2.join()
    
    print("Multithreading Complete")
    return "SUCCESS!"
              


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def extract_route(path, route):
    read = pd.read_csv
    concat = pd.concat
    contents = os.listdir(path)
    content_length = len(contents)
   
    accumulator = read(path+"\\"+contents[0], index_col=None, header=0, encoding="utf-8")
    
    for i in range(content_length):
        print("extracting", route, "from file", i)
        next_df = read(path+"\\"+contents[i], index_col=None, header=0)
        accumulator, next_df = accumulator.align(next_df, axis=1)
        accumulator = concat([accumulator[( accumulator["LineID"] == route)], \
                                 next_df [(   next_df["LineID"]   == route)]] \
                                 , axis=0)
        
    return accumulator
              

    
    
def find_routes(path):
    read = pd.read_csv
    contents = os.listdir(path)
    content_length = len(contents)
    
    accumulator = read(path+"\\"+contents[0], index_col=None, header=0, encoding="utf-8")
    
    set_of_routes = set(accumulator.LineID.unique())
    unite = set_of_routes.union
    
    for i in range(1,content_length):
        next_df = read(path+"\\"+contents[i], index_col=None, header=0, encoding ="utf-8")
        next_set_of_routes = set(next_df.LineID.unique())
        set_of_routes = unite(next_set_of_routes)
        print("Constructing Route Set... Current File:", i)
    
    numset = {str(x) for x in set_of_routes if ( isinstance(x, int) or isinstance(x, float) ) }
    strset = {x for x in set_of_routes if isinstance(x, str)}
    set_of_routes = sorted(numset.union(strset), key=str)
    
    return set_of_routes




def complete_extraction(month):
    path = os.getcwd() + "\\re_constructed_" + month
    all_routes = find_routes(path)
    
    for route in all_routes:
        dataframe = extract_route(path, route)
        dataframe.to_csv(month+"_routes\\alpha-"+route+".csv")

    return "Success"