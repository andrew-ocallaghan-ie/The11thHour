import os
import glob

import time
import multiprocessing
from multiprocessing import Pool
import threading

import pandas as pd
import numpy as np








#===================================Section1==========================================#
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
    time_frames = set(dataframe[ dataframe["Journey_Pattern_ID"] == "null" ].Timeframe.unique())
    total1=len(time_frames)
    
    for day in time_frames:
        vjid = list( dataframe[ (dataframe   [ "Timeframe" ]      ==  day  ) &\
                                (dataframe ["Journey_Pattern_ID"] == "null") ]\
                                .Vehicle_Journey_ID.unique())                     
        
    return fix_JPID(vjid, day, dataframe)




def drop_null_JPIDS(dataframe):

    return dataframe[dataframe["Journey_Pattern_ID"] != "null" ]





def fix_LID_and_Dir(dataframe):
    modified_frame = dataframe
    modified_frame["Journey_Pattern_ID"] = dataframe["Journey_Pattern_ID"].astype("str")
    modified_frame["LineID"] =  dataframe["Journey_Pattern_ID"].str[:4]
    modified_frame["LineID"] = dataframe["LineID"].str.lstrip("0")
    modified_frame["Direction"] = dataframe["Journey_Pattern_ID"].str[4]
    
    return modified_frame


#=================================================================================================#

def drop_columns(dataframe):
    
    return dataframe.drop(["Congestion",
                           "Delay",
                           "Block_ID",
                           "Operator"],
                           axis=1)


    


def drop_it_like_its_stop(dataframe):
    modified_frame = dataframe.loc[(dataframe.At_Stop == 1)]

    return modified_frame


#==================================================================================================#

def remove_idle_at_stop(dataframe):
    
    return dataframe.drop_duplicates(subset='Stop_ID', keep='first')
   


    
def stops_made(dataframe):
    for i in range(dataframe['Timestamp'].size):
        dataframe['Stops_Made'].values[i] = i+1
    
    return dataframe




def make_stops_made(dataframe):
    modified_frame = dataframe
    modified_frame["Stops_Made"] = 0
    modified_frame = modified_frame.groupby(['Vehicle_Journey_ID', 'Timeframe'])
    modified_frame = modified_frame.apply(remove_idle_at_stop)
    modified_frame = modified_frame.groupby(['Vehicle_Journey_ID', 'Timeframe'])
    modified_frame = modified_frame.apply(stops_made)
    
    return modified_frame    




def all_things_time(dataframe):
    dataframe["Time"] = pd.to_datetime(dataframe["Timestamp"], unit="us")
    dataframe["Day_Of_Week"] = dataframe["Time"].dt.dayofweek
    dataframe["Hour_Of_Day"] = dataframe["Time"].dt.hour
    dataframe["Min_Of_Hour"] = dataframe["Time"].dt.minute
    
    dataframe["End_Time"] = dataframe.groupby(["Vehicle_Journey_ID",
                                               "Timeframe"])["Timestamp"].transform(max)
    
    dataframe["Start_Time"] = dataframe.groupby(["Vehicle_Journey_ID",
                                               "Timeframe"])["Timestamp"].transform(min)
    
    dataframe["Late"] = ( (dataframe["End_Time"]) - (dataframe["Start_Time"]) )
    
    dataframe["Journey_Time"] = pd.to_timedelta(dataframe["Late"], unit="us").astype("timedelta64[m]")
    
    dataframe["Scheduled_Time_OP"] = 0
    
    dataframe.LineID = dataframe.LineID.astype("str")
    
    times_dataframe = pd.read_csv("route_times.csv",
                                  encoding = "utf-8",
                                  header = 0,
                                  index_col = None,
                                  converters = {"Route":str,  
                                                "Notes": str})
    
    time_dict = dict(zip(times_dataframe.Route, times_dataframe.Scheduled_Time_OP))
    
    #heres the problem 
    for Route, Scheduled_Time_OP in time_dict.items():
            #print(Route,"\n" ,dataframe.loc[ (dataframe.LineID == str(Route)) ])
            dataframe.loc[ (dataframe.LineID == str(Route)), "Scheduled_Time_OP" ] =  Scheduled_Time_OP
            dataframe["Mins_Late"] = ( (dataframe["Journey_Time"].astype(int)) - dataframe["Scheduled_Time_OP"] )
            
    dataframe["Late"] = np.where( (dataframe["Mins_Late"]) > 1, 1, 0 )
    
    return dataframe




def make_speed(dataframe):
    
    #a variable used to determine if the bus can use Time Travel to actually arrive on time!
    dataframe["Time_Traveling"] = ( (dataframe["Timestamp"]) - (dataframe["Start_Time"]) )
    dataframe["Time_Traveling"] = ( pd.to_timedelta(dataframe["Time_Traveling"], unit="us").astype("timedelta64[m]"))
    dataframe.LineID = dataframe.LineID.astype("str")
    dataframe.Stop_ID = dataframe.Stop_ID.astype("str")
 
    sequence_dataframe = pd.read_csv("route_seq.csv",
                                     encoding = "latin1",
                                     header = 0,
                                     index_col = None,
                                     converters = {"LineID":str,
                                                   "Stop_ID":str,
                                                   "Stop_Sequence": str})
    
    sequence_dataframe = sequence_dataframe[["LineID",
                                             "Stop_ID",
                                             "Stop_Sequence"]]

    dataframe = pd.merge(dataframe,
                         sequence_dataframe,
                         how='inner',
                         on=['LineID', 'Stop_ID'])
    
    dataframe["Speed"] = ( (dataframe["Time_Traveling"].astype(int) / (dataframe["Stop_Sequence"].astype(int)) ) ) 
    
    return dataframe


def binning(dataframe):
    bins=[10,20,30,40,50,60,70,80,90,100,110]
    dataframe['time_bins'] = np.digitize(dataframe.Journey_Time.values,
                                         bins=bins)
    
    return dataframe

#=====================================================================================================#

def delete_excess_columns(dataframe):
    return dataframe.drop(["End_Time",
                           "Start_Time",
                           "Scheduled_Time_OP",
                           "Lat",
                           "Lon"])



def drop_zero_JTs(dataframe):
    return dataframe[dataframe["Journey_Time"]>0]

def drop_JT_margin(dataframe, margin):
    return datafram[dataframe.Journey_Time > (dataframe.Scheduled_Time_OP)*margin]

def drop_zero_OP_Schedules(dataframe):
    return datafram[dataframe.Schedule_Time_OP == 0 ]



def drop_middle(dataframe):
    concat = pd.concat
    stops = pd.read_csv("stops.csv",
                        encoding="utf-8", 
                        converters={"First_1":str, 
                                    "First_2":str, 
                                    "First_3":str, 
                                    "Last_1":str, 
                                    "Last_2":str, 
                                    "Last_3":str})
    
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


#=============================================================================================#

def re_construct(path, files, month, columns):
    read = pd.read_csv
    fix_jpid = wrap_JPID
    fix_lidir = fix_LID_and_Dir
    
    for file in files:
        print("Reconstructing", file)
        
        read_address = os.path.join(path, file)
        modify_me = read(read_address, 
                         index_col=None, 
                         header=0, 
                         encoding="utf-8", 
                         converters = {"Journey_Pattern_ID":str,
                                       "LineID":str,
                                       "Direction":str,
                                       "Stop_ID":str })
        
        modify_me.columns = columns
        modify_me = drop_columns(modify_me)
        print("\tDropped Columns")
        
        modify_me = drop_it_like_its_stop(modify_me)
        print("\tDropped Not Stop data")
        
        #modify_me = fix_jpid(modify_me)
        modify_me = drop_null_JPIDS(modify_me)
        #print("\tFixed JPIDs")
              
        modify_me = fix_lidir(modify_me)
        print("\tDirection & LineID Re-Derived")
    
        
        modify_me = make_stops_made(modify_me)
        print("\tIdling removed and stops_made column created")
        
        modify_me = modify_me.drop_duplicates(keep="first")
        print("\tDropped Dupes")
        
        all_things_time(modify_me)
        print("\tTiming Columns added")
        
        modify_me = make_speed(modify_me)
        print("\tSpeed made")
        
        modify_me = binning(modify_me)
        print("\tTime Binning Applied")
        
        modify_me = drop_zero_JTs(modify_me)
        print("\tDrop zero Journey Times")
        
        #modify_me = drop_JT_margin(modify_me, 0.1)
        #print("\tDropped Improbable Journey_Times")
              
        #modify_me = drop_zero_OP_Schedules(modify_me)
        #print("\tDropped Empty Schedule Info")
        
                          
        #modify_me = delete_excess_columns(modify_me)
        #print("Deleted Intermediate Columns") 
                          
        
        #modify_me = drop_middle(modify_me)
        #print("\tDropped Middle of each journey")
        
        
        
        write_address = os.path.join( "re_constructed_" + month, "re_con" + file )
        modify_me.to_csv( write_address, encoding = "utf-8", index=False)
        print(write_address, "\tSaved!")
        
        

        
def wrap_re_construct(month):

    path = os.path.join(os.getcwd(), "re_constructed_" + month)
    os.makedirs(path, exist_ok=True)
    
    path = os.path.join(os.getcwd(), month + "DayRawCopy")
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
        
    thread1 = threading.Thread(target = re_construct, 
                               kwargs=dict(path=path,                       
                                           files=iter1,
                                           month=month, 
                                           columns=columns))
    
    thread2 = threading.Thread(target = re_construct, 
                               kwargs=dict(path=path, 
                                           files=iter2, 
                                           month=month, 
                                           columns=columns))
        
    thread1.start()
    thread2.start()
        
    thread1.join()
    thread2.join()
    
    print("Multithreading Complete")
    return "SUCCESS!"
              


    
    

    
    
    
#===============================Section 2 ===========================#    
def extract_route(path, route):
    print("Extrating", route, "from:", path)
    
    read = pd.read_csv
    concat = pd.concat
    contents = os.listdir(path)
    content_length = len(contents)

    read_address = os.path.join(path, contents[0])
    accumulator = read(read_address,
                       index_col=None,
                       header=0,
                       encoding="utf-8")
     
    for i in range(content_length):
        
        next_read_address = os.path.join(path, contents[i])
        next_df = read(next_read_address,
                       index_col=None,
                       header=0, 
                       encoding = "utf-8")
        
        accumulator, next_df = accumulator.align(next_df, axis=1)
        accumulator = concat([accumulator[( accumulator["LineID"] == route)], \
                                 next_df [(   next_df["LineID"]   == route)]] \
                                 , axis=0)
        
    return accumulator
              

    
    
def find_routes(path):
    read = pd.read_csv
    contents = os.listdir(path)
    content_length = len(contents)
    
    read_address = os.path.join(path, contents[0])
    accumulator = read(read_address,
                       index_col=None,
                       header=0,
                       encoding="utf-8",
                       converters = {"LineID":str})

    set_of_routes = set(accumulator.LineID.unique())
    unite = set_of_routes.union
    
    for i in range(1,content_length):
        
        next_read_address = os.path.join(path, contents[i])
        next_df = read(next_read_address,
                       index_col=None,
                       header=0,
                       encoding ="utf-8",
                       converters = {"LineID":str})
        
        next_set_of_routes = set(next_df.LineID.unique())
        set_of_routes = unite(next_set_of_routes)
    
    numset = {str(x) for x in set_of_routes if ( isinstance(x, int) or isinstance(x, float) ) }
    strset = {x for x in set_of_routes if isinstance(x, str)}
    set_of_routes = sorted(numset.union(strset), key=str)

    return set_of_routes




def complete_extraction(month):
    print("Starting Extraction : ", month)
    
    path = os.path.join(os.getcwd(), month + "_routes")
    os.makedirs(path, exist_ok=True)
    
    path = os.path.join(os.getcwd(), "re_constructed_" + month)
    all_routes = find_routes(path)
    
    for route in all_routes:
        
        dataframe = extract_route(path, route)
        write_address = os.path.join(month + "_routes", "beta" + route + ".csv")
        dataframe.to_csv(write_address)

    return "Success"




