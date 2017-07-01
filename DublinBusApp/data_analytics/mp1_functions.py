#import sections something to cover all bases
import pandas as pd
import numpy as np

import sys
import os
import glob

import time
import multiprocessing
from multiprocessing import Pool
import threading


        
        
#could be optimised further: instead of allocating each thread and arrbitrary case load 
#(one might finish early and the other late, unlikely though)
#use a queue , so each thread takes the next in the queue, giving them an even workload.
           
def fix_JPID(vjids, day, dataframe):
    
    #should help loop run a little faster
    loc = dataframe.loc
    for run in vjids: #vehicle_journey_id
        #vehicle ids
        vids = set(dataframe[ (dataframe    [ "Timeframe" ]     ==  day   ) &\
                          (dataframe ["Journey_Pattern_ID"] == "null" ) &\
                          (dataframe ["Vehicle_Journey_ID"] ==  run   ) ]\
                          .Vehicle_ID.unique())

        for vehicle in vids:
            #list of potential journeys eliguble for re-construction
            re_construct = list(loc[ (dataframe    ["Timeframe"]       ==   day )  &\
                                     (dataframe ["Vehicle_Journey_ID"] ==   run )  &\
                                     (dataframe    ["Vehicle_ID"]      == vehicle ),\
                                     "Journey_Pattern_ID" ].unique() )  


            #re-constructs journey pattern to non-null entry
            if len(re_construct) == 2:
                if re_construct[0] != "null":
                    #replaces nulls
                    loc[ (dataframe    ["Timeframe"]       ==  day    ) &\
                         (dataframe ["Vehicle_Journey_ID"] ==  run    ) &\
                         (dataframe    ["Vehicle_ID"]      == vehicle ), \
                         "Journey_Pattern_ID" ] = re_construct[0]

                else:
                    #replaces nulls
                    loc[ (dataframe    ["Timeframe"]        == day     ) &\
                         (dataframe ["Vehicle_Journey_ID"]  == run     ) &\
                         (dataframe    ["Vehicle_ID"]       == vehicle ), \
                         "Journey_Pattern_ID" ] = re_construct[1]
                    
    return dataframe
                

#This wraps the fixing function to run it on seperate threads
def wrap_JPID(dataframe):
    #every day
    time_frames = set(dataframe[ dataframe["Journey_Pattern_ID"] == "null" ].Timeframe.unique())
    total1=len(time_frames)
    
#     for raw data this loop is redundant but negligable. its here for extensability of purpose.
    for day in time_frames:
        
    #find list of vehicle journey ids with nulls during that day
        vjid = list( dataframe[ (dataframe   [ "Timeframe" ]      ==  day  ) &\
                                (dataframe ["Journey_Pattern_ID"] == "null") ]\
                                .Vehicle_Journey_ID.unique())                     
        
     
    return fix_JPID(vjid, day, dataframe)



def fix_LID_and_Dir(dataframe):
    modified_frame = dataframe
    modified_frame["Journey_Pattern_ID"] = dataframe["Journey_Pattern_ID"].astype("str")
    modified_frame["LineID"] =  dataframe["Journey_Pattern_ID"].str[:4]
#     modified_frame[modified_frame["LineID"].str[:3] == "000"] =  dataframe["Journey_Pattern_ID"].str[3]
#     modified_frame[modified_frame["LineID"].str[:2] == "00"] =  dataframe["Journey_Pattern_ID"].str[2:4]
#     modified_frame[modified_frame["LineID"].str[:1] == "0"] =  dataframe["Journey_Pattern_ID"].str[1:4]
    modified_frame["Direction"] = dataframe["Journey_Pattern_ID"].str[4]
    
    return modified_frame

def drop_it_like_its_stop(dataframe):
    modified_frame = dataframe.loc[(dataframe.At_Stop == 1)]
    return modified_frame
   

def re_construct(path, files, month, columns):

    read = pd.read_csv
    fix_jpid = wrap_JPID
    fix_lidir = fix_LID_and_Dir
    
    for file in files:
        print("Reconstructing", file)
        #reads csv from data folder in cwd    
        #This line may need to be changed to use "/" instead of "\\" to run on mac or linux.
        modify_me = read(path+"\\"+file, index_col=None, header=0, encoding="utf-8")
        
        #Sets columns names for dataframe (so it can operated easily and readably)
        modify_me.columns = columns
        modify_me = drop_it_like_its_stop(modify_me)
        print("\tDropped Not Stop data")
        modify_me = modify_me.to_csv(path+"\\"+file, encoding="utf-8", index=False)
        modify_me = read(path+"\\"+file, encoding="utf-8",index_col=False, converters = {"Journey_Pattern_ID":str, "LineID":str, "Direction":str })
        modify_me = fix_jpid(modify_me)
        print("\tFixed JPIDs")
        modify_me = fix_lidir(modify_me)
        print("\tDirection & LineID Re-Derived")
        modify_me = modify_me.drop_duplicates(keep="first")
        print("\tDropped Dupes")
      
        modify_me.to_csv("re_constructed_"+month+"\\re_con_"+file, encoding = "utf-8", index=False)
        print("\tSaved!")
        
        
def wrap_re_construct(month):
    path = os.getcwd()
    path = path +"\\"+month+"DayRawCopy"
    
    #stores contents of a folder as a list.
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
    
      
        #Define load for each thread
    #todo: use queue to improve (marginal)
    total2 = len(contents)
    iter1 = set(contents[:total2//2])
    iter2 = set(contents[total2//2:])
        
        
    #Instantiate Threads
    thread1 = threading.Thread(target = re_construct, kwargs=dict(path=path, files=iter1, month=month, columns=columns))
    thread2 = threading.Thread(target = re_construct, kwargs=dict(path=path, files=iter2, month=month, columns=columns))
        
        
    #Start each in parallel
    thread1.start()
    thread2.start()
        
        
    #threads wait for eachother to complete and merge.
    thread1.join()
    thread2.join()
    
    print("Multithreading Complete")
    return "SUCCESS!"


#     

        
    

        