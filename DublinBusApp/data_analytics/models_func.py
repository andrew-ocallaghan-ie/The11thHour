import pandas as pd
import numpy as np

#this is the devil...
from patsy import dmatrices

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.externals import joblib 
from statsmodels.formula.api import ols

import os
import time
import multiprocessing
from multiprocessing import Pool
import threading

import shutil

"""todo use decorators to seperate admin logic from modeling logic"""
def generate_models(iterators):
    
    path = os.path.join(os.getcwd(), "xbeta_PKLS")
    os.makedirs(path, exist_ok=True)
    
    path = os.path.join(os.getcwd(),  "xbeta_tests")
    os.makedirs(path, exist_ok=True)
    print("Folders Made")
    
    path = os.path.join(os.getcwd(), "NovJan_routes")
    contents = os.listdir(path)
    
    total2 = len(contents)
    iter1 = set(contents[:total2//2])
    iter2 = set(contents[total2//2:])
    
    
    thread1 = threading.Thread(target = model_file, 
                               kwargs=dict(path=path,                       
                                           files=iterators,
                                           ))
    
    """thread2 = threading.Thread(target = model_file, 
                               kwargs=dict(path=path,
                                           files=iter2, 
                                           month=month))"""
        
    thread1.start()
    #thread2.start()
        
    thread1.join()
    #thread2.join()
    
    
    
    path = os.path.join( os.getcwd(),  "xbeta_tests")
    tempfiles = os.listdir(path)
    f = open("xbeta_Tests.txt", "w+")
    print("merging files...")
    for infile in tempfiles:
        path = os.path.join(os.getcwd(), "NovJan_tests", infile)
        infile = open(path)
        f.write(infile.read())
        infile.close()
    f.close
    print("files merged!")
    
    
    directory_address = os.path.join(os.getcwd(),  "NovJan_tests")
    shutil.rmtree(directory_address, ignore_errors=False, onerror = None)
    print("removed tree")

    print("Multithreading Complete")
    return "SUCCESS!"
    
    
    

def model_file(path, files):
    
    for data in files:
        print("Modeling",data)
        test_address = os.path.join("xbeta_Tests", data[:-3] + "RF.txt")
        
        read_address = os.path.join(path, data)
        """modify_me = read(read_address, 
                         index_col=None, 
                         header=0, 
                         encoding="utf-8", 
                         converters = {"Journey_Pattern_ID":str,
                                       "LineID":str,
                                       "Direction":str,
                                       "Stop_ID":str,
                                       "Time_Bin_Start":str,
                                       "Scheduled_Speed_Per_Stop":float,
                                       "Stops_To_Travel":int})"""
        
        modify_me = pd.read_hdf(read_address)
        modify_me.Direction = modify_me.Direction.astype("str")
        modify_me.Time_Bin_Start = modify_me.Time_Bin_Start.astype("str")
        modify_me.Scheduled_Speed_Per_Stop = modify_me.Scheduled_Speed_Per_Stop.astype(float)
        modify_me.Stops_To_Travel = modify_me.Stops_To_Travel.astype(int)
        columns = ["Day_Of_Week", "Time_Bin_Start", "Wind_Speed", "Temperature", "Holiday", "Scheduled_Speed_Per_Stop", "Stops_To_Travel","Stop_Sequence", "Time_To_Travel"]
        X = modify_me[columns]
        y = modify_me["Time_To_Travel"]
        print("mod orig", modify_me.shape)
        rows,cols = modify_me.shape
        
        """y, X = dmatrices('Time_To_Travel ~ Day_Of_Week + Time_Bin_Start + Wind_Speed + Temperature + Holiday + Scheduled_Speed_Per_Stop + Stops_To_Travel + Stop_Sequence',
                         modify_me,
                         return_type="dataframe")"""
        print(X.shape, y.shape)
        #X = np.array(X).reshape(rows,cols-1)
        
        y = np.ravel(y)
        #y = np.array(y).reshape(rows,1)
        print(X.shape, y.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=33) 
        
        scaler = preprocessing.StandardScaler().fit(X_train)
        
        pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=10, n_jobs=1))
        
        hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                            'randomforestregressor__max_depth': [None, 5, 3, 1]}
        
        clf = GridSearchCV(pipeline,
                           hyperparameters,
                           cv=8)
        
        ahh =  [[
                1, #intercept
                1, #day
                "9111", #time_bin_start
                22, # wind
                22, # temp
                1, # holiday
                1, # sched speed per stop
                10, # stopts to travel
                22 #src stop seq
                ]]
        
        #ahh = np.array(ahh).reshape(1,-1)
        
        clf.fit(X_train, y_train)
        print(X_test.shape)
        pred = clf.predict(X_test)
        print(clf.predict(ahh)) # stop_seq
        
        
        f = open(test_address, "w+")
        title = "\n\n===" + data +" Random Forest==="
        f.write( title )
        
        r2 = "\nr2 : " + str( r2_score(y_test, pred) )
        f.write( r2  )
       
        mse = "\nmse : " + str( mean_squared_error(y_test, pred) )
        f.write( mse )
        
        f.close
        
        write_address = os.path.join("xbeta_PKLS", data +"rf_regressor.pkl")
        joblib.dump(clf, write_address)
        pred = joblib.load(write_address)
        print(pred.predict([[1, #intercept
                             1, #day
                             "9111", #time_bin_start
                             "22", # wind
                             "22", # temp
                             1, # holiday
                             1, # sched speed per stop
                             10, # stopts to travel
                             22 ]])) # stop_seq
        
        print("PKL saved for NovJan", data)
    
    