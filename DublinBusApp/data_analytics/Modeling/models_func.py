import pandas as pd
import numpy as np

from datetime import date, datetime
from patsy import dmatrices

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import export_graphviz, DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.tree import export_graphviz 
from sklearn import tree
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 
from statsmodels.formula.api import ols

import os
import time
import multiprocessing
from multiprocessing import Pool
import threading

import shutil

def generate_models(month):
    
    path = os.path.join(os.getcwd(), month + "_PKLS")
    os.makedirs(path, exist_ok=True)
    
    path = os.path.join(os.getcwd(), month + "_tests")
    os.makedirs(path, exist_ok=True)
    print("Folders Made")
    
    path = os.path.join(os.getcwd(), month + "_routes")
    contents = os.listdir(path)
    
    total2 = len(contents)
    iter1 = set(contents[:total2//2])
    iter2 = set(contents[total2//2:])
    
    
    thread1 = threading.Thread(target = model_file, 
                               kwargs=dict(path=path,                       
                                           files=iter1,
                                           month=month))
    
    thread2 = threading.Thread(target = model_file, 
                               kwargs=dict(path=path,
                                           files=iter2, 
                                           month=month))
        
    thread1.start()
    thread2.start()
        
    thread1.join()
    thread2.join()
    
    
    
    path = os.path.join( os.getcwd(), month + "_tests")
    tempfiles = os.listdir(path)
    f = open(month+"_Tests.txt", "w+")
    print("merging files...")
    for infile in tempfiles:
        path = os.path.join(os.getcwd(),month + "_tests", infile)
        infile = open(path)
        f.write(infile.read())
        infile.close()
    f.close
    print("files merged!")
    
    
    directory_address = os.path.join(os.getcwd(), month + "_tests")
    shutil.rmtree(directory_address, ignore_errors=False, onerror = None)
    print("removed tree")

    print("Multithreading Complete")
    return "SUCCESS!"
    
    
    

def model_file(path, files, month):
    read = pd.read_csv
    
    for data in files:
        print("Modeling", month ,data)
        test_address = os.path.join(month + "_Tests", data[:-4] + "RF.txt")    
        f = open(test_address,"w+")
        
        read_address = os.path.join(path, data)
        modify_me = read(read_address, 
                         index_col=None, 
                         header=0, 
                         encoding="utf-8", 
                         converters = {"Journey_Pattern_ID":str,
                                       "LineID":str,
                                       "Direction":str,
                                       "Stop_ID":str,
                                       "Time_Bin_Start":str,
                                       "Scheduled_Speed_Per_Stop":float,
                                       "Stops_To_Travel":int})
        
        
        y, X = dmatrices('Time_To_Travel ~ Day_Of_Week + Time_Bin_Start + Scheduled_Speed_Per_Stop + Stops_To_Travel + Stop_Sequence',
                         modify_me,
                         return_type="dataframe") 
        y = np.ravel(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=33) 
        
        scaler = preprocessing.StandardScaler().fit(X_train)
        
        pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=10))
        
        hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                            'randomforestregressor__max_depth': [None, 5, 3, 1]}
        
        clf = GridSearchCV(pipeline,
                           hyperparameters,
                           cv=8)
        
        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_test)
        
        title = "\n\n===" + data +" Random Forest==="
        f.write( title )
        
        r2 = "\nr2 : " + str( r2_score(y_test, pred) )
        f.write( r2  )
       
        mse = "\nmse : " + str( mean_squared_error(y_test, pred) )
        f.write( mse )
        
        f.close
        
        write_address = os.path.join(month + "_PKLS", data +"rf_regressor.pkl")
        joblib.dump(clf, write_address)
        print("PKL saved for",month, data)
    
    