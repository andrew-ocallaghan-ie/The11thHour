```#------------------------------------------------------------------------------#
#------------------------------IMPORTS-----------------------------------------#
#------------------------------------------------------------------------------#
#https://docs.python.org/3/library/concurrent.futures.html
#from concurrent.futures import ThreadPoolExecutor #not needed
from concurrent.futures import ProcessPoolExecutor
from datetime import date
#conda install -c conda-forge geopandas
#http://geopandas.org/
from geopandas import GeoDataFrame
from geopandas import sjoin
#https://pandas.pydata.org/pandas-docs/stable/
import pandas as pd
#http://numba.pydata.org/numba-doc/dev/user/vectorize.html
from numba import vectorize
#http://www.numpy.org/
import numpy as np
#https://pypi.python.org/pypi/tqdm
from tqdm import tqdm
#https://docs.python.org/3/library/os.html
from os import getcwd
from os import listdir
from os import makedirs
from os import path
#http://toblerity.org/shapely/shapely.geometry.html
from shapely.geometry import Point
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn import preprocessing
#sklearn.ensemble.RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
#http://scikit-learn.org/stable/modules/model_persistence.html
from sklearn.externals import joblib
#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score #this can be negative, see docs!
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.classification import accuracy_score
#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline
from sklearn.pipeline import make_pipeline
#https://docs.python.org/3.6/library/time.html
from time import time
#https://docs.python.org/3.1/library/warnings.html
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category = DtypeWarning)
#------------------------------------------------------------------------------#
#------------------------------CLASSES-----------------------------------------#
#------------------------------------------------------------------------------#
 
class cleaning:
    #https://data.dublinked.ie/dataset/dublin-bus-gps-sample-data-from-dublin-city-council-insight-project
    '''Class that carries out cleaning action on a dataframe.
     - initialised with a pandas dataframe of raw Dublin Bus AVL data
     - assumes column names from DublinBus AVL data
     - single attribute 'df'
     - calls the 'clean' method to execute all other methods in order '''
    #--------------------------------------------------------------------------#
    def __init__(self, df):
        '''initialses dataframe as df, and sets columns'''
        self.df = df
        columns   = ['Timestamp', 'LineID', 'Direction', 'Journey_Pattern_ID', 
                    'Timeframe',  'Vehicle_Journey_ID', 'Operator', 
                    'Congestion', 'Lon', 'Lat', 'Delay', 'Block_ID',
                    'Vehicle_ID', 'Stop_ID', 'At_Stop']
        self.df.columns = columns
    #--------------------------------------------------------------------------#
    
    def drop_useless_columns(self):
        '''drops poorly defined, un-used, non-critical columns'''
        useless = ['Congestion',  #ambiguous definition and behaviour
                   'Delay',       #ambiguous definition and behaviour
                   'Block_ID',    #ambiguous definition and behaviour
                   'Operator',    #not used in analysis
                   'At_Stop',     #innacurate
                   'Stop_ID'      #innacurate
                   ]
        self.df = self.df.drop(useless, axis=1)
                
    #--------------------------------------------------------------------------#
    def fix_journey_pattern_id(self):
        '''takes nulls in Journey Patter ID and re-derives them using this method'''
        single_journeys = [ 'LineID', 'Vehicle_Journey_ID', 'Vehicle_ID' ]
        grouped_df = self.df.groupby(single_journeys, sort=False)
            
        def re_derive_nulls(df):
            '''if there are 2 Journey Pattern IDs  AND one is null AND other is valid,
             reassign whole group to valid the Journey Pattern ID'''
            options =set(df.Journey_Pattern_ID.unique()) 
            if len(options) == 2 and 'null' in options:
                options.remove('null')
                valid = options.pop()
                df.Journey_Pattern_ID = valid
            return df
          
        self.df =  grouped_df.apply(re_derive_nulls).reset_index()
        
    #--------------------------------------------------------------------------# 
    
    def drop_literal_nulls(self):
        '''drops rows where Journey Pattern ID is not 'null' str literal'''
        self.df = self.df[self.df.Journey_Pattern_ID != 'null' ]
        
    #--------------------------------------------------------------------------#  
    
    def fix_direction_column(self):
        '''Re-derives Direction from Journey Patter ID. 4th index'''
        self.df.Journey_Pattern_ID = self.df.Journey_Pattern_ID.astype('str')    
        self.df.Direction = self.df.Journey_Pattern_ID.str[4]
    
    #--------------------------------------------------------------------------#
    
    def fix_line_id(self):
        '''Re-derives LineID from Journey Pattern ID. first indices [0-4)'''
        self.df.Journey_Pattern_ID = self.df.Journey_Pattern_ID.astype('str')
        self.df.LineID = self.df.Journey_Pattern_ID.str[:4]
        self.df.LineID = self.df.LineID.str.lstrip('0') #strips leading zeros
   
    #--------------------------------------------------------------------------#
    
    def remove_ideling(self):
        '''drops lat lon replicas for idelling in a certain locality'''
        self.df = self.df.sort_values('Timestamp')
        single_journeys = ['Journey_Pattern_ID', 'Vehicle_Journey_ID',
                           'Vehicle_ID', 'Lat', 'Lon']
        self.df.drop_duplicates(single_journeys, keep='first')
    
    #--------------------------------------------------------------------------#
    
    def drop_not_stop_info(self):
        '''merges processed N.T.A. DublinBus stop info with raw data
         for resolving stop positions and stopIDs and sequence'''
        
        def empty_df():
            '''helps handle occasional goepandas sjoin error in a groupby
            where sjoin is expected to return an empty dataframe, it instead fails
            this constructs the empty df allowing all the results of 
            'groupby(x).apply(closest_stop) to be re_concatenated into 1 df'''
            
            columns = ['Lat_left', 'Lon_left', 'Stop_Sequence', 'geometry',
                       'index_right', 'Timestamp', 'LineID', 'Direction',
                       'Journey_Pattern_ID', 'Vehicle_Journey_ID', 'Lon_right',
                       'Lat_right', 'Vehicle_ID'] #column structure of 'combined'            
            return  pd.DataFrame(columns=columns)
        
        def init_geo_df(df):
            '''inits geodf given source df with Lat Lon attributes'''
            crs = {'init': 'epsg:4326'}
            geometry = [Point(xy) for xy in zip(df.Lon, df.Lat)]
            return GeoDataFrame(df, crs=crs, geometry=geometry)
        
        seq_df= pd.read_hdf('route_seq')
        seq_df = seq_df[['LineID', 'Lat', 'Lon', 'Direction','Stop_Sequence']]
        seq_df = init_geo_df(seq_df)
        seq_df.geometry = seq_df.geometry.buffer(0.004) #catchment radius for stop, in degrees
        self.df = init_geo_df(self.df)
        
        def closest_stop(df):
            '''maps each row for route and direction permutation to...
             nearest valid stop within that route-direction set'''
            route = df.LineID.unique()[0]
            direction = df.Direction.unique()[0]
            seq_df2 = seq_df[(seq_df.LineID==route) & (seq_df.Direction==direction)]
            seq_df2 = seq_df2.drop(['LineID','Direction'], axis = 1)
            try:
                combined = sjoin(seq_df2, df, how='inner', op='intersects')
                return combined
            except: 
                print('viva la excepcion!')
                return empty_df() #I handle the geopandas empty array error!    
        
        pattern = ['LineID', 'Direction']
        self.df = self.df.groupby(pattern, sort=False).apply(closest_stop).reset_index(drop=True)
        
        @vectorize(nopython=True)
        def distance(x1,y1,x2,y2):
            '''performs euclidean 2d euclidean distance calculation on arrays'''
            return (x1-x2)**2 + (y1-y2)**2        
        self.df['dist'] = distance(self.df.Lat_left.values,  self.df.Lon_left.values, 
                                   self.df.Lat_right.values, self.df.Lon_right.values)
        
        def closest_entry(df):
            '''takes rows for each journey that minimise distance to local stop'''
            df.sort_values(['dist'])
            return df[df.dist == df.dist.min()]
        
        single_journeys = ['Journey_Pattern_ID','Vehicle_Journey_ID',
                           'Vehicle_ID', 'Stop_Sequence']
        self.df = self.df.groupby(single_journeys, sort=False).apply(closest_entry).reset_index(drop=True)
            
    #--------------------------------------------------------------------------#
    
    def clean(self):
        '''executes above instructions'''
        self.drop_useless_columns()
        self.fix_journey_pattern_id() 
        self.drop_literal_nulls()    
        self.fix_direction_column()  
        self.fix_line_id()         
        self.remove_ideling()  
        self.drop_not_stop_info()
        return self.df
        
    #--------------------------------------------------------------------------#
class preparing:
    '''class prepares dataframe with derived columns necessary for modeling
    - single attribute 'df' 
    - 'prepare' method carries out operations in order'''
    def __init__(self, dataframe):
        '''initialises dataframe as df'''
        self.df = dataframe
        
    #--------------------------------------------------------------------------#
    def create_time_categories(self):
        '''creates Time, Day, Hour and Time Bin Start
        - Time bin start represents different 15 minute periods of the day
        - Deletes intermediate columns that are not re-used'''
        self.df['Time'] = pd.to_datetime(self.df.Timestamp, unit='us')
        self.df['Day_Of_Week'] = self.df.Time.dt.dayofweek
        self.df['Hour'] = self.df.Time.dt.hour.astype('str')
                
        def make_time_bin_start(df):
            '''creates Time Bin Start and drops intermediate columns'''
            df['Minute'] = df.Time.dt.minute
            df['Min_15'] = np.where((df.Minute > 15), '1', '0')
            df['Min_30'] = np.where((self.df.Minute > 30), '1', '0')
            df['Min_45'] = np.where((self.df.Minute > 45), '1', '0')
            time_list = ['Hour', 'Min_15', 'Min_30', 'Min_45']
            df['Time_Bin_Start'] = df[time_list].apply(lambda x: ''.join(x), axis=1)
            df = df.drop(['Minute', 'Min_15', 'Min_30','Min_45'], axis = 1)
            return df
        
        self.df = make_time_bin_start(self.df)
        self.df = self.df.drop(['Hour'], axis=1)
     
    #--------------------------------------------------------------------------#   
    
    def create_start_end_times(self):
        '''creates start times and end times and journey times columns for df'''
        as_delta = pd.to_timedelta
        single_journeys = ['Vehicle_Journey_ID', 'Vehicle_ID', 'Journey_Pattern_ID']
        grouped_df = self.df.groupby(single_journeys, sort=False)
        self.df['End_Time'] = grouped_df['Timestamp'].transform(max)
        self.df['Start_Time'] = grouped_df['Timestamp'].transform(min)
        self.df['Journey_Time'] = self.df.End_Time - self.df.Start_Time
        self.df.Journey_Time = as_delta(self.df.Journey_Time, unit='us').astype('timedelta64[m]')
    
    #--------------------------------------------------------------------------#
    
    def drop_impossible_journey_times(self):
        '''returns rows where journey times are greater than zero'''
        self.df = self.df[(self.df.Journey_Time) > 0]
            
    #-------------------------------------------------------------------