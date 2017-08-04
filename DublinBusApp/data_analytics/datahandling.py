#------------------------------------------------------------------------------#
#------------------------------IMPORTS-----------------------------------------#
#------------------------------------------------------------------------------#

#https://docs.python.org/3/library/concurrent.futures.html
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait

from datetime import date

#https://pandas.pydata.org/pandas-docs/stable/
import pandas as pd

#http://www.numpy.org/
import numpy as np


from tqdm import tqdm

from os import getcwd
from os import listdir
from os import makedirs
from os import path

#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn import preprocessing

#sklearn.ensemble.RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

#http://scikit-learn.org/stable/modules/model_persistence.html
from sklearn.externals import joblib

#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.classification import accuracy_score

#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline
from sklearn.pipeline import make_pipeline

from time import time

#------------------------------------------------------------------------------#
#------------------------------CLASSES-----------------------------------------#
#------------------------------------------------------------------------------#
 
class cleaning:
    #https://data.dublinked.ie/dataset/dublin-bus-gps-sample-data-from-dublin-city-council-insight-project
    '''Class that carries out cleaning action on a dataframe
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
    
    def drop_it_like_its_stop(self):
        '''drops any data classified as NOT at stop'''
        self.df = self.df[self.df.At_Stop == 1]

    #--------------------------------------------------------------------------#
    
    def drop_useless_columns(self):
        '''drops poorly defined, un-used, non-critical columns'''
        useless = ['Congestion', 'Delay', 'Block_ID', 'Operator', 'Lat', 'Lon', 'At_Stop']
        self.df = self.df.drop(useless, axis=1)
        
    #--------------------------------------------------------------------------#

    def fix_journey_pattern_id(self):
        '''takes nulls and re-derives them'''
        single_journeys = [ 'Timeframe', 'LineID', 'Vehicle_Journey_ID',  
                           'Vehicle_ID' ]
        grouped_df = self.df.groupby(single_journeys)
            
        def re_derive_nulls(df):
            '''if there are 2 Journey Pattern IDs  AND one is null AND the other is valid,
             reassign whole group to valid the Journey Pattern ID'''
            options =set(df.Journey_Pattern_ID.unique()) 
            if len(options) == 2 and 'null' in options:
                options.remove('null')
                valid = options.pop()
                df.Journey_Pattern_ID = valid
            return df
            
        self.df =  grouped_df.apply(re_derive_nulls)
        
    #--------------------------------------------------------------------------# 
    
    def drop_literal_nulls(self):
        '''drops rows where Journey Pattern ID is not 'null' literal'''
        self.df = self.df[self.df.Journey_Pattern_ID != 'null' ]
        
    #--------------------------------------------------------------------------#  
    
    def fix_direction_column(self):
        '''Re-derives Direction from Journey Patter ID'''
        self.df.Journey_Pattern_ID = self.df.Journey_Pattern_ID.astype('str')    
        self.df.Direction = self.df.Journey_Pattern_ID.str[4]
    
    #--------------------------------------------------------------------------#
    
    def fix_line_id(self):
        '''Re-derives LineID from Journey Pattern ID'''
        self.df.Journey_Pattern_ID = self.df.Journey_Pattern_ID.astype('str')
        self.df.LineID = self.df.Journey_Pattern_ID.str[:4]
        self.df.LineID = self.df.LineID.str.lstrip('0') #strips leading zeros
   
    #--------------------------------------------------------------------------#
    
    def remove_idling(self):
        '''creates stops_made column'''
        single_journeys = ['Timeframe', 'Journey_Pattern_ID', 'Vehicle_Journey_ID', 'Vehicle_ID']
        grouped_df = self.df.groupby(single_journeys)
        
        def remove_idle_at_stop(df):
            '''drops repeated stop_ids after at stop==1 filter'''
            return df.drop_duplicates(subset='Stop_ID', keep='first')
        
        self.df = grouped_df.apply(remove_idle_at_stop)
        
    #--------------------------------------------------------------------------#
    
    def clean(self):
        '''executes above instructions'''
        self.drop_it_like_its_stop() # drop at stop == 0
        self.drop_useless_columns() # drops lat, lon, delay etc.
        self.fix_journey_pattern_id() # re-derieves JPID
        self.drop_literal_nulls() # drops 'null' from JPID
        self.fix_direction_column() # actually fixes direction
        self.fix_line_id() # fixes lineid from JPID
        self.remove_idling() # 
        return self.df
        
    #--------------------------------------------------------------------------#

class preparing:
    '''class prepares dataframe with new columns
    - single attribute 'df' 
    - carry out operations in order'''

    def __init__(self, dataframe):
        '''initialises dataframe as df'''
        self.df = dataframe
        
    #--------------------------------------------------------------------------#

    def create_time_categories(self):
        '''creates Time, Day, Hour and Time Bin Start
        Time bin start represents different 15 minute periods of the day
        Deletes intermediate columns that are not re-used'''
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
        self.df = self.df.drop(["Hour"], axis=1)
     
    #--------------------------------------------------------------------------#   
    
    def create_start_end_times(self):
        '''creates start and end times columns for df'''
        single_journeys = ['Vehicle_Journey_ID', 'Timeframe', 'Vehicle_ID', 'Journey_Pattern_ID']
        self.df['End_Time'] = self.df.groupby(single_journeys)['Timestamp'].transform(max)
        self.df['Start_Time'] = self.df.groupby(single_journeys)['Timestamp'].transform(min)
    
    #--------------------------------------------------------------------------#
    
    def drop_impossible_journey_times(self):
        '''returns rows where journey times are greater than zero'''
        as_delta = pd.to_timedelta
        self.df = self.df[(self.df.End_Time - self.df.Start_Time) > 0]
            
    #--------------------------------------------------------------------------#       

    def create_holiday(self):
        '''creates bool column if date corresponds to a school holiday'''
        date_before = date(2012, 12, 31)
        date_after = date(2013, 1, 5)
        self.df['Holiday'] = np.where((self.df.Time.dt.date < date_after)  \
                                    & (self.df.Time.dt.date > date_before), 1, 0)
    
    #--------------------------------------------------------------------------#

    def create_scheduled_time_op(self):
        '''merges route_times with df to create scheduled time op column'''
    
        def get_times_dataframe():
            '''retrieves and prepares times dataframe'''
            times_dataframe = pd.read_hdf('route_times')
            times_dataframe = times_dataframe.rename(columns={'Route':'LineID'})
            times_dataframe = times_dataframe[['LineID', 'Scheduled_Time_OP']]
            return times_dataframe
            
        self.df = pd.merge(self.df, get_times_dataframe(), how='inner', on=['LineID'])
                
    #--------------------------------------------------------------------------#

    def create_stop_sequence(self): #create_stop_sequence
        '''creates stop sequence column by merging with sequence dataframe'''
        
        def get_sequence_dataframe():
            '''retrieves and prepares sequence dataframe'''
            sequence_dataframe = pd.read_hdf('route_seq')
            sequence_dataframe.LineID = sequence_dataframe.LineID.astype('str')
            sequence_dataframe.Direction = sequence_dataframe.Direction.astype('str')
            sequence_dataframe.Stop_ID = sequence_dataframe.Stop_ID.astype('str')
            unique_signiture = ['LineID', 'Direction', 'Destination']
            grouped_df = sequence_dataframe.groupby
            sequence_dataframe['Max_Stop_Sequence'] = grouped_df(unique_signiture).Stop_Sequence.transform(max)
            excess_columns = ['Stop_name', 'Lat', 'Lon', 'Destination']
            sequence_dataframe = sequence_dataframe.drop(excess_columns, axis=1)
            return sequence_dataframe
 
        shared_columns = ['LineID', 'Stop_ID', 'Direction']
        self.df = pd.merge(self.df, get_sequence_dataframe(), how='inner', on=shared_columns)
    
    #--------------------------------------------------------------------------#
    
    def create_scheduled_speed_per_stop(self):
        '''creates scheduled speed which is the scheduled mean rate of stop traversal'''
        as_delta = pd.to_timedelta
        single_journeys = ['Vehicle_Journey_ID', 'Timeframe', 'Vehicle_ID', 'Journey_Pattern_ID']
        self.df['End_Stop'] = self.df.groupby(single_journeys)['Stop_Sequence'].transform(max)
        self.df['Stops_To_Travel'] = (self.df.End_Stop.astype(int) - self.df.Stop_Sequence.astype(int))
        self.df['Scheduled_Speed_Per_Stop'] = self.df.Scheduled_Time_OP/self.df.Max_Stop_Sequence
        self.df['Time_To_Travel'] = self.df.End_Time - self.df.Timestamp
        self.df.Time_To_Travel = as_delta(self.df.Time_To_Travel, unit='us').astype('timedelta64[m]')
        self.df = self.df.drop(['End_Stop'], axis=1)    
    
    #--------------------------------------------------------------------------#
    
    def create_weather_columns(self):
        '''creates weather columns such as temerature and windspeed
        merges weather info with df by timestamp'''
        
        def get_weather_dataframe():
            '''gets weather dataframe and prepares it for merge'''
            weather_dataframe = pd.read_hdf('weather')
            weather_dataframe['Time'] = pd.to_datetime(weather_dataframe['Time'])
            weather_dataframe.sort_values(['Time'], ascending=[True], inplace=True)
            weather_dataframe['Hour_Of_Day'] = weather_dataframe['Time'].dt.hour
            weather_dataframe.sort_values(['Time'], ascending=[True], inplace=True)
            return weather_dataframe
        
        self.df.sort_values(['Time'], ascending=[True], inplace=True)
        self.df =  pd.merge_asof(self.df, get_weather_dataframe(), on='Time')
        
    #--------------------------------------------------------------------------#
    
    def drop_non_modeling_columns(self):
        '''drops columns that can't be modeled, improves write speed'''
        #timeframe, vjid, vid, stopid, timestamp
        useless = ['Time','Max_Stop_Sequence','Rain','Timeframe','Vehicle_Journey_ID',
                   'Vehicle_ID','Stop_ID','Timestamp']
        self.df = self.df = self.df.drop(useless, axis=1)
    
    #--------------------------------------------------------------------------#
   
    def prepare(self):
        '''applies predefined preparation methods'''
        self.create_time_categories()  # time, day, hour columns
        self.create_start_end_times() # creates start and end times
        self.drop_impossible_journey_times() # drops end_to_end journey times == 0
        self.create_holiday() # creates category for school holidays
        self.create_scheduled_time_op() # creates scheduled travel time
        self.create_stop_sequence() # creates stop sequences for routes
        self.create_scheduled_speed_per_stop() # scheduled speed
        self.create_weather_columns() # creates weather columns        self.drop_non_modeling_columns() # drops excess columns
        return self.df

#------------------------------------------------------------------------------#
    
class extracting:
    '''class for separating data by route
    is initialised with list of file paths'''
    
    def __init__(self):
        path_to_folder = path.join(getcwd(), 're_con')
        all_files = listdir(path_to_folder)
        path_to_files = list(map(lambda data: path.join(path_to_folder, data), all_files))
        self.files = path_to_files
    #--------------------------------------------------------------------------#
    
    def extract(self, route):
        '''extracts single route from files and returns df of that rooute'''
        accumulator = pd.read_hdf(self.files[0])
        accumulator = accumulator[accumulator.LineID == route].drop(['LineID'], axis=1)
        for data in self.files:
            df = pd.read_hdf(data)
            df =df[df.LineID == route].drop(['LineID'], axis=1)
            accumulator, df = accumulator.align(df, axis=1)
            accumulator = pd.concat([accumulator, df])
        return accumulator

#------------------------------------------------------------------------------#

class modelling:
    '''class that contains processes for modeling''' 
    
    def __init__(self):
        '''creats columns for record dataframe'''
        metric_columns = ['route', 'r2', 'rmse', 'mae', 'ten_percent-delta', 'five_min-delta','five_min_late','exp_variance']
        self.record = pd.DataFrame(columns=metric_columns)
    
    #--------------------------------------------------------------------------#    
        
    def create_model(self, data):
        '''creates rf regressor for data,'''
        data_columns = ["Day_Of_Week", "Time_Bin_Start", "Wind_Speed", "Temperature",
                   "Holiday", "Scheduled_Speed_Per_Stop", "Stops_To_Travel","Stop_Sequence"]
        self.X_train, self.X_test, self.y_train, self.y_test = \
        train_test_split(data[data_columns],np.ravel(data["Time_To_Travel"]), test_size=0.2, random_state=33)
        scaler = preprocessing.StandardScaler().fit(self.X_train)
        pipeline = make_pipeline(preprocessing.StandardScaler(), 
                                 RandomForestRegressor(n_estimators=10, n_jobs=1))
        hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                           'randomforestregressor__max_depth': [None, 5, 3, 1]}
        reg = GridSearchCV(pipeline, hyperparameters, cv=8)
        reg.fit(self.X_train, self.y_train)
        self.model = reg
        
    #--------------------------------------------------------------------------#

    def create_test_metrics(self):
        '''takes most recently created rf regressor and creates test metrics for it'''
        pred = self.model.predict(self.X_test)
        actual = self.y_test
        
        def create_tolerance_metrics():
            data = pd.DataFrame(self.X_test)
            data['actual'] = actual
            data["predicted"] = pred
            data['five_percent'] = np.where( abs(data.actual-data.predicted)/data.actual <= 0.1, 1, 0)
            data['five_min_delta'] = np.where(abs(data.actual-data.predicted)/data.actual <= 5, 1, 0)
            data['five_min_late'] = np.where( (data.actual-data.predicted)/data.actual <= 5, 1, 0)
            data['true'] = 1
            return data
        
        data = create_tolerance_metrics()
        self.rsquared =  r2_score(actual, pred)
        self.rmse = mean_squared_error(actual, pred)**0.5
        self.mae = mean_absolute_error(self.y_test, pred)
        self.rel_acc = accuracy_score(data.true, data.five_percent)
        self.abs_acc = accuracy_score(data.true, data.five_min_delta)
        self.late_acc = accuracy_score(data.true, data.five_min_late)
        self.exp_var = explained_variance_score(actual, pred)
 
    #--------------------------------------------------------------------------#
        
    def record_test_metrics(self, route):
        '''takes most recently evaluated test metrics and stores them in record'''
        values = [route, self.rsquared, self.rmse, self.mae, self.rel_acc, self.abs_acc, self.late_acc, self.exp_var]
        keys = self.record.columns
        row = dict(zip(keys, values))
        record = pd.DataFrame(data = row, columns = self.record.columns, index =[route])
        self.record = self.record.append(record)
        
    #--------------------------------------------------------------------------#
        
    def model_route(self, data, route):
        '''runs modeling process for a set of data'''
        self.create_model(data)
        self.create_test_metrics()
        self.record_test_metrics(route)
        return self.model
 
#------------------------------------------------------------------------------#
#------------------------------Functions---------------------------------------#
#------------------------------------------------------------------------------#
 
def timing(func):
    '''timing decorator for functions'''
    def wrapper(*arg, **kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time()
        res = func(*arg, **kw)
        t2 = time()
        return (t2 - t1), res, func.__name__
    return wrapper
        
#------------------------------------------------------------------------------#

def setup_folders():
    '''creates folder for re_con, routes and pkls'''
    for folder in ['routes', 're_con', 'pkls']:
        try:
            path_to_routes_folder = path.join(getcwd(), folder)
            makedirs(path_to_routes_folder, exist_ok=True)
        except:
            print(folder, 'made already')

#------------------------------------------------------------------------------#

def re_construction(data_file):
    print(data_file)
    path_to_folder = path.join(getcwd(), 'DayRawCopy')
    files = listdir(path_to_folder)
    path_to_file = path.join(path_to_folder, data_file)
    '''executes cleaning and prepatation on raw data files'''
    try:
        df = cleaning(pd.read_csv(path_to_file)).clean()
        df = preparing(df).prepare()
        write_address = path.join(getcwd(), 're_con' , data_file[:-4] + '.h5')
        df.to_hdf(write_address, key='moo', mode='w')
        return "wooot"
    except: 
        return "re_con fails..."    
 
 #-----------------------------------------------------------------------------#
    
def get_all_routes():
        '''gets all routes from files'''
        routes = set()
        path_to_folder = path.join(getcwd(), 're_con')
        all_files = listdir(path_to_folder)
        path_to_files = list(map(lambda data: path.join(path_to_folder, data), all_files))
        for data in path_to_files:
            df = pd.read_hdf(data)
            routes = routes.union(set(df.LineID.unique()))
        return routes
    
#------------------------------------------------------------------------------#    
    
def extraction(route):
    '''splits data by route'''
    results = []
    try:
        df = extracting().extract(route)
        write_address = path.join(getcwd(), 'routes', 'xbeta'+str(route)+'.h5')
        df.to_hdf(write_address, key='moo', mode = 'w')        
        results.append(route +'it worked!')
    except:
        results.append(route, 'for **** sake')
    return results

#------------------------------------------------------------------------------#

def quantification(data_file):
    path_to_folder = path.join(getcwd(), 'routes')
    path_to_file =  path.join(path_to_folder, data_file)
    route = data_file[5:-3]
    model = modelling()
    rf_reg = model.model_route(pd.read_hdf(path_to_file), route)
    write_address = path.join(getcwd(), 'pkls', str(route)+'rf.pkl')
    joblib.dump(rf_reg, write_address)
    return model.record

#------------------------------------------------------------------------------#
#-----------------------------Main Methods-------------------------------------#
#------------------------------------------------------------------------------#

def main1():
    '''multiprocesses cleaning and prep of raw data, cpu heavy'''
    path_to_raw = path.join(getcwd(), 'DayRawCopy')
    files = listdir(path_to_raw)
    with ProcessPoolExecutor(max_workers=4) as Executor:
        for file, result in tqdm(zip(files, Executor.map(re_construction, files, chunksize=4))):
            print(file[:-4], result)

#------------------------------------------------------------------------------#
   
def main2():
    '''seperates routes into seperate files, ready for training'''
    routes = get_all_routes()
    with ProcessPoolExecutor(max_workers=4) as Executor:
        for route, result in tqdm(zip(routes, Executor.map(extraction, routes, chunksize = 4))):
            print(route, result)
    
#------------------------------------------------------------------------------#
    
def main3():
    '''prepares a model for each route'''
    path_to_folder = path.join(getcwd(), 'routes' )
    files = listdir(path_to_folder)
    
    pool = ProcessPoolExecutor(max_workers=3)
    dataframes = []
    wag = zip(files, pool.map(quantification, files))
    for key, val in wag:
        dataframes.append(val)
    dataframe = pd.concat(dataframes, axis=0)
    dataframe.to_csv('Model_Summary')
    


#------------------------------------------------------------------------------#
#---------------------------If Run Directly------------------------------------#
#------------------------------------------------------------------------------#
        
if __name__ == '__main__':
    
    @timing
    def processing():
        setup_folders()
        t1=time()
        #main1()
        t2=time()
        print((t2-t1)//60)
        #main2()
        t3=time()
        print((t3-t2)//60)
        t4=time()
        main3()
        print((t4-t3)//60)

    processing()

