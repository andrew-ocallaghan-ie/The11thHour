import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import os

class cleaning:
    '''Class that carries out cleaning action on a dataframe
     - initialised with a pandas dataframe of raw Dublin Bus AVL data
     - assumes column names from DublinBus AVL data
     - single attribute 'df'
     - methods should be applied in order'''
    #------------------------------------------------------------------------------#

    def __init__(self, df):
        '''initialses dataframe as df, and sets columns'''
        self.df = df
        columns   = ['Timestamp',
                    'LineID', 
                    'Direction',
                    'Journey_Pattern_ID', 
                    'Timeframe', 
                    'Vehicle_Journey_ID', 
                    'Operator', 
                    'Congestion', 
                    'Lon',
                    'Lat', 
                    'Delay', 
                    'Block_ID',
                    'Vehicle_ID',
                    'Stop_ID',
                    'At_Stop']
        self.df.columns = columns
        pass

    #------------------------------------------------------------------------------#
    
    def drop_it_like_its_stop(self):
        '''drops any data classified as NOT at stop'''
        self.df = self.df[self.df.At_Stop == 1]

    #------------------------------------------------------------------------------#
    
    def drop_useless_columns(self):
        '''drops poorly defined, un-used, non-critical columns'''
        useless = ['Congestion', 'Delay', 'Block_ID', 'Operator', 'Lat', 'Lon', 'At_Stop']
        self.df = self.df.drop(useless, axis=1)
        
    #------------------------------------------------------------------------------#

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
        
    #------------------------------------------------------------------------------# 
    
    def drop_literal_nulls(self):
        '''drops rows where Journey Pattern ID is not 'null' literal'''
        self.df = self.df[self.df.Journey_Pattern_ID != 'null' ]
        
    #------------------------------------------------------------------------------#  
    
    def fix_direction_column(self):
        '''Re-derives Direction from Journey Patter ID'''
        self.df.Journey_Pattern_ID = self.df.Journey_Pattern_ID.astype('str')    
        self.df.Direction = self.df.Journey_Pattern_ID.str[4]
    
    #------------------------------------------------------------------------------#
    
    def fix_line_id(self):
        '''Re-derives LineID from Journey Pattern ID'''
        self.df.Journey_Pattern_ID = self.df.Journey_Pattern_ID.astype('str')
        self.df.LineID = self.df.Journey_Pattern_ID.str[:4]
        self.df.LineID = self.df.LineID.str.lstrip('0') #strips leading zeros
   
    #------------------------------------------------------------------------------#
    
    def remove_idling(self):
        '''creates stops_made column'''
        single_journeys = ['Timeframe', 'Journey_Pattern_ID', 
                           'Vehicle_Journey_ID', 'Vehicle_ID', 'Direction']
        grouped_df = self.df.groupby(single_journeys)
        
        def remove_idle_at_stop(df):
            '''drops repeated stop_ids after at stop==1 filter'''
            return df.drop_duplicates(subset='Stop_ID', keep='first')
        
        self.df = grouped_df.apply(remove_idle_at_stop)
        
    #------------------------------------------------------------------------------#
    
    def clean(self):
        '''executes above instructions'''
        self.drop_it_like_its_stop() # drop at stop == 0
        self.drop_useless_columns() # drops lat, lon, delay etc.
        #self.patch_direction_column() #small fix for direction
        self.fix_journey_pattern_id() # re-derieves JPID
        self.drop_literal_nulls() # drops 'null' from JPID
        self.fix_direction_column() # actually fixes direction
        self.fix_line_id() # fixes lineid from JPID
        self.remove_idling() # 
        return self.df
        
    #------------------------------------------------------------------------------#

class preparing:
    '''class prepares dataframe with new columns
    - single attribute 'df' 
    - carry out operations in order'''

    def __init__(self, dataframe):
        '''initialises dataframe as df'''
        self.df = dataframe
        
    #------------------------------------------------------------------------------#

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
        self.df = self.df.drop(["Hour"])
     
    #------------------------------------------------------------------------------#   
    
    def create_journey_times(self):
        '''creates journey time which is end to end time'''
        as_delta = pd.to_timedelta
        single_journeys = ['Vehicle_Journey_ID', 'Timeframe',
                           'Vehicle_ID', 'Journey_Pattern_ID', 'Direction']
        self.df['End_Time'] = self.df.groupby(single_journeys)['Timestamp'].transform(max)
        self.df['Start_Time'] = self.df.groupby(single_journeys)['Timestamp'].transform(min)
        self.df['Journey_Time'] = ( self.df.End_Time - self.df.Start_Time )
        self.df.Journey_Time = as_delta(self.df.Journey_Time, unit='us').astype('timedelta64[m]')
        
    #------------------------------------------------------------------------------#
       
    def drop_impossible_journey_times(self):
        '''returns rows where journey times are greater than zero'''
        self.df = self.df[self.df.Journey_Time > 0]
        self.df = self.df.drop(["Journey_Time"])
    
    #------------------------------------------------------------------------------#

    def create_holiday(self):
        '''creates bool column if date corresponds to a school holiday'''
        date_before = datetime.date(2012, 12, 31)
        date_after = datetime.date(2013, 1, 5)
        self.df['Holiday'] = np.where((self.df.Time.dt.date < date_after) \
                                      & (self.df.Time.dt.date > date_before), 1, 0)
    
    #------------------------------------------------------------------------------#

    def create_scheduled_time_op(self):
        '''merges route_times with df to create scheduled time op column'''
    
        def get_times_dataframe():
            '''retrieves and prepares times dataframe'''
            times_dataframe = pd.read_hdf('route_times')
            times_dataframe = times_dataframe.rename(columns={'Route':'LineID'})
            times_dataframe = times_dataframe[['LineID', 'Scheduled_Time_OP']]
            return times_dataframe
            
        self.df = pd.merge(self.df, get_times_dataframe(), how='inner', on=['LineID'])
                
    #------------------------------------------------------------------------------#

    def create_stop_sequence(self): #create_stop_sequence
        '''creates stop sequence column by merging with sequence dataframe'''
        
        def get_sequence_dataframe():
            '''retrieves and prepares sequence dataframe'''
            sequence_dataframe = pd.read_hdf('route_seq')
            sequence_dataframe.LineID = sequence_dataframe.LineID.astype('str')
            sequence_dataframe.Direction = sequence_dataframe.Direction.astype('str')
            sequence_dataframe.Stop_ID = sequence_dataframe.Stop_ID.astype('str')
            unique_signiture = ['LineID', 'Direction', 'Destination']
            sequence_dataframe['Max_Stop_Sequence'] = sequence_dataframe.groupby(unique_signiture).Stop_Sequence.transform(max)
            excess_columns = ['Stop_name', 'Lat', 'Lon', 'Destination']
            sequence_dataframe = sequence_dataframe.drop(excess_columns, axis=1)
            return sequence_dataframe
 
        shared_columns = ['LineID', 'Stop_ID', 'Direction']
        self.df = pd.merge(self.df, get_sequence_dataframe(), how='inner', on=shared_columns)
    
    #------------------------------------------------------------------------------#
    
    def create_scheduled_speed_per_stop(self):
        '''creates scheduled speed which is the scheduled mean rate of stop traversal'''
        as_delta = pd.to_timedelta
        single_journeys = ['LineID', 'Vehicle_Journey_ID', 'Timeframe', 
                           'Vehicle_ID', 'Journey_Pattern_ID']
        self.df['Start_Stop'] = self.df.groupby(single_journeys)['Stop_Sequence'].transform(min)
        self.df['End_Stop'] = self.df.groupby(single_journeys)['Stop_Sequence'].transform(max)
        self.df['Stops_To_Travel'] = self.df.End_Stop.astype(int) - self.df.Start_Stop.astype(int)
        self.df = self.df.drop(['Start_Stop', 'End_Stop'], axis=1)
        self.df['Time_To_Travel'] = self.df.End_Time - self.df.Timestamp
        self.df.Time_To_Travel = as_delta(self.df.Time_To_Travel, unit='us').astype('timedelta64[m]')
        self.df['Scheduled_Speed_Per_Stop'] = self.df.Scheduled_Time_OP/self.df.Max_Stop_Sequence    
    
    #------------------------------------------------------------------------------#
    
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
        
    #------------------------------------------------------------------------------#
    
    def drop_non_modeling_columns(self):
        '''drops columns that can't be modeled'''
        useless = ['Timestamp',
                   'Timeframe',
                   'Time',
                   'End_Time',
                   'Start_Time',
                   'Max_Stop_Sequence',
                   'Journey_Pattern_ID', 
                   'Vehicle_Journey_ID', 
                   'Vehicle_ID',
                   'Stop_ID',
                   'Rain']
        self.df = self.df = self.df.drop(useless, axis=1)
    
    #------------------------------------------------------------------------------#
    def prepare(self):
        '''applies predefines preparation methods'''
        self.create_time_categories()  # time, day, hour columns
        self.create_journey_times() # end to end journey time
        self.drop_impossible_journey_times() # drops journey times == 0
        self.create_holiday() # creates category for school holidays
        self.create_scheduled_time_op() # creates scheduled travel time
        self.create_stop_sequence() # creates stop sequences for routes
        self.create_scheduled_speed_per_stop() # scheduled speed
        self.create_weather_columns() # creates weather columns
        self.drop_non_modeling_columns()
        return self.df

    #------------------------------------------------------------------------------#
    
class extracting:
    '''class for seperating data by route'''
    
    def __init__(self, files, month, columns):
        self.routes = set()
        self.month = month
        self.files = files
        self.columns = columns
    
    def get_all_routes(self):
        '''gets all routes from csvs'''
        for data in self.files:
            df = pd.read_hdf(data)
            self.routes = self.routes.union(set(df.LineID.unique()))
    
    def extract(self, route):
        '''extracts single route'''
        accumulator = pd.read_hdf(self.files[0])
        accumulator.columns = self.columns
        for data in self.files:
            df = pd.read_hdf(data)
            df = pd.DataFrame(df)
            #accumulator, df = accumulator.align(df, axis=1)
            accumulator = pd.concat([accumulator, df])
        return accumulator

#----------------------------------------------------------------------------------#

def setup(month):
        '''creates folders'''
        path_to_routes_folder = os.path.join(os.getcwd(), month + '_routes')
        os.makedirs(path_to_routes_folder, exist_ok=True)
        path_to_re_con = os.path.join(os.getcwd(), 're_con_' + month)
        os.makedirs(path_to_re_con, exist_ok=True)

#----------------------------------------------------------------------------------#

def re_construction(month):
    '''execustes cleaning and prepatation on raw data files'''
    path_to_folder = os.path.join(os.getcwd(), month + 'DayRawCopy')
    files = os.listdir(path_to_folder)
    for data_file in tqdm(files):
        file_address = os.path.join(path_to_folder, data_file)
        df = pd.read_csv(file_address)
        df = cleaning(df).clean()
        df = preparing(df).prepare()
        write_address = os.path.join(os.getcwd(), 're_con_' + month, data_file[:-4] + '.h5')
        df.to_hdf(write_address, key='moo', mode='w')

#----------------------------------------------------------------------------------#

def extraction(month):
    '''splits data by route'''
    path_to_folder = os.path.join(os.getcwd(), 're_con_'+month)
    files = os.listdir(path_to_folder)
    path_to_files = list(map(lambda data: os.path.join(path_to_folder, data), files))
    columns = pd.read_hdf(path_to_files[0]).columns
    data = extracting(path_to_files, month, columns)
    data.get_all_routes()
    for route in tqdm(data.routes):
        write_address = os.path.join(os.getcwd(), month+'_routes', 'xbeta'+str(route)+'.h5')
        df = data.extract(route)
        print(df.shape)
        df.to_hdf(write_address, key='moo', mode = 'w')

#----------------------------------------------------------------------------------#

def processing(month):
    #setup(month)
    #re_construction(month)
    extraction(month)
    return 'Finished'

def join_months(iterator):
    path = os.path.join(os.getcwd(), 'NovJan_routes')
    os.makedirs(path, exist_ok=True)
    Nov_path = os.path.join(os.getcwd(), 'Nov_routes')
    Jan_path = os.path.join(os.getcwd(), 'Jan_routes')
    contents = os.listdir(Nov_path)
    for file in iterator:
        if file != '.DS_Store':
            Nov_path_to_file = os.path.join(Nov_path, file)
            Nov_dataframe = pd.read_hdf(Nov_path_to_file)
            Jan_path_to_file = os.path.join(Jan_path, file)
            Jan_dataframe = pd.read_hdf(Jan_path_to_file)
            Jan_dataframe, Nov_dataframe = Jan_dataframe.align(Nov_dataframe, axis=1)
            Combined_dataframe = pd.concat([Jan_dataframe,Nov_dataframe])
            write_address = os.path.join(path,'x' + file)
            Combined_dataframe.to_hdf(write_address, key='moo', mode='w')

    return 'success'

if __name__ == '__main__':
    import multiprocessing
    from multiprocessing import Pool
        
    pool = Pool(processes=2)
    months=['Jan', 'Nov']
    result = pool.map(processing, months)
    pool.close()
    pool.join()
    print('Result: ', result)
    
#     
    
=======
import pandas as pd
import numpy as np
import datetime
from data_analytics.Cleaning.decorators import print_func
from tqdm import tqdm
import os

class cleaning:
    '''Class that carries out cleaning action on a dataframe
     - initialised with a pandas dataframe of raw Dublin Bus AVL data
     - assumes column names from DublinBus AVL data
     - single attribute 'df'
     - methods should be applied in order'''
    #------------------------------------------------------------------------------#

    def __init__(self, df):
        '''initialses dataframe as df, and sets columns'''
        self.df = df
        columns   = ['Timestamp',
                    'LineID', 
                    'Direction',
                    'Journey_Pattern_ID', 
                    'Timeframe', 
                    'Vehicle_Journey_ID', 
                    'Operator', 
                    'Congestion', 
                    'Lon',
                    'Lat', 
                    'Delay', 
                    'Block_ID',
                    'Vehicle_ID',
                    'Stop_ID',
                    'At_Stop']
        self.df.columns = columns
        pass

    #------------------------------------------------------------------------------#
    
    def drop_it_like_its_stop(self):
        '''drops any data classified as NOT at stop'''
        self.df = self.df[self.df.At_Stop == 1]

    #------------------------------------------------------------------------------#
    
    def drop_useless_columns(self):
        '''drops poorly defined, un-used, non-critical columns'''
        useless = ['Congestion', 'Delay', 'Block_ID', 'Operator', 'Lat', 'Lon', 'At_Stop']
        self.df = self.df.drop(useless, axis=1)
        
    #------------------------------------------------------------------------------#

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
        
    #------------------------------------------------------------------------------# 
    
    def drop_literal_nulls(self):
        '''drops rows where Journey Pattern ID is not 'null' literal'''
        self.df = self.df[self.df.Journey_Pattern_ID != 'null' ]
        
    #------------------------------------------------------------------------------#  
    
    def fix_direction_column(self):
        '''Re-derives Direction from Journey Patter ID'''
        self.df.Journey_Pattern_ID = self.df.Journey_Pattern_ID.astype('str')    
        self.df.Direction = self.df.Journey_Pattern_ID.str[4]
    
    #------------------------------------------------------------------------------#
    
    def fix_line_id(self):
        '''Re-derives LineID from Journey Pattern ID'''
        self.df.Journey_Pattern_ID = self.df.Journey_Pattern_ID.astype('str')
        self.df.LineID = self.df.Journey_Pattern_ID.str[:4]
        self.df.LineID = self.df.LineID.str.lstrip('0') #strips leading zeros
   
    #------------------------------------------------------------------------------#
    
    def remove_idling(self):
        '''creates stops_made column'''
        single_journeys = ['Timeframe', 'Journey_Pattern_ID', 
                           'Vehicle_Journey_ID', 'Vehicle_ID', 'Direction']
        grouped_df = self.df.groupby(single_journeys)
        
        def remove_idle_at_stop(df):
            '''drops repeated stop_ids after at stop==1 filter'''
            return df.drop_duplicates(subset='Stop_ID', keep='first')
        
        self.df = grouped_df.apply(remove_idle_at_stop)
        
    #------------------------------------------------------------------------------#
    
    def clean(self):
        '''executes above instructions'''
        self.drop_it_like_its_stop() # drop at stop == 0
        self.drop_useless_columns() # drops lat, lon, delay etc.
        #self.patch_direction_column() #small fix for direction
        self.fix_journey_pattern_id() # re-derieves JPID
        self.drop_literal_nulls() # drops 'null' from JPID
        self.fix_direction_column() # actually fixes direction
        self.fix_line_id() # fixes lineid from JPID
        self.remove_idling() # 
        return self.df
        
    #------------------------------------------------------------------------------#

class preparing:
    '''class prepares dataframe with new columns
    - single attribute 'df' 
    - carry out operations in order'''

    def __init__(self, dataframe):
        '''initialises dataframe as df'''
        self.df = dataframe
        
    #------------------------------------------------------------------------------#

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
        self.df = self.df.drop(["Hour"])
     
    #------------------------------------------------------------------------------#   
    
    def create_journey_times(self):
        '''creates journey time which is end to end time'''
        as_delta = pd.to_timedelta
        single_journeys = ['Vehicle_Journey_ID', 'Timeframe',
                           'Vehicle_ID', 'Journey_Pattern_ID', 'Direction']
        self.df['End_Time'] = self.df.groupby(single_journeys)['Timestamp'].transform(max)
        self.df['Start_Time'] = self.df.groupby(single_journeys)['Timestamp'].transform(min)
        self.df['Journey_Time'] = ( self.df.End_Time - self.df.Start_Time )
        self.df.Journey_Time = as_delta(self.df.Journey_Time, unit='us').astype('timedelta64[m]')
        
    #------------------------------------------------------------------------------#
       
    def drop_impossible_journey_times(self):
        '''returns rows where journey times are greater than zero'''
        self.df = self.df[self.df.Journey_Time > 0]
        self.df = self.df.drop(["Journey_Time"])
    
    #------------------------------------------------------------------------------#

    def create_holiday(self):
        '''creates bool column if date corresponds to a school holiday'''
        date_before = datetime.date(2012, 12, 31)
        date_after = datetime.date(2013, 1, 5)
        self.df['Holiday'] = np.where((self.df.Time.dt.date < date_after) \
                                      & (self.df.Time.dt.date > date_before), 1, 0)
    
    #------------------------------------------------------------------------------#

    def create_scheduled_time_op(self):
        '''merges route_times with df to create scheduled time op column'''
    
        def get_times_dataframe():
            '''retrieves and prepares times dataframe'''
            times_dataframe = pd.read_hdf('route_times')
            times_dataframe = times_dataframe.rename(columns={'Route':'LineID'})
            times_dataframe = times_dataframe[['LineID', 'Scheduled_Time_OP']]
            return times_dataframe
            
        self.df = pd.merge(self.df, get_times_dataframe(), how='inner', on=['LineID'])
                
    #------------------------------------------------------------------------------#

    def create_stop_sequence(self): #create_stop_sequence
        '''creates stop sequence column by merging with sequence dataframe'''
        
        def get_sequence_dataframe():
            '''retrieves and prepares sequence dataframe'''
            sequence_dataframe = pd.read_hdf('route_seq')
            sequence_dataframe.LineID = sequence_dataframe.LineID.astype('str')
            sequence_dataframe.Direction = sequence_dataframe.Direction.astype('str')
            sequence_dataframe.Stop_ID = sequence_dataframe.Stop_ID.astype('str')
            unique_signiture = ['LineID', 'Direction', 'Destination']
            sequence_dataframe['Max_Stop_Sequence'] = sequence_dataframe.groupby(unique_signiture).Stop_Sequence.transform(max)
            excess_columns = ['Stop_name', 'Lat', 'Lon', 'Destination']
            sequence_dataframe = sequence_dataframe.drop(excess_columns, axis=1)
            return sequence_dataframe
 
        shared_columns = ['LineID', 'Stop_ID', 'Direction']
        self.df = pd.merge(self.df, get_sequence_dataframe(), how='inner', on=shared_columns)
    
    #------------------------------------------------------------------------------#
    
    def create_scheduled_speed_per_stop(self):
        '''creates scheduled speed which is the scheduled mean rate of stop traversal'''
        as_delta = pd.to_timedelta
        single_journeys = ['LineID', 'Vehicle_Journey_ID', 'Timeframe', 
                           'Vehicle_ID', 'Journey_Pattern_ID']
        self.df['Start_Stop'] = self.df.groupby(single_journeys)['Stop_Sequence'].transform(min)
        self.df['End_Stop'] = self.df.groupby(single_journeys)['Stop_Sequence'].transform(max)
        self.df['Stops_To_Travel'] = self.df.End_Stop.astype(int) - self.df.Start_Stop.astype(int)
        self.df = self.df.drop(['Start_Stop', 'End_Stop'], axis=1)
        self.df['Time_To_Travel'] = self.df.End_Time - self.df.Timestamp
        self.df.Time_To_Travel = as_delta(self.df.Time_To_Travel, unit='us').astype('timedelta64[m]')
        self.df['Scheduled_Speed_Per_Stop'] = self.df.Scheduled_Time_OP/self.df.Max_Stop_Sequence    
    
    #------------------------------------------------------------------------------#
    
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
        
    #------------------------------------------------------------------------------#
    
    def drop_non_modeling_columns(self):
        '''drops columns that can't be modeled'''
        useless = ['Timestamp',
                   'Timeframe',
                   'Time',
                   'End_Time',
                   'Start_Time',
                   'Max_Stop_Sequence',
                   'Journey_Pattern_ID', 
                   'Vehicle_Journey_ID', 
                   'Vehicle_ID',
                   'Stop_ID',
                   'Rain']
        self.df = self.df = self.df.drop(useless, axis=1)
    
    #------------------------------------------------------------------------------#
    def prepare(self):
        '''applies predefines preparation methods'''
        self.create_time_categories()  # time, day, hour columns
        self.create_journey_times() # end to end journey time
        self.drop_impossible_journey_times() # drops journey times == 0
        self.create_holiday() # creates category for school holidays
        self.create_scheduled_time_op() # creates scheduled travel time
        self.create_stop_sequence() # creates stop sequences for routes
        self.create_scheduled_speed_per_stop() # scheduled speed
        self.create_weather_columns() # creates weather columns
        self.drop_non_modeling_columns()
        return self.df

    #------------------------------------------------------------------------------#
    
class extracting:
    '''class for seperating data by route'''
    
    def __init__(self, files, month, columns):
        self.routes = set()
        self.month = month
        self.files = files
        self.columns = columns
    
    def get_all_routes(self):
        '''gets all routes from csvs'''
        for data in self.files:
            df = pd.read_hdf(data)
            self.routes = self.routes.union(set(df.LineID.unique()))
    
    def extract(self, route):
        '''extracts single route'''
        accumulator = pd.read_hdf(self.files[0])
        accumulator.columns = self.columns
        for data in self.files:
            df = pd.read_hdf(data)
            df = pd.DataFrame(df)
            #accumulator, df = accumulator.align(df, axis=1)
            accumulator = pd.concat([accumulator, df])
        return accumulator

#----------------------------------------------------------------------------------#

def setup(month):
        '''creates folders'''
        path_to_routes_folder = os.path.join(os.getcwd(), month + '_routes')
        os.makedirs(path_to_routes_folder, exist_ok=True)
        path_to_re_con = os.path.join(os.getcwd(), 're_con_' + month)
        os.makedirs(path_to_re_con, exist_ok=True)

#----------------------------------------------------------------------------------#

def re_construction(month):
    '''execustes cleaning and prepatation on raw data files'''
    path_to_folder = os.path.join(os.getcwd(), month + 'DayRawCopy')
    files = os.listdir(path_to_folder)
    for data_file in tqdm(files):
        file_address = os.path.join(path_to_folder, data_file)
        df = pd.read_csv(file_address)
        df = cleaning(df).clean()
        df = preparing(df).prepare()
        write_address = os.path.join(os.getcwd(), 're_con_' + month, data_file[:-4] + '.h5')
        df.to_hdf(write_address, key='moo', mode='w')

#----------------------------------------------------------------------------------#

def extraction(month):
    '''splits data by route'''
    path_to_folder = os.path.join(os.getcwd(), 're_con_'+month)
    files = os.listdir(path_to_folder)
    path_to_files = list(map(lambda data: os.path.join(path_to_folder, data), files))
    columns = pd.read_hdf(path_to_files[0]).columns
    data = extracting(path_to_files, month, columns)
    data.get_all_routes()
    for route in tqdm(data.routes):
        write_address = os.path.join(os.getcwd(), month+'_routes', 'xbeta'+str(route)+'.h5')
        df = data.extract(route)
        print(df.shape)
        df.to_hdf(write_address, key='moo', mode = 'w')

#----------------------------------------------------------------------------------#

def processing(month):
    #setup(month)
    #re_construction(month)
    extraction(month)
    return 'Finished'

def join_months(iterator):
    path = os.path.join(os.getcwd(), 'NovJan_routes')
    os.makedirs(path, exist_ok=True)
    Nov_path = os.path.join(os.getcwd(), 'Nov_routes')
    Jan_path = os.path.join(os.getcwd(), 'Jan_routes')
    contents = os.listdir(Nov_path)
    for file in iterator:
        if file != '.DS_Store':
            Nov_path_to_file = os.path.join(Nov_path, file)
            Nov_dataframe = pd.read_hdf(Nov_path_to_file)
            Jan_path_to_file = os.path.join(Jan_path, file)
            Jan_dataframe = pd.read_hdf(Jan_path_to_file)
            Jan_dataframe, Nov_dataframe = Jan_dataframe.align(Nov_dataframe, axis=1)
            Combined_dataframe = pd.concat([Jan_dataframe,Nov_dataframe])
            write_address = os.path.join(path,'x' + file)
            Combined_dataframe.to_hdf(write_address, key='moo', mode='w')

    return 'success'

if __name__ == '__main__':
    import multiprocessing
    from multiprocessing import Pool
        
    pool = Pool(processes=2)
    months=['Jan', 'Nov']
    result = pool.map(processing, months)
    pool.close()
    pool.join()
    print('Result: ', result)
    
#     
    
>>>>>>> reggie
