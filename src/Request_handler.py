import optparse
import os
import datetime
import traceback
import osmnx as ox
import numpy as np
import pandas as pd

from Data_structures import Data_folders, Date_operational_range, Completed_requests_stats, Failed_requests_stats
from benchmark_1.utilities import log_runtime_and_memory

class Request_handler:

    def __init__(self, data_folders: Data_folders, process_requests: bool):
        self.request_folder_path = data_folders.request_folder_path
        self.routing_data_folder = data_folders.routing_data_folder
        self.processed_requests_folder_path = data_folders.processed_requests_folder_path

        self.columns_of_interest = ["Request Creation Date", "Request Creation Time", "Number of Passengers", "Booking Type", 
                                    "Requested Pickup Time", "Origin Lat", "Origin Lng", "Destination Lat", "Destination Lng", 
                                    "Request Status", "On-demand ETA", "Ride Duration"]
        self.status_of_interest = ["Completed", "Unaccepted Proposal", "Cancel"]
        self.G = self._initialize_map()
        self.month_lengths = {2022: [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                              2023: [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]}

        if process_requests:
            self.scheduled_requests_df, self.online_requests_df = self._initialize_dataframes()
            self._save_requests_dataframes()
        else:
            self.scheduled_requests_df, self.online_requests_df = self._load_requests_dataframes()
            self._correct_datetime_for_dataframes()
        
        self._filter_requests_dataframes()
        
        #self._expand_dataframes()

    def _initialize_map(self):
        graphml_filepath = os.path.join(self.routing_data_folder, "graph_structure.graphml")
        G = ox.load_graphml(graphml_filepath)

        return G
    
    def _load_requests_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        scheduled_requests_path = os.path.join(self.processed_requests_folder_path, "scheduled_requests.csv")
        scheduled_requests_df = pd.read_csv(scheduled_requests_path)

        online_requests_path = os.path.join(self.processed_requests_folder_path, "online_requests.csv")
        online_requests_df = pd.read_csv(online_requests_path)
        
        return scheduled_requests_df, online_requests_df
    
    def _correct_datetime_for_dataframes(self):
        self.scheduled_requests_df["Request Creation Time"] = pd.to_datetime(self.scheduled_requests_df["Request Creation Time"])
        self.scheduled_requests_df["Requested Pickup Time"] = pd.to_datetime(self.scheduled_requests_df["Requested Pickup Time"])
        self.online_requests_df["Request Creation Time"] = pd.to_datetime(self.online_requests_df["Request Creation Time"])
        self.online_requests_df["Requested Pickup Time"] = pd.to_datetime(self.online_requests_df["Requested Pickup Time"])
    
    def _filter_requests_dataframes(self):
        sched_requests_mask = self.scheduled_requests_df["Request Status"].isin(self.status_of_interest)
        online_requests_maks = self.online_requests_df["Request Status"].isin(self.status_of_interest)

        self.scheduled_requests_df = self.scheduled_requests_df[sched_requests_mask]
        self.online_requests_df = self.online_requests_df[online_requests_maks]

    def _save_requests_dataframes(self):
        scheduled_requests_path = os.path.join(self.processed_requests_folder_path, "scheduled_requests.csv")
        self.scheduled_requests_df.to_csv(scheduled_requests_path, index=True)
        
        online_requests_path = os.path.join(self.processed_requests_folder_path, "online_requests.csv")
        self.online_requests_df.to_csv(online_requests_path, index=True)
    
    def _initialize_dataframes(self):
        combined_requests_df = self._read_xlsx_files_into_data_frame()
        scheduled_requests_df, online_requests_df = self._divide_requests_dataframe(dataframe=combined_requests_df)
        processed_scheduled_requests_df = self._process_requests_dataframe(dataframe=scheduled_requests_df)
        processed_online_requests_df = self._process_requests_dataframe(dataframe=online_requests_df)

        return processed_scheduled_requests_df, processed_online_requests_df
    
    def _expand_dataframes(self):
        sched_df = self.scheduled_requests_df
        online_df = self.online_requests_df
        self.scheduled_requests_df = sched_df.loc[sched_df.index.repeat(sched_df['Number of Passengers'])].reset_index(drop=True)
        self.online_requests_df = online_df.loc[online_df.index.repeat(online_df['Number of Passengers'])].reset_index(drop=True)
        self.scheduled_requests_df["Number of Passengers"] = 1
        self.online_requests_df["Number of Passengers"] = 1

    def _read_xlsx_files_into_data_frame(self):
        dataframes_list = []
        for filename in os.listdir(self.request_folder_path):
            requests_filepath = os.path.join(self.request_folder_path, filename)
            df = pd.read_excel(requests_filepath)
            dataframes_list.append(df)
        
        combined_requests_df = pd.concat(dataframes_list)
        combined_requests_df.reset_index(drop=True, inplace=True)

        return combined_requests_df
    
    def _divide_requests_dataframe(self, dataframe):
        filtered_requests_df = dataframe[self.columns_of_interest]
        scheduled_requests_df = filtered_requests_df[(filtered_requests_df["Booking Type"] == "Prebooking")]
        online_requests_df = filtered_requests_df[(filtered_requests_df["Booking Type"] == "On Demand")]

        return scheduled_requests_df, online_requests_df 
    
    def _get_nodes_from_coordinates(self, dataframe):
        origin_nodes = ox.nearest_nodes(self.G, dataframe["Origin Lng"], dataframe["Origin Lat"])
        destination_nodes = ox.nearest_nodes(self.G, dataframe["Destination Lng"], dataframe["Destination Lat"])

        return origin_nodes, destination_nodes
    
    def _process_requests_dataframe(self, dataframe):
        origin_nodes, destination_nodes = self._get_nodes_from_coordinates(dataframe=dataframe)
        processed_requests_df = dataframe.copy()
        processed_requests_df["Origin Node"] = origin_nodes
        processed_requests_df["Destination Node"] = destination_nodes

        return processed_requests_df
    
    
    def get_requests_for_given_date_and_hour_range(self, date_operational_range: Date_operational_range):
        if date_operational_range.end_hour < date_operational_range.start_hour:
            current_date_string = str(date_operational_range.year)+"-"+str(date_operational_range.month)+"-"+str(date_operational_range.day)
            current_date_object = pd.to_datetime(current_date_string).date()

            current_scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == current_date_object) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= 23)
            
            current_online_mask = (self.online_requests_df["Requested Pickup Time"].dt.date == current_date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= 23)
            
            if date_operational_range.day+1 > self.month_lengths[date_operational_range.year][date_operational_range.month-1]:
                following_day = 1
                if date_operational_range.month + 1 > 12:
                    following_month = 1
                    following_year = date_operational_range.year + 1
                else:
                    following_month = date_operational_range.month + 1
                    following_year = date_operational_range.year
            else:
                following_day = date_operational_range.day+1
                following_month = date_operational_range.month
                following_year = date_operational_range.year
            following_date_string = str(following_year)+"-"+str(following_month)+"-"+str(following_day)
            following_date_object = pd.to_datetime(following_date_string).date()

            following_scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == following_date_object) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= 0) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            following_online_mask = (self.online_requests_df["Requested Pickup Time"].dt.date == following_date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= 0) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            
            scheduled_requests_df = self.scheduled_requests_df[current_scheduled_mask | following_scheduled_mask]
            scheduled_requests_df = scheduled_requests_df.sort_values(by=["Requested Pickup Time"])
            online_requests_df = self.online_requests_df[current_online_mask | following_online_mask]
            online_requests_df = online_requests_df.sort_values(by=["Requested Pickup Time"])
        else:
            date_string = str(date_operational_range.year)+"-"+str(date_operational_range.month)+"-"+str(date_operational_range.day)
            date_object = pd.to_datetime(date_string).date()

            scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == date_object) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            online_mask = (self.online_requests_df["Requested Pickup Time"].dt.date == date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            scheduled_requests_df = self.scheduled_requests_df[scheduled_mask]
            scheduled_requests_df = scheduled_requests_df.sort_values(by=["Requested Pickup Time"])
            online_requests_df = self.online_requests_df[online_mask]
            online_requests_df = online_requests_df.sort_values(by=["Requested Pickup Time"])
        
        return scheduled_requests_df, online_requests_df
    
    def get_requests_for_given_minute_range(self, year: int, month: int, day: int, hour: int, 
                                            start_minute: int, end_minute: int):
        date_string = str(year)+"-"+str(month)+"-"+str(day)
        date_object = pd.to_datetime(date_string).date()

        scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == date_object) \
            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour == hour) \
            & (self.scheduled_requests_df["Requested Pickup Time"].dt.minute >= start_minute) \
            & (self.scheduled_requests_df["Requested Pickup Time"].dt.minute <= end_minute)
        
        online_mask = (self.online_requests_df["Requested Pickup Time"].dt.date == date_object) \
            & (self.online_requests_df["Requested Pickup Time"].dt.hour == hour) \
            & (self.online_requests_df["Requested Pickup Time"].dt.minute >= start_minute) \
            & (self.online_requests_df["Requested Pickup Time"].dt.minute <= end_minute)
        
        return self.scheduled_requests_df[scheduled_mask], self.online_requests_df[online_mask]
    
    @log_runtime_and_memory
    def get_requests_before_given_date(self, year: int, month: int, day: int) -> pd.DataFrame:
        '''get all historical values that happened before start of our system 
        return them as a merged values'''
        date_string = str(year)+"-"+str(month)+"-"+str(day)
        date_object = pd.to_datetime(date_string).date()

        scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date < date_object)
        online_mask = (self.online_requests_df["Requested Pickup Time"].dt.date < date_object)
        
        scheduled_requests = self.scheduled_requests_df[scheduled_mask]
        online_requests = self.online_requests_df[online_mask]
        
        merged_requests = pd.concat([scheduled_requests, online_requests])
        
        return merged_requests
    
    def get_online_requests_for_given_minute(self, date_operational_range: Date_operational_range, year: int, 
                                             month: int, day: int, hour: int, minute: int):
        date_string = str(year)+"-"+str(month)+"-"+str(day)
        date_object = pd.to_datetime(date_string).date()

        if date_operational_range.end_hour < date_operational_range.start_hour:
            if hour < date_operational_range.end_hour:
                hour_boundary = date_operational_range.end_hour
            else:
                hour_boundary = 23
        else:
            hour_boundary = date_operational_range.end_hour
        
        online_mask = (self.online_requests_df["Request Creation Time"].dt.date == date_object) \
            & (self.online_requests_df["Request Creation Time"].dt.hour == hour) \
            & (self.online_requests_df["Request Creation Time"].dt.minute == minute) \
            & (self.online_requests_df["Requested Pickup Time"].dt.date == date_object) \
            & (self.online_requests_df["Requested Pickup Time"].dt.hour <= hour_boundary)
        
        online_requests_df = self.online_requests_df[online_mask]
        
        return online_requests_df
    
    def get_initial_requests(self, date_operational_range: Date_operational_range):
        if date_operational_range.end_hour < date_operational_range.start_hour:
            current_date_string = str(date_operational_range.year)+"-"+str(date_operational_range.month)+"-"+str(date_operational_range.day)
            current_date_object = pd.to_datetime(current_date_string).date()

            if date_operational_range.day+1 > self.month_lengths[date_operational_range.year][date_operational_range.month-1]:
                following_day = 1
                if date_operational_range.month + 1 > 12:
                    following_month = 1
                    following_year = date_operational_range.year + 1
                else:
                    following_month = date_operational_range.month + 1
                    following_year = date_operational_range.year
            else:
                following_day = date_operational_range.day+1
                following_month = date_operational_range.month
                following_year = date_operational_range.year
            following_date_string = str(following_year)+"-"+str(following_month)+"-"+str(following_day)
            following_date_object = pd.to_datetime(following_date_string).date()

            current_scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == current_date_object) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= 23)
            
            current_online_mask = (self.online_requests_df["Request Creation Time"].dt.date == current_date_object) \
                & (self.online_requests_df["Request Creation Time"].dt.hour < date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.date == current_date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= 23)
            
            following_scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == following_date_object) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= 0) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            following_online_mask = (self.online_requests_df["Request Creation Time"].dt.date == current_date_object) \
                & (self.online_requests_df["Request Creation Time"].dt.hour < date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.date == following_date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= 0) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            online_requests_df = self.online_requests_df[current_online_mask | following_online_mask]
            scheduled_requests_df = self.scheduled_requests_df[current_scheduled_mask | following_scheduled_mask]
            
            combined_requests_df = pd.concat([scheduled_requests_df, online_requests_df])
        
        else:
            date_string = str(date_operational_range.year)+"-"+str(date_operational_range.month)+"-"+str(date_operational_range.day)
            date_object = pd.to_datetime(date_string).date()

            scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == date_object) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            online_mask = (self.online_requests_df["Request Creation Time"].dt.date == date_object) \
                & (self.online_requests_df["Request Creation Time"].dt.hour < date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.date == date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            online_requests_df = self.online_requests_df[online_mask]
            scheduled_requests_df = self.scheduled_requests_df[scheduled_mask]
            
            combined_requests_df = pd.concat([scheduled_requests_df, online_requests_df])
        
        return combined_requests_df
    
    def generate_operating_ranges(self, date_operational_range: Date_operational_range):
        if date_operational_range.end_hour < date_operational_range.start_hour:
            current_hour_range = list(range(date_operational_range.start_hour, 24))
            first_day_range = [date_operational_range.day] * len(current_hour_range)
            first_month_range = [date_operational_range.month] * len(current_hour_range)
            first_year_range = [date_operational_range.year] * len(current_hour_range)

            following_hour_range = list(range(0, date_operational_range.end_hour+1))
            hour_range = current_hour_range + following_hour_range

            if date_operational_range.day+1 > self.month_lengths[date_operational_range.year][date_operational_range.month-1]:
                if date_operational_range.month + 1 > 12:
                    second_day_range = [1] * len(following_hour_range)
                    second_month_range = [1] * len(following_hour_range)
                    second_year_range = [date_operational_range.year + 1] * len(following_hour_range)
                else:
                    second_day_range = [1] * len(following_hour_range)
                    second_month_range = [date_operational_range.month + 1] * len(following_hour_range)
                    second_year_range = [date_operational_range.year] * len(following_hour_range)
            else:
                second_day_range = [date_operational_range.day + 1] * len(following_hour_range)
                second_month_range = [date_operational_range.month] * len(following_hour_range)
                second_year_range = [date_operational_range.year] * len(following_hour_range)
            day_range = first_day_range + second_day_range
            month_range = first_month_range + second_month_range
            year_range = first_year_range + second_year_range
        else:
            hour_range = list(range(date_operational_range.start_hour, date_operational_range.end_hour+1))
            day_range = [date_operational_range.day] * len(hour_range)
            month_range = [date_operational_range.month] * len(hour_range)
            year_range = [date_operational_range.year] * len(hour_range)

        
        return hour_range, day_range, month_range, year_range
    
    def extract_data_statistics_for_given_date(self, date_operational_range: Date_operational_range):
        scheduled_requests_df, online_requests_df = self.get_requests_for_given_date_and_hour_range(date_operational_range=date_operational_range)
        combined_requests_df = pd.concat([scheduled_requests_df, online_requests_df])

        print("Request info: " + str(len(combined_requests_df.index)))

        completed_requests_stats = Completed_requests_stats(combined_requests_df=combined_requests_df)
        failed_requests_stats = Failed_requests_stats(combined_requests_df=combined_requests_df)

        return completed_requests_stats, failed_requests_stats
    
    def extract_metadata(self):
        combined_requests_df = pd.concat([self.scheduled_requests_df, self.online_requests_df])

        unique_count = combined_requests_df["Request Creation Date"].nunique()

        return unique_count
        
 
if __name__ == '__main__':
    """Performs execution delta of the process."""
    # Unit tests
    pStart = datetime.datetime.now()
    try:
        parser = optparse.OptionParser()
        parser.add_option("--year", type=int, dest='year', default=2022, help='Select year of interest.')
        parser.add_option("--month", type=int, dest='month', default=8, help='Select month of interest.')
        parser.add_option("--day", type=int, dest='day', default=17, help='Select day of interest.')
        parser.add_option("--start_hour", type=int, dest='start_hour', default=19, help='Select start hour for the time range of interest') #19
        parser.add_option("--end_hour", type=int, dest='end_hour', default=2, help='Select end hour for the time range of interest')
        (options, args) = parser.parse_args()
        
        data_folders = Data_folders()
        date_operational_range = Date_operational_range(year=options.year, 
                                                    month=options.month,
                                                    day=options.day,
                                                    start_hour=options.start_hour,
                                                    end_hour=options.end_hour)

        rqh = Request_handler(data_folders=data_folders, process_requests=True)
        number_of_days_in_the_data = rqh.extract_metadata()
        print("Number of days in the data loaded = " + str(number_of_days_in_the_data))
        completed_requests_stats, failed_requests_stats = rqh.extract_data_statistics_for_given_date(date_operational_range=date_operational_range)

        print(completed_requests_stats)
        print(failed_requests_stats)

        failed_requests_avg_wait_time = failed_requests_stats.avg_wait_time_for_failed_requests

        count_of_potentially_failed_requests = 0
        for wait_time_value in failed_requests_stats.wait_times_for_canceled_requests:
            if wait_time_value > failed_requests_avg_wait_time:
                count_of_potentially_failed_requests += 1
        
        for wait_time_value in completed_requests_stats.passenger_wait_times:
            if wait_time_value > failed_requests_avg_wait_time:
                count_of_potentially_failed_requests += 1

        print("Number of requests with higher wait times than the average wait time of real cancelled requests = " + str(count_of_potentially_failed_requests))
        

    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart))

        
