"""NN_utils.py

This script instantiates the demand prediction models, trains them, and evaluates them using the held out test set. 
All hyperparameters must be specified in the main execution branch of the script at the end of these file.

Example:
    Default usage:
        $ python3 NN_utils.py

"""
import os
import ast
import math
import json
import torch
import pickle
import datetime
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Data_structures import Data_folders, Date_operational_range


class Request_Prediction_Handler:

    def __init__(self, data_folders: Data_folders, perfect_accuracy=True):
        self.status_of_interest = ["Completed", "Unaccepted Proposal", "Cancel"]
        self.processed_requests_folder_path = data_folders.processed_requests_folder_path
        self.predicted_requests_folder_path = data_folders.predicted_requests_folder_path

        self.sched_reqs_df, self.pred_online_reqs_df = self._load_requests_dataframes(perfect_accuracy=perfect_accuracy)
        self._correct_datetime_for_dataframes()
        if perfect_accuracy:
            self._filter_requests_dataframes()
        self.month_lengths = {2022: [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                              2023: [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]}
        
        #self._expand_dataframes()

    def _load_requests_dataframes(self, perfect_accuracy=False):
        scheduled_requests_path = os.path.join(self.processed_requests_folder_path, "scheduled_requests.csv")
        scheduled_requests_df = pd.read_csv(scheduled_requests_path)

        if perfect_accuracy:
            true_online_requests_path = os.path.join(self.processed_requests_folder_path, "online_requests.csv")
            predicted_online_requests_df = pd.read_csv(true_online_requests_path)
        else:
            predicted_online_requests_path = os.path.join(self.predicted_requests_folder_path, "online_requests.csv")
            predicted_online_requests_df = pd.read_csv(predicted_online_requests_path)
        
        return scheduled_requests_df, predicted_online_requests_df
    
    def _correct_datetime_for_dataframes(self):
        self.sched_reqs_df["Requested Pickup Time"] = pd.to_datetime(self.sched_reqs_df["Requested Pickup Time"])
        self.pred_online_reqs_df["Requested Pickup Time"] = pd.to_datetime(self.pred_online_reqs_df["Requested Pickup Time"])
    
    def _filter_requests_dataframes(self):
        sched_requests_mask = self.sched_reqs_df["Request Status"].isin(self.status_of_interest)
        online_requests_maks = self.pred_online_reqs_df["Request Status"].isin(self.status_of_interest)

        self.sched_reqs_df = self.sched_reqs_df[sched_requests_mask]
        self.pred_online_reqs_df = self.pred_online_reqs_df[online_requests_maks]
    
    def _expand_dataframes(self):
        sched_df = self.sched_reqs_df
        online_df = self.pred_online_reqs_df
        self.sched_reqs_df = sched_df.loc[sched_df.index.repeat(sched_df['Number of Passengers'])].reset_index(drop=True)
        self.pred_online_reqs_df = online_df.loc[online_df.index.repeat(online_df['Number of Passengers'])].reset_index(drop=True)
        self.sched_reqs_df["Number of Passengers"] = 1
        self.pred_online_reqs_df["Number of Passengers"] = 1
    
    def get_requests_for_given_date_and_hour_range(self, date_operational_range: Date_operational_range):
        if date_operational_range.end_hour < date_operational_range.start_hour:
            current_date_string = str(date_operational_range.year)+"-"+str(date_operational_range.month)+"-"+str(date_operational_range.day)
            current_date_object = pd.to_datetime(current_date_string).date()

            current_scheduled_mask = (self.sched_reqs_df["Requested Pickup Time"].dt.date == current_date_object) \
                & (self.sched_reqs_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.sched_reqs_df["Requested Pickup Time"].dt.hour <= 23)
            
            current_online_mask = (self.pred_online_reqs_df["Requested Pickup Time"].dt.date == current_date_object) \
                & (self.pred_online_reqs_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.pred_online_reqs_df["Requested Pickup Time"].dt.hour <= 23)
            
            if date_operational_range.day+1 > self.month_lengths[date_operational_range.year][date_operational_range.month-1]:
                following_day = 1
                if date_operational_range.month + 1 > 12:
                    following_month = 1
                    following_year = date_operational_range.year + 1
                else:
                    following_month = date_operational_range.month + 1
                    following_year = date_operational_range.year
            else:
                following_day = date_operational_range.day + 1
                following_month = date_operational_range.month
                following_year = date_operational_range.year
            following_date_string = str(following_year)+"-"+str(following_month)+"-"+str(following_day)
            following_date_object = pd.to_datetime(following_date_string).date()

            following_scheduled_mask = (self.sched_reqs_df["Requested Pickup Time"].dt.date == following_date_object) \
                & (self.sched_reqs_df["Requested Pickup Time"].dt.hour >= 0) \
                & (self.sched_reqs_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            following_online_mask = (self.pred_online_reqs_df["Requested Pickup Time"].dt.date == following_date_object) \
                & (self.pred_online_reqs_df["Requested Pickup Time"].dt.hour >= 0) \
                & (self.pred_online_reqs_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            
            scheduled_requests_df = self.sched_reqs_df[current_scheduled_mask | following_scheduled_mask]
            scheduled_requests_df = scheduled_requests_df.sort_values(by=["Requested Pickup Time"])
            online_requests_df = self.pred_online_reqs_df[current_online_mask | following_online_mask]
            online_requests_df = online_requests_df.sort_values(by=["Requested Pickup Time"])
        else:
            date_string = str(date_operational_range.year)+"-"+str(date_operational_range.month)+"-"+str(date_operational_range.day)
            date_object = pd.to_datetime(date_string).date()

            scheduled_mask = (self.sched_reqs_df["Requested Pickup Time"].dt.date == date_object) \
                & (self.sched_reqs_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.sched_reqs_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            online_mask = (self.pred_online_reqs_df["Requested Pickup Time"].dt.date == date_object) \
                & (self.pred_online_reqs_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.pred_online_reqs_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            scheduled_requests_df = self.sched_reqs_df[scheduled_mask]
            scheduled_requests_df = scheduled_requests_df.sort_values(by=["Requested Pickup Time"])
            online_requests_df = self.pred_online_reqs_df[online_mask]
            online_requests_df = online_requests_df.sort_values(by=["Requested Pickup Time"])
        
        return scheduled_requests_df, online_requests_df
    
    def get_requests_for_given_minute_range(self, year: int, month: int, day: int, hour: int, 
                                            start_minute: int, end_minute: int):
        date_string = str(year)+"-"+str(month)+"-"+str(day)
        date_object = pd.to_datetime(date_string).date()

        scheduled_mask = (self.sched_reqs_df["Requested Pickup Time"].dt.date == date_object) \
            & (self.sched_reqs_df["Requested Pickup Time"].dt.hour == hour) \
            & (self.sched_reqs_df["Requested Pickup Time"].dt.minute >= start_minute) \
            & (self.sched_reqs_df["Requested Pickup Time"].dt.minute <= end_minute)
        
        online_mask = (self.pred_online_reqs_df["Requested Pickup Time"].dt.date == date_object) \
            & (self.pred_online_reqs_df["Requested Pickup Time"].dt.hour == hour) \
            & (self.pred_online_reqs_df["Requested Pickup Time"].dt.minute >= start_minute) \
            & (self.pred_online_reqs_df["Requested Pickup Time"].dt.minute <= end_minute)
        
        return self.sched_reqs_df[scheduled_mask], self.pred_online_reqs_df[online_mask]
    
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

class Requests_predictor:

    def __init__(self):
        pass

    def compile_and_train(self):
        pass

    def calculate_error_metrics(values, predictions):
        result_metrics = {'mae' : mean_absolute_error(values, predictions),
                        'rmse' : mean_squared_error(values, predictions) ** 0.5,
                        'r2' : r2_score(values, predictions)}
        
        print("Mean Absolute Error:       ", result_metrics["mae"])
        print("Root Mean Squared Error:   ", result_metrics["rmse"])
        print("R^2 Score:                 ", result_metrics["r2"])
        return result_metrics
    
    def _save_request_predictions(self):
        pass

class Request_Data_Manager:

    def __init__(self, data_folders: Data_folders, preprocess = False,
                 weather_output_file="weather_features.csv"):
        self.data_folders = data_folders
        
        self.weather_fields = [ 'temperature_2m', 'precipitation', 'windspeed_10m']

        if preprocess:
            print("Preprocessing Feature Data ...")
            self._preprocess_weather_data(output_filename=weather_output_file)
    
    def _preprocess_ride_data(self):
        pass

    def _preprocess_weather_data(self, month_label="month", day_label="day", hour_label="hour", year_label="year"):
        weather_values = {}
        for field in self.weather_fields:
            weather_values[field] = []
        
        weather_values[month_label] = []
        weather_values[day_label] = []
        weather_values[hour_label] = []
        weather_values[year_label] = []
        
        for filename in sorted(os.listdir(self.weather_folder)):

            filepath= os.path.join(self.weather_folder, filename)
            with open(filepath, "r") as file_handle:
                weather_data = file_handle.read()

            weather_object = json.loads(weather_data)
            for field in self.weather_fields:
                weather_values[field] += weather_object["hourly"][field]
            
            for time_stamp in weather_object["hourly"]["time"]:
                date_time_components = time_stamp.split("-")
                month = date_time_components[1]
                year = date_time_components[0]
                time_components = date_time_components[2].split("T")
                hour_components = time_components[1].split(":")

                weather_values[month_label].append(int(month))
                weather_values[year_label].append(int(year))
                weather_values[day_label].append(int(time_components[0]))
                weather_values[hour_label].append(int(hour_components[0]))
        



if __name__ == '__main__':
    """Performs execution delta of the process."""
    # Unit tests
    pStart = datetime.datetime.now()
    try:
        request_pred = Requests_predictor()

    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart))  
