#!/usr/bin/env python
"""Weather_data_pull.py

This script downloads weather data for the location of interest for a given time range

Example:
    Default usage:
        $ python3 Weather_data_pull.py

"""
import os
import json
import requests
import datetime
import traceback

class Weather_repo:
    def __init__(self, fields=None, weather_url_api=None, data_folder="data/weather/"):
        if weather_url_api is None:
            self.weather_url_api = 'https://archive-api.open-meteo.com/v1/archive'
        else:
            self.weather_url_api = weather_url_api

        if fields is None:
            self.weather_fields = ['temperature_2m', 'precipitation', 'windspeed_10m']
        else:
            self.weather_fields = fields
        if not os.path.isdir(data_folder):
            os.mkdir(data_folder)
        self.data_folder = data_folder

    def populate_request_params(self, latitude=42.3614251, longitude=-71.1283633, start_date="2023-02-02", end_date="2023-02-03"):
        
        request_params ={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": self.weather_fields
        }
        return request_params

    def get_weather_data(self, location="test", latitude=42.3614251, longitude=-71.1283633, start_date="2022-01-01", end_date="2023-07-31"):
        request_params = self.populate_request_params(latitude=latitude, longitude=longitude, start_date=start_date, end_date=end_date)
        try:
            response = requests.get(self.weather_url_api, params=request_params, timeout=100)

            file_name = self.data_folder + location + "_" + start_date + "_" + end_date + ".json"

            if response.status_code != 200:
                print(response)
                print(f'Error: {response.content}')
            else:
                content = response.json()
                json_object = json.dumps(content)

                with open(file_name, mode='w') as file:
                    writer = file.write(json_object)
                    
                print(f'Data for {location} is stored in {file_name}')
        
        except Exception as e:
            print(e)


if __name__ == '__main__':
    """Performs execution delta of the process."""
    # Unit tests
    pStart = datetime.datetime.now()
    try:
        Weather_repo().get_weather_data(location="Allston", 
                                        latitude=42.3614251, 
                                        longitude=-71.1283633, 
                                        start_date="2022-01-01", 
                                        end_date="2023-07-31")
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart))