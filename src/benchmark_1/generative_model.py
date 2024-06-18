import random
import statistics
from datetime import datetime
from collections import Counter
from typing import Iterator, NamedTuple
from dataclasses import dataclass, field
from copy import deepcopy

import benchmark_1.config as config
from benchmark_1.new_DS import SimRequest, SimRequestChain
from Request_handler import Request_handler
from State import State

import pandas as pd

from benchmark_1.utilities import log_runtime_and_memory

random.seed(1)

class Memory: 
    '''as request come during simulation save them and when required
    apply some methods on it; optionaly start memory from dataframe of 
    multiple scheduled requests'''
    @log_runtime_and_memory
    def __init__(self, requests: pd.DataFrame, start_hour: int) -> None:
        self.start_hour = start_hour
        self.historic_requests: list[SimRequest] = []
        for request_index, request_row in requests.iterrows():
            #TODO alternatively sample from a pickle or make pandas dataframe operations on this
            request_pickup_time = ((((request_row["Requested Pickup Time"].hour - self.start_hour) * 60) + \
                                      request_row["Requested Pickup Time"].minute) * 60) + \
                                      request_row["Requested Pickup Time"].second

            # assert request_pickup_time >= 0, 'negative pickup time in the data for memory'
            
            if request_pickup_time >= 0:
                self.historic_requests.append(SimRequest(origin = request_row['Origin Node'], 
                                                        destination = request_row['Destination Node'], 
                                                        index = request_index, 
                                                        pickup_time= request_pickup_time,
                                                        pickup_times_timestamp= request_row['Requested Pickup Time'],
                                                        capacity = request_row['Number of Passengers']))


    def compute_mean_and_var(self) -> tuple[float, float]:
        '''based on requests seen so far set estimate how many requests per day
        have been made, model this as normal distribution; return mean and std
        of this distribution'''
        if self.historic_requests:
            daily_requests = Counter([historic_request.pickup_times_timestamp.day
                                    for historic_request in self.historic_requests]).values()
            if len(daily_requests) > 1:
                return statistics.mean(daily_requests), statistics.variance(daily_requests)
            else:
                return 1, 0
        else:
            raise ValueError('No data in the historic data set')
    
    def get_date_range(self) -> tuple[datetime, datetime]:
        '''use for debuggin to fine the range of the data'''
        return (min(self.historic_requests, key = lambda x: x.pickup_times_timestamp).pickup_times_timestamp,
                max(self.historic_requests, key = lambda x: x.pickup_times_timestamp).pickup_times_timestamp)
        
    def append(self, request_id: int, new_request: pd.Series) -> None:
        '''update Memory with new requests;
        !!! this is only going work if the date selected has hour larger than specified range'''

        #QUESTION DANIEL TODO how to define this for our applicaton?
        request_pickup_time = ((((new_request["Requested Pickup Time"].hour - self.start_hour) * 60) + \
                                  new_request["Requested Pickup Time"].minute) * 60) + \
                                  new_request["Requested Pickup Time"].second

        if request_pickup_time >= 0:

        # assert request_pickup_time >= 0, 'negative pickup time in the data for memory'
        
            self.historic_requests.append(SimRequest(new_request['Origin Node'], 
                                                    new_request['Destination Node'], 
                                                    request_id,  
                                                    pickup_time= request_pickup_time, 
                                                    pickup_times_timestamp= new_request['Requested Pickup Time']))
        
    def __iter__(self) -> Iterator[SimRequest]:
        '''to suppport working with counter'''
        if self.historic_requests:
            return iter(self.historic_requests)

@dataclass
class RequestsHistogram:
    '''frequency of requests in the historic data set'''

    def __init__(self, memory: Memory) -> None:
        counter = Counter(memory)
        self.requests = SimRequestChain(list(counter.keys()))
        self.weights = list(counter.values())
        self.rehashed_requests_capacities: dict[int, int] = {}
        self.rehashed_requests_pickup_times: dict[int, datetime] = {}

    def sample(self, request_number: int) -> SimRequestChain:
        '''sample a chain of requests of length n_chains 
        from from the histogram of the historic data requests;
        return only distinct values!'''
        sampled_chain = random.choices(self.requests.chain,
                                       weights=self.weights,
                                       k=request_number)
        
        lookup = set()
        for pos, index in enumerate(sampled_chain):
            if index not in lookup:
                lookup.add(index)
            else:
                rehashed_request = deepcopy(sampled_chain[pos])
                rehashed_request.index = abs(hash(datetime.now()))
                sampled_chain[pos] = rehashed_request
                self.rehashed_requests_capacities[rehashed_request.index] = rehashed_request.capacity
                self.rehashed_requests_pickup_times[rehashed_request.index] = rehashed_request.pickup_time #TODO should this be a timestamp?
        return SimRequestChain(chain=sampled_chain)
        
@dataclass
class RequestBank:
    '''bank or sampled request chains'''
    values: list[SimRequestChain] = field(default_factory=list)

    @log_runtime_and_memory
    def update(self, memory: Memory) -> tuple[dict[int,int], dict[int, datetime]]:
        '''boostrap from histogram data many requests,
        computed offline sampled online during runtime'''
        mu, var = memory.compute_mean_and_var()
        histogram = RequestsHistogram(memory)
        sampled_gaussian = int(random.gauss(mu, var**0.5))
        num_requests = min(sampled_gaussian, config.MCTS_DEPTH)
        self.values = [histogram.sample(num_requests)
                            for _ in range(config.SAMPLED_BANK_SIZE)]
        return (histogram.rehashed_requests_capacities, 
                histogram.rehashed_requests_pickup_times)

    def sample(self) -> SimRequestChain:
        '''sample a request chain from the bank'''
        return random.choice(self.values)

class GenerativeModel:
    '''build offline bank of bootstrapped requests from dataset;
    each request chain should have the length estimated by the normal
    distribution computed based on the available dataset;
    create this from initial_requests and minute requests and recompute on 
    the minutes intervals'''

    #TODO parallelize creation of this module
    @log_runtime_and_memory
    def __init__(self, state_object: State, request_handler: Request_handler) -> None:
        reqh = request_handler
        merged_requests = reqh.get_requests_before_given_date(state_object.date_operational_range.year, 
                                                              state_object.date_operational_range.month, 
                                                              state_object.date_operational_range.day)
        self._historic_data: Memory = Memory(merged_requests, state_object.date_operational_range.start_hour)
        self._requests_bank: RequestBank = RequestBank()
        rehashed_capactiites, reshashed_pickups = self._requests_bank.update(self._historic_data)
        self.historic_requests_capacities: dict[int, int] = {request.index: request.capacity for request in self._historic_data} | rehashed_capactiites
        self.historic_requests_pickup_times: dict[int, datetime] = {request.index: request.pickup_time for request in self._historic_data} | reshashed_pickups
        assert any(value > 0 for value in self.historic_requests_pickup_times.values()), 'negative pickup values in the data for generative model'
    
    def sample(self) -> SimRequestChain:
        '''public method used to online sample requests from offline
        computed bank of chains of requests'''
        return self._requests_bank.sample()
    
    def check_memory_range(self) -> tuple[datetime, datetime]:
        '''debugging method to check the range of the data'''
        return self._historic_data.get_date_range()
    