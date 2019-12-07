from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv


SECONDS = 600    #600 for trips

"""
=========================================
only need to change folder, filedate and
 filter_function to customize each run
=========================================
"""


folder = 'Feb_2017_DATA'
filedate = '20170208'


#data_file ="/Users/Lijiayi/Documents/CALPATH/PenetrationAnalysis/Feb_2017_DATA/data_raw/probe_data_I210.2017test.waynep1.csv"
#data_file2 = "/Users/Lijiayi/Documents/CALPATH/PenetrationAnalysis/" + folder + "/CONSUMER_test_dataset.waynep1.csv"
#data_file3 = "/Users/Lijiayi/Documents/CALPATH/PenetrationAnalysis/" + folder + "/FLEET_test_dataset.waynep1.csv"

directory = "/Users/Lijiayi/Documents/CALPATH/PenetrationAnalysis/"


data_file = directory + folder + "/data_raw/probe_data_I210." + filedate + ".waynep1.csv"
data_file1 = directory + folder + "/output_probe_data_I210." + filedate + ".waynep1.csv"
data_file2 = directory + folder + "/output_probe_data_I210." + filedate + "CONSUMER" + ".waynep1.csv"
data_file3 = directory + folder + "/output_probe_data_I210." + filedate + "FLEET" + ".waynep1.csv"
#data_file3 = "/Users/Lijiayi/Documents/CALPATH/PenetrationAnalysis/" + folder + "/plots_I210." + filedate + ".waynep1_trip.csv"



"""
to analyze different
categories of data
"""
empty_filter = lambda id, provider, deltatime: provider[0:5] =='CONSU' or provider[0:5]=='FLEET'
filter_func = lambda id, provider, deltatime: provider[0:5] == 'CONSU' #and deltatime < timedelta(i) and deltatime >= timedelta(i-1)
filter_func1 = lambda id, provider, deltatime: provider[0:5] == 'FLEET'

interested_provider = 'FLEET40'  # ****

all_probes = {}

AllTrips = namedtuple('AllTrips', ['provider', 'timestamps', 'trips'])
Trip = namedtuple('Trip', ['sample_rate', 'trip', 'deltas'])


all_probes = {} # probe_id (key), AllTrips (value)
with open(data_file, 'r') as dfile:
    print('Loading files [', end='', flush=True)
    reader = csv.DictReader(dfile)
    progress = 0
    for row in reader:
        id = row['PROBE_ID']
        timestamp = datetime.strptime(row['SAMPLE_DATE'], '%Y-%m-%d %H:%M:%S')
        if id not in all_probes:
            provider = row['PROBE_DATA_PROVIDER']
            all_probes[id] = AllTrips(provider, [timestamp], [])  # ****
        else:
            all_probes[id].timestamps.append(timestamp)
        if progress % 10000 == 0:
            print('=', end='', flush=True)
        if progress > 10000000:
            break
        progress += 1
    print('Done!')

for id, all_trips in all_probes.items():
    all_trips.timestamps.sort()  # for each probe_id, sort all the timestamps
    templist = []  # store each distinct series of records(trip)
    segmentation = []
    for time in all_trips.timestamps:  # iterate through sorted list of probe_id,
        if not templist:  # if the templist is empty right now, just add current timestamp to it
            templist.append(time)
        else:
            tdelta = (time - templist[
                -1]).total_seconds()  # take time difference (expressed in seconds) to see whether we need to restart a templist
            if (tdelta < SECONDS):
                templist.append(time)
            else:
                avg_rate = 0
                deltas = []
                if len(templist) > 1:
                    avg_rate = (templist[-1] - templist[0]) / (len(templist) - 1)
                    prev = templist[0]
                    for curr in templist[1:]:
                        deltas.append(curr - prev)
                        prev = curr
                all_trips.trips.append(Trip(avg_rate, templist, deltas))

                templist = []
                templist.append(time)

    avg_rate = 0
    deltas = []
    if len(templist) > 1:
        avg_rate = (templist[-1] - templist[0]) / (len(templist) - 1)
        prev = templist[0]
        for curr in templist[1:]:
            deltas.append(curr - prev)
            prev = curr
    all_trips.trips.append(Trip(avg_rate, templist, deltas))


def median_rates(starttime=0, endtime=24, filter=None):
    medians = {}   # with trips
    all_med = []   #without trips

    for probe_id, all_trips in all_probes.items():
        probe_med = []
        for trip in all_trips.trips:
            for timestamp, deltatime in zip(trip.trip, trip.deltas):
                if filter is not None:
                    satisfied = filter(probe_id, all_trips.provider, deltatime)  # boolean
                    if not satisfied:
                        continue
                if timestamp.hour < starttime or timestamp.hour > endtime:
                    continue
                if type(deltatime) == timedelta:
                    probe_med.append(deltatime)
                    all_med.append(deltatime)
                else:
                    all_med.append(timedelta(deltatime))
                    probe_med.append(timedelta(deltatime))  # sample rate for each trip
        if len(probe_med) != 0:
            probe_med.sort()
            medians[probe_id] = (probe_med[int(len(probe_med) / 2)], len(probe_med))

    median_probe = 0
    if len(all_med) != 0:
        all_med.sort()
        median_probe = all_med[int(len(all_med)/2)]    #median deltatime (sample rate)

    return (medians, median_probe, len(all_med))



def mean_rates(starttime = 0, endtime = 24, filter = None):
    means = {}   #with trips
    all_med = [] #without trips

    for probe_id, all_trips in all_probes.items():
        probe_med = []
        for trip in all_trips.trips:
            for timestamp, deltatime in zip(trip.trip, trip.deltas):   #******8
                if filter is not None:
                    satisfied = filter(probe_id, all_trips.provider, deltatime) #boolean
                    if not satisfied:
                        continue
                if timestamp.hour < starttime or timestamp.hour > endtime:
                    continue

                if type(deltatime) == timedelta:
                    probe_med.append(deltatime)
                    all_med.append(deltatime)
                else:
                    probe_med.append(timedelta(deltatime))
                    all_med.append(timedelta(deltatime))
        if len(probe_med) != 0:
            means[probe_id] = (sum(probe_med, timedelta())/len(probe_med), len(probe_med))

    mean_probe = 0   # mean of all probe sample rates
    if len(all_med) != 0:
        mean_probe = sum(all_med, timedelta())/(len(all_med))
    return (means, mean_probe, len(all_med))

# median_rates = []
# for probe_id, all_trips in all_probes.items():
#     for trip in all_trips.trips:
#         weight = len(trip.trip)
#         rate = 0
#         if type(trip.sample_rate) == timedelta:
#             if (trip.sample_rate != 0) and (trip.sample_rate != timedelta(0)):
#                 rate = timedelta(minutes=1) / trip.sample_rate
#         median_rates.append(weight * rate)
#
#
#
#
# overall_avg = sum(avg_rates)/sum(avg_rates)
# print("overall_avg of ", interested_provider, overall_avg, "on day", data_file)


def print_probe(id, verbose=False):
    print('Probe:', id)
    print('Provider:', all_probes[id].provider)
    print('Trips:')
    for trip in all_probes[id].trips:
        rate = 0
        if type(trip.sample_rate) == timedelta:
            rate = timedelta(minutes=1) / trip.sample_rate
        print('\tSample Rate:', '{:.3f}'.format(rate), 'per minute')
        print('\tStart Time:', trip.trip[0])
        print('\tEnd Time:', trip.trip[-1])
        print('\tTrip Length', len(trip.trip))
        if verbose:
            print('\tHistogram:')
            histogram = [0, 0, 0, 0, 0]
            for delta in trip.deltas:
                if delta < timedelta(minutes=1):
                    histogram[0] += 1
                elif delta < timedelta(minutes=2):
                    histogram[1] += 1
                elif delta < timedelta(minutes=3):
                    histogram[2] += 1
                elif delta < timedelta(minutes=4):
                    histogram[3] += 1
                else:
                    histogram[4] += 1
            LENGTH = 60
            total_length = sum(histogram) or 1
            for i, h in enumerate(histogram):
                print('\t\t', '{0} to {1} minutes:'.format(i, i + 1), '=' * int(LENGTH * h / total_length))
        print()


best = max(all_probes, key=lambda id: len(all_probes[id].timestamps))
#print_probe(best, verbose=True)


print("start writing files")
def write_file(data_file, filter):
    with open(data_file, 'w') as dfile:
        writer = csv.DictWriter(dfile, ['probe_id', 'median_sample_rate', 'mean_sample_rate', 'trip length'])
        writer.writeheader()
        (means, mean_probe, length) = mean_rates(filter = filter)
        (medians, median_probe, length1) = median_rates(filter = filter)
        for probe_id in means.keys():
            row = {}
            row['probe_id']      = probe_id
            if medians[probe_id][0] != timedelta(0):
                row['median_sample_rate']   = timedelta(seconds=60) /medians[probe_id][0]   #
            else:
                row['median_sample_rate'] = 0
            if means[probe_id][0] != timedelta(0):
                row['mean_sample_rate'] = timedelta(seconds=60) /means[probe_id][0]   #
            else:
                row['mean_sample_rate'] = 0

            row['trip length'] = means[probe_id][1] #should be the same as means[probe_id][0]
            writer.writerow(row)
        # tail row for summary information
        row = {}
        row['probe_id'] = 'overall'
        if median_probe != timedelta(0) and type(median_probe) == timedelta:
            row['median_sample_rate'] = timedelta(seconds=60) / median_probe #overall median sample rate
        else:
            row['median_sample_rate'] = 0
        if mean_probe != timedelta(0) and type(mean_probe) == timedelta:
            row['mean_sample_rate'] = timedelta(seconds=60) / mean_probe #overall median sample rate
        else:
            row['mean_sample_rate'] = 0
        row['trip length'] = length #inprecise, calculated the number of deltatimes
        writer.writerow(row)


write_file(data_file1, empty_filter)
print('finished writing Overall data')
write_file(data_file2, filter_func)
print('finished writing CONSUMER data')
write_file(data_file3, filter_func1)
print('finished writing FLEET data')

def groupby(file, field, groupby):
    df = pd.read_csv(file)
    groupby_regiment = df[str(field)].groupby(df[str(groupby)])
    groupby_regiment.describe()


"""
with open(data_file3, 'w') as dfile:
    writer = csv.DictWriter(dfile, ['time_range', 'mean_probe_rate', 'median_probe_rate', 'length'])
    writer.writeheader()
    for i in range(1, 24):
        filter_func = lambda id, provider, deltatime: provider[0:5] == 'CONSU' #and deltatime < timedelta(i) and deltatime >= timedelta(i-1)
        (means, mean_probe, length) = median_rates(i-1, i, filter_func)
        (medians, median_probe, length1) = mean_rates(i-1, i, filter_func)
        row = {}
        row['time_range'] = ' '+ str(i) + ' to ' + str(i+1)
        row['mean_probe_rate'] = mean_probe
        row['median_probe_rate'] = median_probe
        row['length'] = length
        writer.writerow(row)
"""

