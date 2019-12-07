from preprocess import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import glob
from collections import defaultdict, namedtuple
filedate = '20170208'



"""
=======================
the parameters 
======================
"""

min_size_list = [3, 5, 7]
barrier_list = [120, 240, 360]
min_travel_time_list = [120, 180, 240]
min_median_speed_list = [5, 10, 15]
min_travel_dist_list = [1000, 1800, 2400]







def test_parameter():
    minSIZE = 1  #5
    BARRIER = 100000 #240
    min_travel_time = 1 #180  # minimum number of seconds for a trip to be considered valid
    min_median_speed = 1 #5  # minimum median speed for a trip to be considered valid
    min_travel_distance = 1 #1000  # travel distance in feet, 1000ft to 0.5mi (2 or 3 blocks, Qijian asks)


            #preprocess(minSIZE, BARRIER, min_travel_time, min_median_speed, min_travel_distance, filedate, 'OVERAL')
    preprocess(minSIZE, BARRIER, min_travel_time, min_median_speed, min_travel_distance, filedate, 'CONSU')
    preprocess(minSIZE, BARRIER, min_travel_time, min_median_speed, min_travel_distance, filedate, 'FLEET')



"""
======================
test the file size 
corresponding to each 
set of parameters
==================
"""


# def file_size(filtered, processed, trip_meta):
#     Parameters = namedtuple('Parameters',['minSIZE', 'BARRIER', 'min_travel_time', 'min_median_speed', 'min_travel_distance'])
#     dictionary = {}
#     minSIZE = 5
#     BARRIER = 240
#     with open(filtered, 'w') as dfile1, open(processed, 'w') as dfile2, open(trip_meta, 'w') as dfile3:
#         for min_travel_time in min_travel_time_list:
#             for min_median_speed in min_median_speed_list:
#                 for min_travel_distance in min_travel_dist_list:
#                     directory = os.getcwd() + '/'
#                     output_folder = 'Feb_2017_DATA/data_raw/AnalysisOutput/' + filedate + '/'
#
#
#                     data_file2 = file_name_constructor("CONSUMER", "output_probe_data_I210.", min_travel_time, min_median_speed, min_travel_distance)
#                     data_file3 = file_name_constructor("FLEET", "output_probe_data_I210.", min_travel_time, min_median_speed, min_travel_distance)
#
#                     trip_meta2 = file_name_constructor("CONSUMER", "trip_meta_I210.", min_travel_time, min_median_speed, min_travel_distance)
#                     trip_meta3 = file_name_constructor("FLEET", "trip_meta_I210.", min_travel_time, min_median_speed, min_travel_distance)
#
#                     input_with_trip_id2 = file_name_constructor("CONSUMER", "trip_meta_I210.", min_travel_time, min_median_speed, min_travel_distance)
#                     input_with_trip_id3 = file_name_constructor("FLEET", "trip_meta_I210.", min_travel_time, min_median_speed, min_travel_distance)




directory = os.getcwd() + '/'
path = directory+'Feb_2017_DATA/data_raw/AnalysisOutput/' + filedate + '/sizeTesting/'
output_path = directory+'Feb_2017_DATA/data_raw/AnalysisOutput/' + filedate +'/'

if not os.path.exists(output_path):
    try:
        os.makedirs(directory + output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
"""
Change path and output_path variables above!
"""
def file_size():
    """"
    file_type: processed [input_with_trip_id] or trip_meta
    provider: consumer or fleet
    """
    header = ['FILE TYPE', 'PROVIDER', 'MIN SIZE of trip', 'GAP (seconds)', 'MIN TRAVEL TIME (seconds)', 'MIN MEDIAN SPEED (km/h)', 'MIN TRAVEL DISTANCE (feet)', 'NUM ROWS']
    minSIZE = 5
    BARRIER = 240
    with open(output_path + 'size_analysis.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        for min_travel_time in min_travel_time_list:
            for min_median_speed in min_median_speed_list:
                for min_travel_distance in min_travel_dist_list:
                    def file_name_constructor(provider="", output_file_name=""):
                        return directory + output_folder + output_file_name + filedate + '_' + str(minSIZE) + '_' + str(
                            BARRIER) + '_' + str(min_travel_time) + '_' + str(min_median_speed) + '_' + str(
                            min_travel_distance) + provider + ".waynep1.csv"
                    directory = os.getcwd() + '/'
                    output_folder = 'Feb_2017_DATA/data_raw/AnalysisOutput/' + filedate + '/sizeTesting/'
                    def file_processor(file):
                        row_count = sum(1 for row in open(file))
                        short_provider = file[-13:-12]
                        if short_provider == 'R':
                            file_type_letter = file[-52:-51]
                            if file_type_letter == 'a':
                                file_type = 'trip_meta'
                            else:
                                file_type = 'processed'
                            provider = 'CONSUMER'
                            row = [file_type, provider, minSIZE, BARRIER, min_travel_time, min_median_speed,
                                   min_travel_distance, row_count]
                            writer.writerow(row)
                    trip_meta2 = file_name_constructor("CONSUMER", "trip_meta_I210.")
                    file_processor(trip_meta2)
                    trip_meta3 = file_name_constructor("FLEET", "trip_meta_I210.")
                    file_processor(trip_meta3)
                    input_with_trip_id2 = file_name_constructor("CONSUMER", "trip_meta_I210.")
                    file_processor(input_with_trip_id2)
                    input_with_trip_id3 = file_name_constructor("FLEET", "trip_meta_I210.")
                    file_processor(input_with_trip_id3)


        """
        for file in glob.glob(path + '/*.csv'):
            ###for each file, find the size of the file and the params!
            row_count = sum(1 for row in open(file))
            short_provider = file[-13:-12]
            if short_provider == 'R':
                file_type_letter = file[-52:-51]
                if file_type_letter == 'a':
                    file_type = 'trip_meta'
                else:
                    file_type = 'processed'
                provider = 'CONSUMER'
                minSize = file[-37:-36]
                BARRIER = file[-35:-32]
                min_travel_time = file[-30:-27]
                min_median_speed = file[-26:-25]
                min_travel_distance = file[-24:-20]
            else:
                file_type_letter = file[-49:-48]
                if file_type_letter == 'a':
                    file_type = 'trip_meta'
                else:
                    file_type = 'processed'
                provider = 'FLEET'
                minSize = file[-34:-33]
                BARRIER = file[-30:-27]
                min_travel_time = file[-27:-24]
                min_median_speed = file[-23:-22]
                min_travel_distance = file[-21:-17]

            #print(minSize, BARRIER, min_travel_time, min_median_speed, min_travel_distance)
            row = [file_type, provider, minSize, BARRIER, min_travel_time, min_median_speed, min_travel_distance, row_count]
            writer.writerow(row)
            """

    print("writing to file complete!")

file_size()





"""
=========================
difference between sample
time and system time
=========================
"""

def transmissionTime(file):
    plt.close()
    df = pd.read_csv(file)
    df['transmissionTime'] = (pd.to_datetime(df['SYSTEM_DATE'], format='%Y-%m-%d %H:%M:%S') - pd.to_datetime(df['SAMPLE_DATE'], format='%Y-%m-%d %H:%M:%S')).dt.total_seconds()
    plt.hist(df[df['transmissionTime'] < 0]['transmissionTime'], bins=np.arange(-1250, 0, 1), cumulative=True, density=1)
    plt.xlabel('difference between system and sample time (in seconds)',
               fontsize=14)  # 'sample rate (number of sample points/minutes)'
    plt.ylabel('count (number of samples)', fontsize=14)
    plt.title('distribution of difference between system and sample time')
    plt.titlesize: 18
    plt.show()




directory = os.getcwd() + '/'
folder = 'Feb_2017_DATA/data_raw/'
data_file = directory + folder + "probe_data_I210." + filedate + ".waynep1.csv"  # original dataset