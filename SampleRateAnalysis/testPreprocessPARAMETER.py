from preprocess import *
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
import os
import csv
from collections import defaultdict, namedtuple

filedate = '20171017'

# directory = os.getcwd() + '/'
# folder = directory +'Analysis/data_raw/'
# output_folder = directory + 'Analysis/analysis_output/Oct_2017/' + filedate +'/'
# if not os.path.exists(output_folder):
#     try:
#         os.makedirs(directory + output_folder)
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise



def file_cat():
    interesting_files = glob.glob("*.csv")
    header_saved = False
    with open('output.csv', 'wb') as fout:
        for filename in interesting_files:
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)
"""
=======================
the parameters 
======================
"""
min_travel_time = 180  # minimum number of seconds for a trip to be considered valid
min_median_speed = 5  # minimum median speed for a trip to be considered valid
min_travel_distance = 1000  # travel distance in feet, 1000ft to 0.5mi (2 or 3 blocks, Qijian asks)
minSIZE = 5
maxGAP = 240

min_size_list = [3, 5, 7]
gap_list = [120, 240, 360]
min_travel_time_list = [120, 240]
min_median_speed_list = [5, 10, 15]
min_travel_dist_list = [1000, 1800, 2400]


def test_parameters():
    for min_median_speed in min_median_speed_list:
        for min_travel_distance in min_travel_dist_list:
            for min_travel_time in min_travel_time_list:
                preprocess(minSIZE, maxGAP, min_travel_time, min_median_speed, min_travel_distance, filedate, 'CONSU')
                preprocess(minSIZE, maxGAP, min_travel_time, min_median_speed, min_travel_distance, filedate, 'FLEET')

def test_preprocess():
    preprocess(minSIZE, maxGAP, min_travel_time, min_median_speed, min_travel_distance, filedate, 'CONSU')
    preprocess(minSIZE, maxGAP, min_travel_time, min_median_speed, min_travel_distance, filedate, 'FLEET')

test_preprocess()

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