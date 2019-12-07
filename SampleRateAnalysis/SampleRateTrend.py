import os, errno
import csv
from sampleRateCalc import preprocess
"""
====PARAMETERS====
"""

minSIZE = 4  # minimum number of timestamps needed to define a trip
BARRIER = 240  # the maximum separation between consecutive timestamps within a trip (in seconds)

directory = "/Users/Lijiayi/Documents/CALPATH/SampleRateAnalysis/"

folder = 'Feb_2017_DATA/data_raw/'

output_folder = 'Feb_2017_DATA/data_raw/AnalysisOutput/'
filedate = '20170208'


empty_filter = lambda provider, median_delta: provider[0:5] == 'CONSU' or provider[0:5] == 'FLEET'
filter_func = lambda provider, median_delta: provider[0:5] == 'CONSU' #and deltatime < timedelta(i) and deltatime >= timedelta(i-1)
filter_func1 = lambda provider, median_delta: provider[0:5] == 'FLEET'
trip_filter = lambda tripsize, median: tripsize > minSIZE and tripsize <=50 and median >= 5


"""
preprocess(minSIZE, maxGAP, filedate, 'OVERAL', 0, 24)
preprocess(minSIZE, maxGAP, filedate, 'CONSU', 0, 24)
preprocess(minSIZE, maxGAP, filedate, 'FLEET', 0, 24)

for i in range(0, 24):
    preprocess(minSIZE, maxGAP, filedate, 'OVERAL')
    preprocess(minSIZE, maxGAP, filedate, 'CONSU')
    preprocess(minSIZE, maxGAP, filedate, 'FLEET')
"""


