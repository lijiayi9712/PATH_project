import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
#from analyzePenetration import day
#from calPATH import folder, filedate, SECONDS
import seaborn as sns

minSIZE = 5  # minimum number of timestamps needed to define a trip
BARRIER = 240  # the maximum separation between consecutive timestamps within a trip (in seconds)

directory = "/Users/Lijiayi/Documents/CALPATH/SampleRateAnalysis/"

folder = 'Feb_2017_DATA/data_raw/'

output_folder = 'Feb_2017_DATA/data_raw/AnalysisOutput/'
filedate = '20170208'


#data_file = "/Users/Lijiayi/Documents/CALPATH/SampleRateAnalysis/" + folder + "/data_raw/probe_data_I210." + filedate + ".waynep1.csv"
#data_file1 = "/Users/Lijiayi/Documents/CALPATH/SampleRateAnalysis/" + folder + "/output_probe_data_I210." + filedate + ".waynep1.csv"
#data_file2 = "/Users/Lijiayi/Documents/CALPATH/SampleRateAnalysis/" + folder + "/output_probe_data_I210." + filedate + "CONSUMER" + ".waynep1.csv"
#data_file3 = "/Users/Lijiayi/Documents/CALPATH/SampleRateAnalysis/" + folder + "/output_probe_data_I210." + filedate + "FLEET" + ".waynep1.csv"
#trip_meta = "/Users/Lijiayi/Documents/CALPATH/SampleRateAnalysis/" + folder + "/trip_meta_I210." + filedate + ".waynep1.csv"

data_file = directory + folder + "probe_data_I210." + filedate + ".waynep1.csv"
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
trip_meta = directory + output_folder + "trip_meta_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"


"""
rates = []
for i in range(1, 31):
    daily = day(i)
    first_penetration_rate = daily[0][0]
    # 1st sample
    second_penetration_rate = daily[1][0]
    # 2nd sample
    rates.append(first_penetration_rate)
    rates.append(second_penetration_rate)

plt.hist(rates, bins=10)
plt.title("Histogram for monthly penetration rate")
plt.show()
"""


"""
plt.figure()
fig2, ax2 = df['mean_sample_rate'].plot.hist(bins=np.linspace(0, 7, 35))
ax2.set_xlabel('sample rate')
ax2.set_ylabel('count')
ax2.set_title(r'mean sample rate of 2014-04-07')
plt.show()
"""

"""
=====================================
histogram:
subplots histogram of median_sample_rate and
mean_sample_rate of day 2017-09-07

histogram of using median_sample_rate
of any day

bins1, bins2 can be np.arange(0, 100, 1) or bins=np.linspace(0, 7, 50))
=====================================
"""


def subplot_histogram(filename, file, type):
    plt.close()
    df = pd.read_csv(file)

    #sns.distplot(df['median_sample_rate'],
    #             label=df['median_sample_rate'],
    #             hist=True,
    #             kde=True,
    #             rug=False,
    #             bins=np.arange(0, 8, 0.4),
     #            ax=ax0)
    #sns.distplot(df['median_delta'], label=df['median_delta'], hist=True, kde=False, rug=False,
    #             bins=np.arange(0, 8, 0.4))
    plt.hist(df['median_delta'], cumulative=True, bins = np.arange(0, 25, 0.5))   #np.arange(0, 40, 1)
    #plt.xlabel('sample rate (number of sample points/minutes)', fontsize=14)
    plt.xlabel('sample delta (in seconds)', fontsize=14) #'sample rate (number of sample points/minutes)'
    plt.ylabel('count (number of probes)', fontsize=14)
    plt.title(str(type) + ' median sample delta(' + filedate + ')'+'_' + str(minSIZE) + '_' +str(BARRIER)) # ' median sample rate('
    plt.titlesize: 18

    #sns.distplot(df['mean_sample_rate'], label=df['mean_sample_rate'], hist=True, kde=True, rug=False, bins=np.arange(0, 8, 0.4), ax=ax1)
    #ax1.hist(df['mean_delta'], bins=bins2)
    #ax1.set_xlabel('sample delta', fontsize=10)
    #ax1.set_ylabel('count (number of probes)', fontsize=10)
    #ax1.set_title(str(type) + ' mean delta(' + filedate + ')')   #' mean sample rate('
    #ax1.titlesize: 10
    #ax1.set_yscale('log', nonposy='clip')
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.125,  # the left side of the subplots of the figure
    #                    right=0.9,  # the right side of the subplots of the figure
    #                    bottom=0.1,  # the bottom of the subplots of the figure
    #                    top=0.9,  # the top of the subplots of the figure
    #                    wspace=0.2,  # the amount of width reserved for blank space between subplots
    #                    hspace=0.2)  # the amount of height reserved for white space between subplots)
    #ax1.set_xticks(fontsize=8, rotation=60)

    plt.savefig(directory + output_folder + filename +str(type) + ".png")
    print('finished outputing graph')

#subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(maxGAP), data_file1, 'OVERAL')
#subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(maxGAP), data_file2, 'CONSU')
#subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(maxGAP), data_file3, 'FLEET')

minSIZE = 2
BARRIER = 120
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')

minSIZE = 10
BARRIER = 120
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')
minSIZE = 20
BARRIER = 120
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')

minSIZE = 2
BARRIER = 240
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')
minSIZE = 5
BARRIER = 240
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')
minSIZE = 10
BARRIER = 240
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')
minSIZE = 20
BARRIER = 240
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')


minSIZE = 2
BARRIER = 360
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')
minSIZE = 5
BARRIER = 360
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')
minSIZE = 10
BARRIER = 360
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')
minSIZE = 20
BARRIER = 360
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')



minSIZE = 2
BARRIER = 600
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')



minSIZE = 5
BARRIER = 600
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')


minSIZE = 10
BARRIER = 600
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')



minSIZE = 20
BARRIER = 600
data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + ".waynep1.csv"
data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) +"CONSUMER" + ".waynep1.csv"
data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(BARRIER) + "FLEET" + ".waynep1.csv"
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file1, 'OVERAL')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file2, 'CONSU')
subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(BARRIER), data_file3, 'FLEET')


def graphcomparation(minSIZE = None, BARRIER = None):

    if minSIZE != None:
        BARRIER = 120 #240, 360, 600
        filename = 'samplerate' + '_' + str(minSIZE) + '_' + str(BARRIER)
        graph1 = im = np.array(Image.open(open(directory + output_folder + filename +str(type) + ".png")))

        BARRIER = 240  # 120 240, 360, 600
        filename = 'samplerate' + '_' + str(minSIZE) + '_' + str(BARRIER)
        graph2 = np.array(Image.open(open(directory + output_folder + filename +str(type) + ".png")))

        BARRIER = 360  # 240, 360, 600
        filename = 'samplerate' + '_' + str(minSIZE) + '_' + str(BARRIER)
        graph3 = np.array(Image.open(open(directory + output_folder + filename +str(type) + ".png")))

        BARRIER = 600  # 240, 360, 600
        filename = 'samplerate' + '_' + str(minSIZE) + '_' + str(BARRIER)
        graph4 = np.array(Image.open(open(directory + output_folder + filename +str(type) + ".png")))

        plt.close()
        f, axarr = plt.subplots(2,2)
        axarr[0, 0].imshow(graph1)
        axarr[0, 1].imshow(graph2)
        axarr[1, 0].imshow(graph3)
        axarr[1, 1].imshow(graph4)




def histogram(filename, data_file):
    plt.close()
    df = pd.read_csv(data_file)
    lastrow = df.tail(1)
    mark = [lastrow['median_sample_rate'], lastrow['mean_sample_rate']]
    df.drop(df.index[len(df) - 1])
    plt.hist(df['median_sample_rate'], bins=np.linspace(0, 7, 50), markevery=mark)
    plt.xlabel('sample rate', fontsize=9)
    plt.ylabel('count', fontsize=9)
    plt.title(r'median sample rate of ' + filedate)
    plt.titlesize: 18
    plt.savefig("summer_research/" + str(filename) + ".png")

def durationAnalysis(filename, file, type):  #durationAnalysis('durationAnalysis'+'_' + str(minSIZE) + '_' +str(maxGAP), trip_meta, 'FLEET')
    plt.close()
    df = pd.read_csv(file)
    plt.hist(df['duration (in min)'], bins = np.arange(0, 200, 2))
    plt.xlabel('duration of trip (in minutes)', fontsize=14)  # 'sample rate (number of sample points/minutes)'
    plt.ylabel('count (number of trip)', fontsize=14)
    plt.title(str(type) + ' trip duration (' + filedate + ')' + '_' + str(minSIZE) + '_' + str(
        BARRIER))
    plt.titlesize: 18
    plt.savefig(directory + output_folder + filename + str(type) + '_Duration_' + ".png")
    print('finished outputing graph')

durationAnalysis('durationAnalysis'+'_' + str(minSIZE) + '_' +str(BARRIER), trip_meta, 'FLEET')

#subplot_histogram(filedate + str(SECONDS/60) + "min", data_file1, 'Overall', np.linspace(0, 3, 8), np.linspace(0, 3, 8))
#subplot_histogram(filedate +'_CONSUMER_' + str(SECONDS/60) + "min", data_file2, 'CONSUMER', np.linspace(0, 3, 8), np.linspace(0, 3, 8))
#subplot_histogram(filedate +'_FLEET_' + str(SECONDS/60) + "min", data_file3, 'FLEET', np.linspace(0, 3, 8), np.linspace(0, 3, 8))



#subplot_histogram(filedate +'_CONSUMER_' + str(SECONDS/60) + "min", data_file2, 'CONSUMER', np.linspace(0, 80, 80), np.linspace(0, 75, 75))
#subplot_histogram(filedate +'_FLEET_' + str(SECONDS/60) + "min", data_file3, 'FLEET', np.linspace(0, 10, 40), np.linspace(0, 10, 40))


#data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(maxGAP) + ".waynep1.csv"
#data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(maxGAP) +"CONSUMER" + ".waynep1.csv"
#data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' +str(maxGAP) + "FLEET" + ".waynep1.csv"
#subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(maxGAP), data_file1, 'OVERAL')
#subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(maxGAP), data_file2, 'CONSU')
#subplot_histogram('samplerate'+'_' + str(minSIZE) + '_' +str(maxGAP), data_file3, 'FLEET')


def histogram_double_axis(filename1, filename2, field1, field2, data_file):
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    fig = plt.figure()  # Create matplotlib figure

    ax1 = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax1.twinx()  # Create another axes that shares the same x-axis as ax.

    width = 0.4

    ax1.hist(df1[str(field1)], bins=np.linspace(0, 20, 40), color='red', width=width, position=1)
    ax2.hist(df2[str(field2)], bins=np.linspace(0, 20, 40), color='blue', width=width, position=0)
    #df1.field1.plot(kind='bar', color='red', ax=ax1, width=width, position=1)
    #df2.field2.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

    ax1.set_ylabel(str(field1))
    ax2.set_ylabel(str(field2))
    plt.savefig("summer_research/" + data_file + str(filename1) + str(filename2) + ".png")

"""
=====================
trip length analysis
=====================
"""

def triplength(file, filedate):
    plt.close()
    df = pd.read_csv(file)
    plt.hist(df['trip size'], bins=np.linspace(0, 200, 100))
    df.drop(df.index[len(df) - 1])
    plt.xlabel('total trip length', fontsize=9)
    plt.ylabel('number of probes', fontsize=9)
    plt.yscale('log', nonposy='clip')
    plt.title('total trip length distribution')
    plt.savefig(output_folder + filedate + "trip_length" + ".png")



"""
=====================================
boxplot:
subplots histogram of median_sample_rate and
mean_sample_rate of day 2017-09-07

histogram of using median_sample_rate
of any day
=====================================
"""


def boxplot(file):
    sns.set_style("whitegrid")
    df = pd.read_csv(file)
    ax = sns.boxplot(df['median_sample_rate'])
    plt.show()
