import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import numpy as np
from dateutil import tz
import datetime
import os
from month_analysis import *

directory = os.getcwd() + '/' + 'SampleAnalysis/'
folder = 'Feb_2017_DATA/data_raw/'

filedate = '20170208/'

output_folder = '/Feb_2017_DATA/data_raw/AnalysisOutput/' + filedate + 'graphs/'

trip_meta_file = directory + folder + 'trip_meta_I210.20170208_5_240_180_5.0_1000FLEET.waynep1.csv'

trip_id_FLEET_file = directory + folder + 'AnalysisOutput/' + filedate + 'input_with_trip_id_I210.20170208_5_240_180_5.0_1000FLEET.waynep1.csv'
trip_id_CONSUMER_file = directory + folder + 'AnalysisOutput/' + filedate + 'input_with_trip_id_I210.20170208_5_240_180_5.0_1000CONSUMER.waynep1.csv'

if not os.path.exists(directory + output_folder):
    try:
        os.makedirs(directory + output_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

"""
Creates a pdf and cdf with x axis as x, using the clipped df
"""
def create_histograms(clipped_df, x, minimum, maximum, file_save, file):
    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(211)  # Create matplotlib axes
    ax2 = fig.add_subplot(212)  # Create another axes that shares the same x-axis as ax.

    ax1.hist(clipped_df[x], cumulative=True, bins=100, density=1)
    ax1.set_xlabel(x,
                   fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title('cdf ' + x + str(minimum) + ' to ' + str(maximum) + ' (' + file[-32:-12] + ')')
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14

    ax2.hist(clipped_df[x], bins=100, density=1)  # np.arange(0, 40, 1)  bins = np.arange(0, 150, 1)
    ax2.set_xlabel(x,
                   fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.set_title('pdf ' + x + str(minimum) + ' to ' + str(maximum) + ' (' + file[-32:-12] + ')')
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_label_coords(0.8, -0.18)
    ax2.titlesize: 14
    # ax2.set_yscale('log', nonposy='clip')
    fig.tight_layout()
    plt.savefig(directory + output_folder + file_save + str(minimum) +
                ' to ' + str(maximum) + '(' + file[ -38:-12] + ')' + ".png")

    #### Uncomment for printed summary stats
    #print(clipped_df.describe())

""" 
Creates stacked histograms
"""
def create_stacked_histograms(clipped_df, x, minimum, maximum, file_save, file):
    #df1x = clipped_df[x]
    #df2x = clipped_df[x]

    #plt.figure()
    #plt.hist([df1x, df2x], stacked=True)
    #plt.show()

    ax1.hist(clipped_df[x], cumulative=True, bins=100, density=1)
    ax1.set_xlabel(x,
                   fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title('cdf ' + x + str(minimum) + ' to ' + str(maximum) + ' (' + file[-32:-12] + ')')
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14



""" column analysis, for each range of probes per trip
    x: column for x-axis of histogram
    file_save: the additional word to add to the saved file name
    file: filepath
    to_clip: column to clip range on. Note that some files do not have relevant
             columns to clip on, so the default is None
    
    histogram_analysis(x='trip size', to_clip='trip_size', step_size=10, file_save='trip_size', 
            file=directory + folder + 'trip_meta_I210.20170208_5_240_180_5.0_1000FLEET.waynep1.csv')

"""
def histogram_analysis(x, step_size, file_save, file,  to_clip=None):
    plt.close()
    df = pd.read_csv(file)
    if to_clip is None:
        df = df[[x]]
    else:
        df = df[[x, to_clip]]

    for i in range(0, 300, step_size):
        minimum = i
        maximum = i + step_size
        if to_clip is None:
            # if not clipping, then create the histogram and break out of loop
            create_histograms(df, x, 'start', 'end', file_save, file)
            break
        else:
            clipped = df[(df[to_clip] > minimum) & (df[to_clip] < maximum)]
        create_histograms(clipped, x, minimum, maximum, file_save, file)

    print('finished outputting graphs')
    # plt.show()

#
"""
Similar to histogram_analysis function, but specifically to check different providers
x: column of interest (ie heading)
provider_list: all providers to search through
overlay: (True or False) overlay all providers on one graph
to_clip: column to use for step_size clipping (ie trip size)

"""
def provider_analysis(x, provider_list, overlay, file_save, file):
    plt.close()
    df = pd.read_csv(file)

    df = df[[x, 'PROBE_DATA_PROVIDER']]


    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(211)  # Create matplotlib axes

    ### Uncomment for overlaid histograms per provider num
    for curr_provider in provider_list:
        clipped = df[(df['PROBE_DATA_PROVIDER'] == curr_provider)]
        ax1.hist(clipped[x], cumulative=False,bins=100, density=1, label=curr_provider)

        create_histograms(clipped, x, 'provider = ' + curr_provider, ' ', file_save + curr_provider, file)

    #remove the following for BASIC HISTOGRAMS
    ax1.set_xlabel(x,
                   fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title('pdf ' + x + ' for FLEET probe' + ' (' + file[-32:-12] + ')')
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14
    #plt.legend(bbox_to_anchor=(1,1))

    plt.savefig(directory + output_folder + file_save + ' (' + file[-38:-12] + ')' + ".png")
    #plt.show()
    print('finished outputting graphs')
    #plt.show()


####RUN TRIP_SIZE_ANALYSIS
#histogram_analysis(x='duration (in min)', to_clip='trip size',
#                   step_size=50, file_save='trip_duration', file=file)
####

####RUN HEADING ANALYSIS
#histogram_analysis(x='HEADING', step_size=150, file_save='heading', file=trip_id_file)

# Heading analysis based on provider:
fleet_nums = ['05', '24', '30', '33', '41', '42', '44', '51', '53', '54']
fleet_providers = []
for num in fleet_nums:
    fleet_providers.append('FLEET' + num)
fleet_providers_1= ['FLEET54']
provider_analysis(x='HEADING', provider_list=fleet_providers_1, overlay=False,
              file_save='heading_per_FLEET_provider', file=trip_id_FLEET_file)

consumer_nums = ['05', '09', '10', '14', '15', '16', '18', '21', '22', '24']
consumer_providers = []
for num in consumer_nums:
    consumer_providers.append('CONSUMER' + num)

#provider_analysis(x='HEADING', provider_list=consumer_providers, overlay=False,
#                  file_save='heading_per_CONSUMER_provider', file=trip_id_CONSUMER_file)