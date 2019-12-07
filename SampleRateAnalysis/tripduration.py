import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from TripAnalysis import *

provider = 'CONSUMER'
time = 10
file = os.path.join('Analysis', os.path.join('analysis_output',os.path.join('2017_10',
          os.path.join('20171017', os.path.join('trip_meta_I210.20171017' + provider +'.waynep1.csv')))))

plt.close()
df = pd.read_csv(file)
df = df[(df['duration (in min)'] >= time - 0.1) & (df['duration (in min)'] <= time + 0.1)]
#df['DateTime'] = [convert_time(d) for d in df['start time']]
# df['Hour'] = [datetime.time(d).hour for d in df['DateTime']]
# df['Hour'].hist(cumulative=False, density=1, bins=np.arange(0, 25, 1))
df['distance'].hist(cumulative=False, density=1)
plt.xlabel('distance', fontsize=14)
plt.ylabel('count (number of trips)', fontsize=14)
title = 'trip_distance(' + '20171017' + provider + ') for trip duration around '+ str(time) + ' min'
plt.title(title)
plt.titlesize: 18
#plt.show()

analysis_output = os.path.join('Analysis', 'analysis_output')
visualization_folder = os.path.join(analysis_output, 'visualization')
plt.savefig(os.path.join(visualization_folder, title + ".png"))