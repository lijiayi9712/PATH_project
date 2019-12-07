import os, sys
import gzip, shutil
import csv
from utils import *
""" Organizes files in the 'data_raw' directory into the appropriate month/day folders"""

analysis_path = os.path.join(os.getcwd())
source = os.path.join(analysis_path, 'data_raw')
assign_path = lambda month: os.path.join(source, month)
files = os.listdir(source)
regex = r's/.*\.(.*)\.(.*)\..*$/\1/'

for f in files[1::]:
    if f.endswith(".gz"):
        filepath = os.path.join(source, f)
        #f = os.path.basename(f)
        filedate = os.path.basename(f)[16:-15]#re.match(regex, f)
        dest = assign_path(digit_month_folder(filedate))
        if not os.path.exists(dest):
            try:
                os.makedirs(dest)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        if not os.path.isfile(dest):
            shutil.move(filepath, dest)
        #full_path = os.path.join(dest, f)