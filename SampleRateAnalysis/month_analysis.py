"""
Goal: Determine what percentage of data is
1. Only missing header
2. Only missing speed
3. Only missing probe_id
etc
"""
import pandas as pd
import os, glob
import csv


path = os.getcwd() + '/Oct_2017_DATA/data_raw/'  # use your path

def df_builder(path):
    """ Combines daily csv files into a df
        path: path to folder where csv files are stored
    """

    ###CHANGE FILE ENDING (.csv or .csv.gz)
    all_files = glob.glob(
        os.path.join(path, "probe_data_I210.201710*.csv"))  # advisable to use os.path.join as this makes concatenation OS independent
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    return pd.concat(df_from_each_file, ignore_index=True)

def percent_missing(df, cols):
    num_rows = df.shape[0]
    print('total points:', num_rows)
    result_row = []
    for col in cols:
        num_null = df[col].isnull().sum()
        percent_missing = num_null / num_rows
        result_row.append(percent_missing)
        print(col, ':', percent_missing, 'percent missing (', num_null, 'points)')
    return result_row


def main(month):
    results = []
    df = df_builder(path)
    print("Finished building...")
    cols = ['PROBE_ID', 'SAMPLE_DATE', 'LAT', 'LON', 'HEADING', 'SPEED', 'PROBE_DATA_PROVIDER', 'SYSTEM_DATE']
    results.append(cols)
    results.append(percent_missing(df, cols))

    month_analysis = path + 'AnalysisOutput/' + 'month_analysis_' + month + ".waynep1.csv"

    with open(month_analysis, 'a') as file:
        writer = csv.writer(file)
        for result in results:
            writer.writerow(result)
    print("writing to file complete!")

main('Oct_2017')

