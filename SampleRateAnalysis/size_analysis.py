from preprocess import *

def size_analysis(date, provider, writer):
    """"
    Writes the sizes of `processed` and `trip_meta` to a given output_file (ie size_analysis.csv) using the given writer.
    In addition, writes the size of the bounding_box files for this provider, and the percent kept in `processed`.
    """
    with open(size_analysis, 'a') as outfile:

        def file_name_constructor(provider="", output_file_name=""):
            return os.path.join(os.path.join(directory, output_folder), output_file_name + date + provider + ".waynep1.csv")

        def file_processor(file):
            #print("file being processed: ", file_type)
            efile = open(file)
            processed_num_rows = len(efile.readlines())

            bounding_box = file_name_constructor(output_file_name="bounding_box_.")

            bounding_df = pd.read_csv(bounding_box)
            bounding_provider= bounding_df.loc[bounding_df['PROBE_DATA_PROVIDER'].str.slice(0,5) == provider[0:5]]
            bounding_provider_rows = bounding_provider.shape[0]

            trip_meta = file_name_constructor(provider, "trip_meta_I210.")
            trip_meta_df = pd.read_csv(trip_meta)
            num_trips = trip_meta_df.shape[0]
            #print("row count: ", row_count)
            #row_count = sum(1 for row in csv.reader(file))
            row = { 'date': date, 
                    'provider': provider, 
                    'bounding num rows': bounding_provider_rows,
                    'processed num rows' : processed_num_rows,
                    'percent of bounding box kept': float(processed_num_rows / bounding_provider_rows),
                    'num trips' : num_trips}
            writer.writerow(row)

        processed = file_name_constructor(provider, "processed_I210.")
        file_processor(processed)
if __name__ == "__main__":
    filedates = clean_filedates(sys.argv[1:])

    folder = os.path.join('Analysis', 'data_raw')
    analysis_output = os.path.join('Analysis', 'analysis_output')

    first_date = filedates[0]
    month_folder = os.path.join(analysis_output, digit_month_folder(first_date))
    month_day = os.path.join(digit_month_folder(first_date), first_date)
    output_folder = os.path.join(analysis_output, month_day)
    directory = os.getcwd()
    if not os.path.exists(os.path.join(directory, output_folder)):
        try:
            os.makedirs(os.path.join(directory, output_folder))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    for provider in ['CONSUMER', 'FLEET']:
        size_analysis = os.path.join(month_folder,'size_analysis_' + provider + '.csv')
        with open(size_analysis, 'a') as outfile:
            test = open(size_analysis)
            outfile_size = len(test.readlines())
            writer = csv.DictWriter(outfile, delimiter=',', lineterminator='\n',
                                                    fieldnames=['date', 'provider', 'bounding num rows', 'processed num rows', 'percent of bounding box kept',
                                                    'num trips'])
            if outfile_size == 0:
                writer.writeheader()
            for date in filedates:
                print('Finding size of', date, provider, '...')
                size_analysis(date, provider, writer)
