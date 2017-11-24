#do the first import to prevent getting zero from division , eg: 12/1222222=0
from __future__ import division
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


data_dir = '/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv'
img_dir = '/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/cam0/stacked/'
new_label_dir = '/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/cam0/stacked_img_label.csv'
df = pd.read_csv(data_dir)
n = []
# print (df)

def get_new_label_except_time(df, ratio):
    df = df[df.columns[1:]]
    df = df.iloc[0] + (df.iloc[1]-df.iloc[0])*ratio
    df = pd.DataFrame(df).transpose()
    # print (df)
    return df

# print ("new lable is {}".format(get_new_label_except_time(df, 0.5)))

def label_gen(image_time, DataFrame):
    # print ("image time is {}".format(image_time))
    # print (DataFrame['#timestamp'])
    #1403715524907143168
    if image_time not in DataFrame['#timestamp'].values:
        try:
            Min = DataFrame['#timestamp'] < image_time
            Max = DataFrame['#timestamp'] > image_time
        except:
            print ("Image time not found in the dataframe : {}".format(image_time))
        # print (Min)
        # print (Max)
        idx_Min = DataFrame.ix[Min, '#timestamp'].idxmax()
        idx_Max = DataFrame.ix[Max, '#timestamp'].idxmin()
        temp = DataFrame.ix[idx_Min:idx_Max]
        # print ("temp is {}".format(temp))
        time_series = temp['#timestamp']
        # print ("times series is {}".format(time_series))
        # print (time_series.iloc[1])
        # print (time_series.iloc[0])
        distance = time_series.iloc[1] - time_series.iloc[0]
        # print ("distance is {}".format(distance))
        delta_t = image_time - time_series.iloc[0]
        # print ("delta timestamp is {}".format(delta_t))
        time_ratio = (delta_t / distance)
        raw_df = get_new_label_except_time(temp, time_ratio)

        time_df = pd.DataFrame.from_dict({'#timestamp': image_time}, orient='index').T
        chunks = [time_df, raw_df]
        new_lable = pd.concat(chunks, axis=1)
        # print (new_lable)

        return new_lable

    elif image_time in DataFrame['#timestamp'].values:
        # print ("Image time in dataframe")
        return df[DataFrame['#timestamp'] == image_time]


# label_gen(1403715524927143300, df)

if __name__ == '__main__':
    f = os.listdir(img_dir)
    f.sort()
    first_img = os.path.splitext(f[20])[0]
    first_img = int(first_img)
    # print ("first img is {}".format(first_img))
    # first_img = 1403715524917143041
    label_df = label_gen(first_img, df)

    for img in f[21:-20]:
        timestamp = os.path.splitext(img)[0]
        timestamp = int(timestamp)
        # print (timestamp)
        temp_label_df = label_gen(timestamp, df)
        label_df = [label_df, temp_label_df]
        label_df = pd.concat(label_df)

    finished_df = label_df.reset_index().drop('index', 1)
    finished_df.to_csv(new_label_dir, index=False)
    # print (label_df.reset_index().drop('index', 1))

    # print (label_df)


# target : 1403715523962142976
# first  : 1403715524907143168
