###############################################################################
# "Sentiment-driven statistical causality in multimodal systems"
#
#  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
#
#  Ioannis Chalkiadakis, ic14@hw.ac.uk
#  April 2021
###############################################################################

import pickle
from collections import Counter
import numpy as np
import pandas as pd
import ipdb
from pandas import DataFrame as df
from plotly import graph_objects as go, io as pio, express as px
from plotly.subplots import make_subplots
from scipy import sparse as sp
import matplotlib.pyplot as plt
from scipy.stats import iqr
import math


def get_time_batches(timeseries_elements):
    # Input needs to be sorted per day/timestamp

    batches = {}
    curr_batch = []
    curr_idx = []
    k = 0
    for i in range(len(timeseries_elements)):
        tstamp = np.datetime64(timeseries_elements[i][0])
        if tstamp in curr_batch:
            curr_idx.append(i)
            curr_batch.append(np.datetime64(tstamp))
            k += 1
        elif len(curr_batch) > 0:
                # sort curr_batch
                sorted_idx = np.argsort(curr_batch)
                date = curr_batch[0].astype('datetime64[D]')
                assert all([cb.astype('datetime64[D]') == date for cb in curr_batch])
                batches[date] = {"number of ngrams": len(curr_idx),
                                          "indices": np.array(curr_idx)[sorted_idx],
                                          "timestamp": date}
                curr_batch = [np.datetime64(timeseries_elements[i][0])]
                curr_idx = [i]
                k = 1
        else:
            curr_batch = [np.datetime64(timeseries_elements[i][0])]
            curr_idx = [i]
            k = 1
    sorted_idx = np.argsort(curr_batch)
    date = curr_batch[0].astype('datetime64[D]')
    assert all([cb.astype('datetime64[D]') == date for cb in curr_batch])
    batches[date] = {"number of ngrams": len(curr_idx),
                     "indices": np.array(curr_idx)[sorted_idx],
                     "timestamp": date}

    return batches


def get_timeseries_summary_per_day(timeseries, timeseries_elements, metadata, summary="median"):

    # Fix formating for datetime64 datatype
    dates = []
    for i in timeseries_elements:
        if "T" in i[0]:
            d = i[0].split("T")
            h, m = d[1].split("-")
            if len(h) == 1:
                h = "0" + h
            time = ":".join([h, m])
            dates.append(np.datetime64("T".join([d[0], time])))
        else:
            dates.append(np.datetime64(i[0]))
    timeseries_elements = [(dates[i], timeseries_elements[i][1], timeseries_elements[i][2])
                           for i in range(len(timeseries_elements))]
    sorted_idx_per_date = np.argsort(dates)
    timeseries_elements = (np.array(timeseries_elements)[sorted_idx_per_date]).tolist()
    metadata = (np.array(metadata)[sorted_idx_per_date])
    timeseries = (np.array(timeseries)[sorted_idx_per_date])

    time_batches = get_time_batches(timeseries_elements)
    meta = dict()
    summary_ts = dict()

    for k in time_batches.keys():
        timestamp = time_batches[k]["timestamp"]
        idx = time_batches[k]["indices"]
        ts = timeseries[idx]
        if summary == "median":
            summary_ts[(timestamp.astype('datetime64[D]'))] = np.median(ts)
        elif summary == "IQR":
            summary_ts[(timestamp.astype('datetime64[D]'))] = iqr(ts)
            assert not math.isnan(summary_ts[timestamp.astype('datetime64[D]')])
        else:
            raise NotImplementedError("Summary method not available")
        ngrams = metadata[idx]
        m = []
        for ng in ngrams:
            # keep only the ngram information
            m.append(ng.split("<br>")[-1])
        meta[(timestamp.astype('datetime64[D]'))] = {"annotation": m, "number of ngrams":
            time_batches[k]["number of ngrams"]}

    return summary_ts, meta, time_batches


def get_timestamp_gaps_dates(dates, output_file="/tmp/gap_days.html"):

    if isinstance(dates[0],np.datetime64):
        dates = [str(d.astype('datetime64[D]')) for d in dates]

    months_days = [(int(i.split('-')[1]), int(i.split('-')[2])) for i in dates]
    gap_days = []
    gap_index = []
    for i in range(len(months_days[1:])):
        day1 = months_days[i]
        day2 = months_days[i + 1]
        if day2[0] == day1[0]:
            gap_days.append(day2[1] - day1[1])
        else:
            if day1[0] in [1, 3, 5, 7, 8, 10, 12]:
                gap_days.append(31 - day1[1] + day2[1])
            elif day1[0] in [4, 6, 9, 11]:
                gap_days.append(30 - day1[1] + day2[1])
            else:
                # both 2018, 2019 have 28 days in February
                gap_days.append(28 - day1[1] + day2[1])
        if gap_days[-1] > 1:
           gap_index.append((i, gap_days[-1]))

    cnts = Counter(gap_days)
    dframe = df({"Counts": list(cnts.values()), "Gap": list(cnts.keys())})
    fig = px.bar(dframe, y="Counts", x="Gap")
    fig.update_layout(title="Gap days between successive publication", title_x=0.5)
    pio.write_html(fig, output_file, auto_open=False)
    try:
        pio.write_image(fig, output_file.replace(".html", ".png"))
        pio.write_image(fig, output_file.replace(".html", ".eps"))
    except:
        pass

    return gap_index