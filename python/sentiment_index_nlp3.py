###############################################################################
# "Sentiment-driven statistical causality in multimodal systems"
#
#  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
#
#  Ioannis Chalkiadakis, ic14@hw.ac.uk
#  April 2021
###############################################################################

import time
import datetime
from python.utils import get_timeseries_summary_per_day, get_timestamp_gaps_dates
import pickle
import ipdb
import os
from scipy.io import savemat
import numpy as np
import pandas as pd


def convolve(x, y):

    return np.sum(np.multiply(x, np.flip(y)))


if __name__ == "__main__":

    DIR = "./data/sentiment"
    DIR_out = "./data/output_nlp3"

    if not os.path.exists(DIR_out):
        os.makedirs(DIR_out)

    sentiment = ["pos", "neg", "neutral"]
    kwargs_list = ["BTC", "ETH", "LTC", "XRP", "TRX"]
    sites = ["cryptoslate", "cryptodaily"]

    ts = ["ts_recursive_cumulative_freq_neg.pickle",
          "ts_recursive_cumulative_freq_pos.pickle", "ts_recursive_cumulative_freq_neutral.pickle",
          "ts_token_entropy_neg.pickle", "ts_token_entropy_pos.pickle",
          "ts_token_entropy_neutral.pickle"]

    matlab_output = dict()
    matlab_output_fwd = dict()
    matlab_sentiments = dict()

    # decay rates
    neg_rate = 0.015
    pos_rate = 0.035
    neu_rate = 0.065


    t0 = time.time()

    for t in ["entropy", "freq"]:
        coins_meta = []
        # unique date keys
        idxs = []
        coins_series = []
        for senti in sentiment:
            series = []
            extra = []
            elem = []
            for top in range(len(kwargs_list)):
                for site in sites:
                    coin = kwargs_list[top]
                    ts_name = "{}/{}_{}/".format(DIR, site, coin)
                    try:
                        with open(ts_name + "/timeseries_elements.pickle", "rb") as f:
                            timeseries_elements = pickle.load(f)
                            metadata = pickle.load(f)
                    except FileNotFoundError:
                        print("{} was not found - check ts construction script".format(ts_name
                                                                                          + "/timeseries_elements.pickle"))
                    meta = []
                    ts_elements = []
                    try:
                        if t == "entropy":
                            with open(ts_name+"/" + "ts_token_entropy_" + senti + ".pickle", "rb") as f:
                                ts_data = pickle.load(f)
                        elif t == "freq":
                            with open(ts_name+"/" + "ts_recursive_cumulative_freq_" + senti + ".pickle", "rb") as f:
                                ts_data = pickle.load(f)
                        series.extend(ts_data)

                        with open(ts_name+"/" + "ts_recursive_cumulative_freq_" + senti + "_metadata.pickle", "rb") as g:
                            ts_elements = pickle.load(g)
                            meta = pickle.load(g)
                        if len(ts_elements) > 0:
                            telements = ts_elements
                            metadata = meta
                        else:
                            telements = timeseries_elements
                        extra.extend(metadata)
                        elem.extend(telements)

                    except FileNotFoundError:
                        print("{} was not found.".format(ts_name+"/"+t))
                        continue

            summary_ts, summary_meta, time_batches = get_timeseries_summary_per_day(series, elem, extra, summary="IQR")
            idxs.extend([sk for sk in summary_ts.keys() if sk not in idxs])
            coins_series.append(summary_ts)
            coins_meta.append(summary_meta)

            with open(DIR_out + "/" + t + "_" + senti + "_pre_join_data.pickle", "wb") as f:
                pickle.dump(summary_ts, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(summary_meta, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(time_batches, f, pickle.HIGHEST_PROTOCOL)

            matlab_pos_meta = [i['number of ngrams'] for i in list(summary_meta.values())]
            matlab_time = list(summary_meta.keys())

            gap_idx = get_timestamp_gaps_dates(matlab_time, DIR_out + "/{}_".format(t) + senti + "_gaps.html")
            gap_indices = [i[0] for i in gap_idx]

            matlab_fwd = []
            matlab_meta_fwd = []
            matlab_time_fwd = []

            for i in range(len(matlab_time)):
                if i in gap_indices:
                    matlab_fwd.extend(gap_idx[gap_indices.index(i)][1]*[summary_ts[matlab_time[i]]])
                    matlab_meta_fwd.extend(gap_idx[gap_indices.index(i)][1] * [0])
                    # from 0 as we also need idxs[i]
                    for j in range(0, gap_idx[gap_indices.index(i)][1], 1):
                        matlab_time_fwd.append(str(matlab_time[i] + np.timedelta64(j, 'D')))
                else:
                    matlab_fwd.append(summary_ts[matlab_time[i]])
                    matlab_meta_fwd.append(summary_meta[matlab_time[i]]['number of ngrams'])
                    matlab_time_fwd.append(str(matlab_time[i]))

            matlab_sentiments[t + "_" + senti] = matlab_fwd
            matlab_sentiments[t + "_" + senti + "_meta"] = matlab_meta_fwd
            matlab_sentiments[t + "_" + senti + "_time"] = matlab_time_fwd

        savemat(DIR_out + "/sentiments_matlab.mat", matlab_sentiments)

        # sorting and weighting
        total_summaries = []
        neg_past = []
        pos_past = []
        neu_past = []
        neg_weights = []
        pos_weights = []
        neu_weights = []
        neg_decay = np.exp(-neg_rate * np.arange(0, len(idxs), 1))
        pos_decay = np.exp(-pos_rate * np.arange(0, len(idxs), 1))
        neu_decay = np.exp(-neu_rate * np.arange(0, len(idxs), 1))
        # sequence: "pos", "neg", "neutral"
        idxs = sorted(idxs)
        for j in range(len(idxs)):
            i = idxs[j]
            total_sum = 0
            try:
                # pos
                pos = coins_series[0][i]
                pos_vol = coins_meta[0][i]['number of ngrams']
            except:
                pos = 0
                pos_vol = 0
            pos_past.append(pos)
            try:
                # neg
                neg = coins_series[1][i]
                neg_vol = coins_meta[1][i]['number of ngrams']
            except:
                neg = 0
                neg_vol = 0
            neg_past.append(neg)
            try:
                # neu
                neu = coins_series[2][i]
                neu_vol = coins_meta[2][i]['number of ngrams']
            except:
                neu = 0
                neu_vol = 0
            neu_past.append(neu)

            p = convolve(pos_past, pos_decay[:j+1])
            pos_weights.append(p)
            ne = convolve(neg_past, neg_decay[:j+1])
            neg_weights.append(ne)
            nu = convolve(neu_past, neu_decay[:j+1])
            neu_weights.append(nu)

            total_summaries.append((pos_vol*p + neg_vol*ne + neu_vol*nu)/(pos_vol + neg_vol + neu_vol))

        with open(DIR_out+"/" + t + "_total.pickle", "wb") as f:
            pickle.dump(total_summaries, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(idxs, f, pickle.HIGHEST_PROTOCOL)

        matlab_output[t + "_total"] = total_summaries
        matlab_output[t + "_time"] = [str(i) for i in idxs]

        gap_idx = get_timestamp_gaps_dates(idxs, DIR_out + "/" + t + "_gaps.html")
        gap_indices = [i[0] for i in gap_idx]
        total_summaries_fwd = []
        idxss = []
        pos_weights_extend = []
        neg_weights_extend = []
        neu_weights_extend = []
        ftxt = open(DIR_out + "/gap_dates.txt", "wt")

        for i in range(len(idxs)):
            if i in gap_indices:
                total_summaries_fwd.extend(gap_idx[gap_indices.index(i)][1]*[total_summaries[i]])
                # for gap days, append zeros when there's a gap
                pos_weights_extend.extend(gap_idx[gap_indices.index(i)][1]*[0])
                neg_weights_extend.extend(gap_idx[gap_indices.index(i)][1]*[0])
                neu_weights_extend.extend(gap_idx[gap_indices.index(i)][1]*[0])
                # from 0 as we also need idxs[i]
                for j in range(0, gap_idx[gap_indices.index(i)][1], 1):
                    idxss.append(idxs[i] + np.timedelta64(j, 'D'))
                    # for j = 0 we have the last day with non zero counts before the gap
                    if j > 0:
                       ftxt.write(str(idxs[i] + np.timedelta64(j, 'D')) + "\n")
            else:
                pos_weights_extend.append(pos_weights[i])
                neg_weights_extend.append(neg_weights[i])
                neu_weights_extend.append(neu_weights[i])
                total_summaries_fwd.append(total_summaries[i])
                idxss.append(idxs[i])
        ftxt.close()
        if not os.path.isfile(DIR_out + "/sentiment_decays_may6.pickle"):
            with open(DIR_out + "/sentiment_decays_may6.pickle", "wb") as f:
                pickle.dump(pos_weights_extend, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(neg_weights_extend, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(neu_weights_extend, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(idxss, f, pickle.HIGHEST_PROTOCOL)

        with open(DIR_out + "/" + t + "_total_fwd.pickle", "wb") as f:
                pickle.dump(total_summaries_fwd, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(idxss, f, pickle.HIGHEST_PROTOCOL)

        matlab_output_fwd[t + "_total"] = total_summaries_fwd
        matlab_output_fwd[t + "_time"]  = [str(i) for i in idxss]
        matlab_output_fwd[t + "_gaps"]  = gap_idx

        gap_idx = get_timestamp_gaps_dates(idxss, DIR_out + "/" + t + "_gaps_after.html")


    savemat(DIR_out + "/summaries_matlab.mat", matlab_output)
    savemat(DIR_out + "/summaries_matlab_carry_fwd.mat", matlab_output_fwd)

    t1 = time.time()
    print("Time to completion: "+str(datetime.timedelta(seconds=t1-t0)))
