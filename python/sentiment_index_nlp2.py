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
    DIR_out = "./data/output_nlp2"

    if not os.path.exists(DIR_out):
        os.makedirs(DIR_out)

    kwargs_list = ["BTC", "ETH", "LTC", "XRP", "TRX"]
    sites = ["cryptoslate", "cryptodaily"]

    ts = ["ts_recursive_cumulative_freq_neg.pickle", "ts_recursive_cumulative_freq.pickle",
          "ts_recursive_cumulative_freq_pos.pickle", "ts_recursive_cumulative_freq_neutral.pickle",
          "ts_token_entropy.pickle", "ts_token_entropy_neg.pickle", "ts_token_entropy_pos.pickle",
          "ts_token_entropy_neutral.pickle"]


    matlab_output = dict()
    matlab_output_fwd = dict()

    # decay rates

    btc_rate = 0.015
    eth_rate = 0.025
    ltc_rate = 0.035
    xrp_rate = 0.045
    trx_rate = 0.065

    t0 = time.time()
    for t in ts:
        coins_meta = []
        # unique date keys
        idxs = []
        coins_series = []
        for top in range(len(kwargs_list)):
            series = []
            extra = []
            elem = []
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
                    with open(ts_name+t, "rb") as f:
                        ts_data = pickle.load(f)
                        if "neg" in ts_name+t or "pos" in ts_name+t or "neutral" in ts_name+t:
                            if "neg" in ts_name+t:
                                senti = "neg"
                            elif "pos" in ts_name+t:
                                senti = "pos"
                            elif "neutral" in ts_name+t:
                                senti = "neutral"
                            with open(ts_name+"ts_recursive_cumulative_freq_" + senti + "_metadata.pickle", "rb") as g:
                                ts_elements = pickle.load(g)
                                meta = pickle.load(g)
                        if len(ts_elements) > 0:
                            telements = ts_elements
                            metadata = meta
                        else:
                            telements = timeseries_elements
                except FileNotFoundError:
                    print("{} was not found.".format(ts_name+t))
                    continue
                series.extend(ts_data)
                extra.extend(metadata)
                elem.extend(telements)

            summary_ts, summary_meta, time_batches = get_timeseries_summary_per_day(series, elem, extra, summary="IQR")
            idxs.extend([sk for sk in summary_ts.keys() if sk not in idxs])
            coins_series.append(summary_ts)
            coins_meta.append(summary_meta)

        with open(DIR_out + "/" + t.replace(".pickle", "_pre_join_data.pickle"), "wb") as f:
            pickle.dump(coins_series, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(coins_meta, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(idxs, f, pickle.HIGHEST_PROTOCOL)

        # sorting and weighting
        total_summaries = []
        total_meta = []
        btc_past = []
        eth_past = []
        ltc_past = []
        xrp_past = []
        trx_past = []
        btc_weights = []
        eth_weights = []
        ltc_weights = []
        xrp_weights = []
        trx_weights = []
        btc_decay = np.exp(-btc_rate*np.arange(0, len(idxs), 1))
        eth_decay = np.exp(-eth_rate * np.arange(0, len(idxs), 1))
        ltc_decay = np.exp(-ltc_rate * np.arange(0, len(idxs), 1))
        xrp_decay = np.exp(-xrp_rate * np.arange(0, len(idxs), 1))
        trx_decay = np.exp(-trx_rate * np.arange(0, len(idxs), 1))
        idxs = sorted(idxs)
        for j in range(len(idxs)):
            i = idxs[j]
            total_sum = 0
            try:
                # BTC val and ngram number
                btc = coins_series[0][i]
                btc_num = coins_meta[0][i]["number of ngrams"]
                btc_sum = coins_meta[0][i]
            except:
                btc = 0
                btc_num = 0
                btc_sum = []
            btc_past.append(btc)
            try:
                # ETH
                eth = coins_series[1][i]
                eth_num = coins_meta[1][i]["number of ngrams"]
                eth_sum = coins_meta[1][i]
            except:
                eth = 0
                eth_num = 0
                eth_sum = []
            eth_past.append(eth)
            try:
                # LTC
                ltc = coins_series[2][i]
                ltc_num = coins_meta[2][i]["number of ngrams"]
                ltc_sum = coins_meta[2][i]
            except:
                ltc = 0
                ltc_num = 0
                ltc_sum = {}
            ltc_past.append(ltc)
            try:
                # XRP
                xrp = coins_series[3][i]
                xrp_num = coins_meta[3][i]["number of ngrams"]
                xrp_sum = coins_meta[3][i]
            except:
                xrp = 0
                xrp_num = 0
                xrp_sum = []
            xrp_past.append(xrp)
            try:
                # TRX
                trx = coins_series[4][i]
                trx_num = coins_meta[4][i]["number of ngrams"]
                trx_sum = coins_meta[4][i]
            except:
                trx = 0
                trx_num = 0
                trx_sum = []
            trx_past.append(trx)

            b = convolve(btc_past, btc_decay[:j+1])
            btc_weights.append(b)
            e = convolve(eth_past, eth_decay[:j+1])
            eth_weights.append(e)
            lt = convolve(ltc_past, ltc_decay[:j+1])
            ltc_weights.append(lt)
            tr = convolve(trx_past, trx_decay[:j+1])
            trx_weights.append(tr)
            xr = convolve(xrp_past, xrp_decay[:j+1])
            xrp_weights.append(xr)

            total_summaries.append(b + e + lt + tr + xr)
            total_meta.append([btc_sum, eth_sum, ltc_sum, xrp_sum, trx_sum])

        with open(DIR_out + "/" + t.replace(".pickle", "_total.pickle"), "wb") as f:
                pickle.dump(total_summaries, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(total_meta, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(idxs, f, pickle.HIGHEST_PROTOCOL)

        matlab_output[t.replace(".pickle", "_total")] = total_summaries
        matlab_output[t.replace(".pickle", "_total_metadata")] = total_meta
        matlab_output[t.replace(".pickle", "_time")] = [str(i) for i in idxs]

        gap_idx = get_timestamp_gaps_dates(idxs, DIR_out + "/" + t.replace(".pickle", "_gaps.html"))
        gap_indices = [i[0] for i in gap_idx]
        total_summaries_fwd = []
        total_meta_fwd = []
        idxss = []
        btc_weights_extend = []
        eth_weights_extend = []
        ltc_weights_extend = []
        xrp_weights_extend = []
        trx_weights_extend = []
        ftxt = open(DIR_out + "/gap_dates.txt", "wt")
        for i in range(len(idxs)):
            if i in gap_indices:
                total_summaries_fwd.extend(gap_idx[gap_indices.index(i)][1]*[total_summaries[i]])
                total_meta_fwd.extend(gap_idx[gap_indices.index(i)][1] * [total_meta[i]])
                # for gap days, append zeros when there's a gap
                btc_weights_extend.extend(gap_idx[gap_indices.index(i)][1]*[0])
                eth_weights_extend.extend(gap_idx[gap_indices.index(i)][1]*[0])
                ltc_weights_extend.extend(gap_idx[gap_indices.index(i)][1]*[0])
                xrp_weights_extend.extend(gap_idx[gap_indices.index(i)][1]*[0])
                trx_weights_extend.extend(gap_idx[gap_indices.index(i)][1]*[0])
                # from 0 as we also need idxs[i]
                for j in range(0, gap_idx[gap_indices.index(i)][1], 1):
                    idxss.append(idxs[i] + np.timedelta64(j, 'D'))
                    # for j = 0 we have the last day with non zero counts before the gap
                    if j > 0:
                       ftxt.write(str(idxs[i] + np.timedelta64(j, 'D')) + "\n")
            else:
                btc_weights_extend.append(btc_weights[i])
                eth_weights_extend.append(eth_weights[i])
                ltc_weights_extend.append(ltc_weights[i])
                xrp_weights_extend.append(xrp_weights[i])
                trx_weights_extend.append(trx_weights[i])
                total_summaries_fwd.append(total_summaries[i])
                total_meta_fwd.append(total_meta[i])
                idxss.append(idxs[i])
        ftxt.close()
        if not os.path.isfile(DIR_out + "/project_decays.pickle"):
            with open(DIR_out + "/project_decays.pickle", "wb") as f:
                pickle.dump(btc_weights_extend, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(eth_weights_extend, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(ltc_weights_extend, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(xrp_weights_extend, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(trx_weights_extend, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(idxss, f, pickle.HIGHEST_PROTOCOL)

        with open(DIR_out + "/" + t.replace(".pickle", "_total_fwd.pickle"), "wb") as f:
                pickle.dump(total_summaries_fwd, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(total_meta_fwd, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(idxss, f, pickle.HIGHEST_PROTOCOL)

        matlab_output_fwd[t.replace(".pickle", "_total")] = total_summaries_fwd
        matlab_output_fwd[t.replace(".pickle", "_total_metadata")] = total_meta_fwd
        matlab_output_fwd[t.replace(".pickle", "_time")] = [str(i) for i in idxss]
        matlab_output_fwd[t.replace(".pickle", "_gaps")] = gap_idx

        gap_idx = get_timestamp_gaps_dates(idxss, DIR_out + "/" + t.replace(".pickle", "_gaps_after.html"))


    savemat(DIR_out + "/summaries_matlab.mat", matlab_output)
    savemat(DIR_out + "/summaries_matlab_carry_fwd.mat", matlab_output_fwd)

    t1 = time.time()
    print("Time to completion: "+str(datetime.timedelta(seconds=t1-t0)))
