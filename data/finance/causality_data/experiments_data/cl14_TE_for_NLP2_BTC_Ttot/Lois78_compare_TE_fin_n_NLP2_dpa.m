function [TE_XY, TE_YX, TEp_XY, TEp_YX]...
                 = ...
         Lois78_compare_TE_fin_n_NLP2_dpa(lag, pairs, if_returns, testing_mode)


%##############################################################################
% "Sentiment-driven statistical causality in multimodal systems"
%
%  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
%
%  Anna Zaremba
%  April 2021
%
%##############################################################################

%% all of the changing ones -- from external loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lag=1;
% pairs = 12;
% if_returns = 0; 
% if_caus_in_mean = 0;
% testing_mode = 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

path(path,'./gpml-matlab-v3.5-2014-12-08')
path(path,'./gpml-matlab-v3.5-2014-12-08/util')
path(path,'./gpml-matlab-v3.5-2014-12-08/mean')
path(path,'./gpml-matlab-v3.5-2014-12-08/cov')
path(path,'./gpml-matlab-v3.5-2014-12-08/lik')
path(path,'./gpml-matlab-v3.5-2014-12-08/inf')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%



load('Lois62_NLP_fin_data_sentiment2_decay_per_asset.mat');

labels_cryptodata = ['BTC';	'ETH';	'LTC';	'TRX';	'FNG';	'XRP'];

% sentiment_order = {'tot','pos','neu','neg'};
labels_NLP = ['Ttot';'Tpos';'Tneu';'Tneg';'Ctot';'Cpos';'Cneu';'Cneg'];

% prices and returns %%%%%%%%%%%%%%%%%%%%%%%%%%%

prices     = vals_cryptodata;
returns    = diff(vals_cryptodata);
if if_returns
    data_used_fin = returns;
    data_used_NLP = [vals_token_entropy(2:end,:),vals_recum_freq(2:end,:)];
    name_data = 'ret';
else
    data_used_fin = prices;
    data_used_NLP = [vals_token_entropy,vals_recum_freq];
    name_data = 'prc';
end

pairs_string = num2str(pairs);
pair_i1  = str2double(pairs_string(1));
pair_i2  = str2double(pairs_string(2));

name_p1  = labels_cryptodata(pair_i1,:);
name_p2  = labels_NLP(pair_i2,:);
%%

% I want sliding windows --> roughly half a year each windows
% moving by 7 day
window_length = 91; % that's roughly half a year

% check the length of the data, if different length, just adjust to the
% shorter one
index_starter_d1 = find(vals_cryptodata_with_zeros(:,pair_i1)>0,1,'first');

if index_starter_d1>1    
    data_used_fin    = data_used_fin(index_starter_d1:end,:);
    data_used_NLP    = data_used_NLP(index_starter_d1:end,:);
end
data_length  = max(size(data_used_fin));
   
%
dates_points = [1: 7: data_length-window_length-1];
dates_points_nr = length(dates_points);

name = ['NLP2_',name_p1,'_',name_p2,'_',name_data,'_lag',num2str(lag)]
        for t_i = 1:dates_points_nr

            dates_start  = dates_points(t_i);
            dates_end    = dates_start + window_length -1;

            data(:,1)    = zscore(data_used_fin(dates_start:dates_end,pair_i1));
            data(:,2)    = zscore(data_used_NLP(dates_start:dates_end,pair_i2));
            Data_cuts(:,:,t_i) = data;
        end
        
if testing_mode
    to_loop=3
else
    to_loop = dates_points_nr;
end

                                         
                         
[TE_XY( pair_i1,pair_i2,:) , TE_YX( pair_i1,pair_i2,:), ...
 TEp_XY(pair_i1,pair_i2,:) , TEp_YX(pair_i1,pair_i2,:)] ...
                           = ...
 test1_TE_data1_like_structure_forLois57...
        (Data_cuts,1, to_loop,1, name, lag);
