%##############################################################################
% "Sentiment-driven statistical causality in multimodal systems"
%
%  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
%
%  Anna Zaremba
%  April 2021
%
%##############################################################################

% covariance in: mean, mean and cov
% prices and returns
% lags: 1 day, 1 week, 1 month

% with FnG as side info and without

% ---> then move to NLP data

function [C_XY, C_YX, chi2_XY, chi2_YX, hyperparameters] ...
                 = ...
         Lois68_GPC_hashRate_n_NLP2_decay_per_asset(lag, pairs, if_returns, ...
                                       if_caus_in_mean, testing_mode)


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



load('Lois67_hash_rate_data')
load('Lois62_NLP_fin_data_sentiment2_decay_per_asset.mat');


% sentiment_order = {'tot','pos','neu','neg'};
labels_NLP = ['Ttot';'Tpos';'Tneu';'Tneg';'Ctot';'Cpos';'Cneu';'Cneg'];

% prices and returns %%%%%%%%%%%%%%%%%%%%%%%%%%%


    data_used_hr = vals_hash_rate;
    data_used_NLP = [vals_token_entropy(2:end,:),vals_recum_freq(2:end,:)];
    name_data = 'HR';

pairs_string = num2str(pairs);
% pair_i1  = 1;
pair_i2  = str2double(pairs_string(2));

name_p1  = 'HR';
name_p2  = labels_NLP(pair_i2,:);
%%

meanf = 'lin';
likf  = 'Gauss';
covf  = 'Matern';


if if_caus_in_mean
    where_cause = 'mean';
    name_cause  = 'mean';
else
    where_cause = 'all';
    name_cause  = 'meancov';
end

% I want sliding windows --> roughly half a year each windows
% moving by 7 day
window_length = 91; % that's roughly half a year

% check the length of the data, if different length, just adjust to the
% shorter one
% index_starter_d1 = find(vals_cryptodata_with_zeros(:,pair_i1)>0,1,'first');

% if index_starter_d1>1    
%     data_used_hr    = data_used_hr(index_starter_d1:end,:);
%     data_used_NLP    = data_used_NLP(index_starter_d1:end,:);
% end
data_length  = max(size(data_used_hr));
   
%
dates_points = [1: 7: data_length-window_length-1];
dates_points_nr = length(dates_points);

name = [name_p1,'_',name_p2,'_',name_data,'_',name_cause, '_lag',num2str(lag),'_SENTIMENT2_dpa']
        for t_i = 1:dates_points_nr

            dates_start  = dates_points(t_i);
            dates_end    = dates_start + window_length -1;

            data(:,1)    = zscore(data_used_hr(dates_start:dates_end,:));
            data(:,2)    = zscore(data_used_NLP(dates_start:dates_end,pair_i2));
            Data_cuts(:,:,t_i) = data;
        end
        
if testing_mode
    to_loop=3
else
    to_loop = dates_points_nr;
end

[C_XY, C_YX, chi2_XY, chi2_YX, hyperparameters] ...
                   = ...        
test1_optim_data1_like_structure_cov_predm_hyp_meancov_correct...
                            (Data_cuts,1, to_loop,...
                             meanf, likf, name, lag, 'true', covf,where_cause);
