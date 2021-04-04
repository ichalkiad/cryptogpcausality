%##############################################################################
% "Sentiment-driven statistical causality in multimodal systems"
%
%  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
%
%  Anna Zaremba
%  April 2021
%
%##############################################################################

% currently for BTC price - sentiment, causality in mean and covariance 

%% plot NLP 1-3 together for one lag
                                      
% NLP 1
% load('./data/finance/causality_data/experiments_data/cl06_NLP1_fin_BTC_Ttot/test1_optim_predmlin_Matern_BTC_Ttot_prc_meancov_lag1_len91_from1_to92_time58767.mat', 'hyperparameters', 'all_data', 'length_data');
% NLP 2
% load('./data/finance/causality_data/experiments_data/cl08_NLP2_fin_BTC_Ttot/test1_optim_predmlin_Matern_BTC_Ttot_prc_meancov_lag1_SENTIMENT2_dpa_len91_from1_to92_time502980.mat', 'hyperparameters', 'all_data', 'length_data');
% NLP 3
load('./data/finance/causality_data/experiments_data/cl09_NLP3_fin_BTC_Ttot/test1_optim_predmlin_Matern_BTC_T_prc_meancov_lag1_SENTIMENT3_dps_len91_from1_to92_time548420.mat', 'hyperparameters', 'all_data', 'length_data');


load('Lois31_NLP_financial_data_aligned_lag.mat', 'dates_cryptodata', 'dates_sentiment')

floor_switch = true
covf = 'Matern'
causal_placement_switch = 'all'
from_loop = 1
to_loop = 92
meanf = 'lin'
likf = 'Gauss'
lag = 7
name = 'NLP3_BTC_Ctot_lag7'

% mkdir . convolutional_structure

[Causality_XY_vec, Causality_YX_vec, XY_chi2cdf_vec, YX_chi2cdf_vec] = ...
         causal_testing_with_covariance_structure(name, all_data, hyperparameters, dates_cryptodata, dates_sentiment, length_data, from_loop, to_loop,...
                                          meanf, likf, lag, floor_switch, covf, causal_placement_switch)
lag = 1
% sentiment index
senti_set = ['NLP1'; 'NLP2'; 'NLP3']

counter = 0;

temp_cXY = zeros(92,size(senti_set,1));
temp_cYX = zeros(92,size(senti_set,1));
temp_chiXY = zeros(92,size(senti_set,1));
temp_chiYX = zeros(92,size(senti_set,1));

repeating_name_part1 =  './data/finance/convolutional_structure/conv_struct_lin_Matern_';
repeating_name_part2 =  '_BTC_Ctot_lag7_len91_from1_to92.mat';

for si = 1:size(senti_set,1)
    
    counter = counter+1;

    file_name_pre = [repeating_name_part1,senti_set(si,1:4), repeating_name_part2]
    try load(file_name_pre, 'length_data',...
        'Causality_XY_vec','Causality_YX_vec',...
        'XY_chi2cdf_vec','YX_chi2cdf_vec')
    catch 
        disp('beep!');
        file_name_pre
    end

    points_nr = length(Causality_XY_vec);
    start_nr  = 92-points_nr+1;
    temp_cXY(start_nr:end,counter)=Causality_XY_vec;
    temp_cYX(start_nr:end,counter)=Causality_YX_vec;
    temp_chiXY(start_nr:end,counter)=XY_chi2cdf_vec;
    temp_chiYX(start_nr:end,counter)=YX_chi2cdf_vec;
    
end


% plot smoothed test statistic


window_length = 91; % that's roughly half a year
data_length  = length(dates_cryptodata);
dates_points = [1: 7: data_length-window_length-1];
dates_points_nr = length(dates_points);

window_dates = dates_cryptodata(dates_points+window_length);
C = window_dates([1,30,60,length(window_dates)]);

labels_cryptodata = ['BTC';	'ETH';	'LTC';	'TRX';	'FNG';	'XRP'];
labels_NLP = ['Ttot';'Tpos';'Tneu';'Tneg';'Ctot';'Cpos';'Cneu';'Cneg'];

% choose all crypto assets, no FnG, entropy sentiment construction
for pairs_string_which = [15]

    pairs_string_w  = num2str(pairs_string_which);
    pair_i1_which  = str2double(pairs_string_w(1));
    pair_i2_which  = str2double(pairs_string_w(2));

    name_p1_which  = labels_cryptodata(pair_i1_which,:);
    name_p2_which  = labels_NLP(pair_i2_which,:);
    figure();
    for si = 1:size(senti_set,1)
       
        if si==1
            linespec = '-.';
            color = '#0072BD';
        elseif si==2
            linespec = '-';
            color =  '#D95319';
        elseif si==3
            linespec = '-';
            color = '#EDB120';
        end
        
        sub_chiXY    = temp_chiXY(:, si);
        sub_chiYX    = temp_chiYX(:, si);

        smooth_on = 1;
        X1 = [1:92];
        
        if smooth_on
             plotXY = csaps(X1,sub_chiXY(:), 0.5);
             plotYX = csaps(X1,sub_chiYX(:), 0.5);
             
             subplot(1,2,1)
             fnplt(plotXY, 2, linespec); hold on

             subplot(1,2,2)
             fnplt(plotYX, 2, linespec); hold on
        else
            subplot(1,2,1)
            plot(sub_chiXY(:))
            subplot(1,2,2)
            plot(sub_chiYX(:))
        end
    end
    subplot(1,2,1)
    legend('NLP1', 'NLP2', 'NLP3'); 
    xlim([1 92])
    ylim([0 1])
    title(['chiXY ', name_p1_which, ' ', name_p2_which,' lag ', num2str(lag)],'fontweight','bold')

    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',14)

    subplot(1,2,2)
    legend('NLP1', 'NLP2', 'NLP3'); 
    xlim([1 92])
    ylim([0 1])
    title(['chiYX ', name_p1_which, ' ', name_p2_which,' lag ', num2str(lag)],'fontweight','bold')

    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',14)
    
end


%% plot NLP2 and Hash Rate, all lags


% lag 1
load('./data/finance/causality_data/experiments_data/cl10_HR_NLP2_Ttot/test1_optim_predmlin_Matern_HR_Ttot_HR_meancov_lag1_SENTIMENT2_dpa_len91_from1_to92_time134880.mat', 'hyperparameters', 'all_data', 'length_data');
% lag 7
%load('./data/finance/causality_data/experiments_data/cl10_HR_NLP2_Ttot/test1_optim_predmlin_Matern_HR_Ttot_HR_meancov_lag7_SENTIMENT2_dpa_len91_from1_to92_time134880.mat', 'hyperparameters', 'all_data', 'length_data');
% lag 30
%load('./data/finance/causality_data/experiments_data/cl10_HR_NLP2_Ttot/test1_optim_predmlin_Matern_HR_Ttot_HR_meancov_lag30_SENTIMENT2_dpa_len91_from1_to92_time134932.mat', 'hyperparameters', 'all_data', 'length_data');

% NEEDED?? If so, ask from Zaremba
%load('Lois67_hash_rate_data');
%load('Lois62_NLP_fin_data_sentiment2_decay_per_asset.mat');
load('Lois31_NLP_financial_data_aligned_lag.mat', 'dates_cryptodata', 'dates_sentiment')

floor_switch = true
covf = 'Matern'
causal_placement_switch = 'all'
from_loop = 1
to_loop = 92
meanf = 'lin'
likf = 'Gauss'
lag = 1
name = 'NLP2_HR_Ctot_lag1'

% mkdir . convolutional_structure

[Causality_XY_vec, Causality_YX_vec, XY_chi2cdf_vec, YX_chi2cdf_vec] = ...
         causal_testing_with_covariance_structure(name, all_data, hyperparameters, dates_cryptodata, dates_sentiment, length_data, from_loop, to_loop,...
                                          meanf, likf, lag, floor_switch, covf, causal_placement_switch)
                                      

                                      

% lag 7
load('./data/finance/causality_data/experiments_data/cl10_HR_NLP2_Ttot/test1_optim_predmlin_Matern_HR_Ttot_HR_meancov_lag7_SENTIMENT2_dpa_len91_from1_to92_time134880.mat', 'hyperparameters', 'all_data', 'length_data');

load('Lois31_NLP_financial_data_aligned_lag.mat', 'dates_cryptodata', 'dates_sentiment')

floor_switch = true
covf = 'Matern'
causal_placement_switch = 'all'
from_loop = 1
to_loop = 92
meanf = 'lin'
likf = 'Gauss'
lag = 7
name = 'NLP2_HR_Ctot_lag7'

% mkdir . convolutional_structure

[Causality_XY_vec, Causality_YX_vec, XY_chi2cdf_vec, YX_chi2cdf_vec] = ...
         causal_testing_with_covariance_structure(name, all_data, hyperparameters, dates_cryptodata, dates_sentiment, length_data, from_loop, to_loop,...
                                          meanf, likf, lag, floor_switch, covf, causal_placement_switch)                                      

                                      

% lag 30
load('./data/finance/causality_data/experiments_data/cl10_HR_NLP2_Ttot/test1_optim_predmlin_Matern_HR_Ttot_HR_meancov_lag30_SENTIMENT2_dpa_len91_from1_to92_time134932.mat', 'hyperparameters', 'all_data', 'length_data');

load('Lois31_NLP_financial_data_aligned_lag.mat', 'dates_cryptodata', 'dates_sentiment')

floor_switch = true
covf = 'Matern'
causal_placement_switch = 'all'
from_loop = 1
to_loop = 92
meanf = 'lin'
likf = 'Gauss'
lag = 30
name = 'NLP2_HR_Ctot_lag30'

% mkdir . convolutional_structure

[Causality_XY_vec, Causality_YX_vec, XY_chi2cdf_vec, YX_chi2cdf_vec] = ...
         causal_testing_with_covariance_structure(name, all_data, hyperparameters, dates_cryptodata, dates_sentiment, length_data, from_loop, to_loop,...
                                          meanf, likf, lag, floor_switch, covf, causal_placement_switch)             
                                      
                                      
                                      
                                      
                                      
lag_set = [1, 7, 30];
counter = 0;

temp_cXY = zeros(92,size(lag_set,2));
temp_cYX = zeros(92,size(lag_set,2));
temp_chiXY = zeros(92,size(lag_set,2));
temp_chiYX = zeros(92,size(lag_set,2));

repeating_name_part1 =  './data/finance/convolutional_structure/conv_struct_lin_Matern_NLP2_HR_Ctot_lag';
repeating_name_part2 =  '_len91_from1_to92.mat';

for si = 1:size(lag_set,2)
    
    counter = counter+1;

    file_name_pre = [repeating_name_part1, num2str(lag_set(si)), repeating_name_part2]
    try load(file_name_pre, 'length_data',...
        'Causality_XY_vec','Causality_YX_vec',...
        'XY_chi2cdf_vec','YX_chi2cdf_vec')
    catch 
        disp('beep!');
        file_name_pre
    end

    points_nr = length(Causality_XY_vec);
    start_nr  = 92-points_nr+1;
    temp_cXY(start_nr:end,counter)=Causality_XY_vec;
    temp_cYX(start_nr:end,counter)=Causality_YX_vec;
    temp_chiXY(start_nr:end,counter)=XY_chi2cdf_vec;
    temp_chiYX(start_nr:end,counter)=YX_chi2cdf_vec;
    
end


% plot smoothed test statistic


window_length = 91; % that's roughly half a year
data_length  = length(dates_cryptodata);
dates_points = [1: 7: data_length-window_length-1];
dates_points_nr = length(dates_points);

window_dates = dates_cryptodata(dates_points+window_length);
C = window_dates([1,30,60,length(window_dates)]);

labels_cryptodata = ['BTC';	'ETH';	'LTC';	'TRX';	'FNG';	'XRP'];
labels_NLP = ['Ttot';'Tpos';'Tneu';'Tneg';'Ctot';'Cpos';'Cneu';'Cneg'];

% choose all crypto assets, no FnG, entropy sentiment construction
for pairs_string_which = [15]

    pairs_string_w  = num2str(pairs_string_which);
    pair_i1_which  = str2double(pairs_string_w(1));
    pair_i2_which  = str2double(pairs_string_w(2));

    name_p1_which  = labels_cryptodata(pair_i1_which,:);
    name_p2_which  = labels_NLP(pair_i2_which,:);
    figure();
    for si = 1:size(lag_set,2)
       
        if si==1
            linespec = '-.';
            color = '#0072BD';
        elseif si==2
            linespec = '-';
            color =  '#D95319';
        elseif si==3
            linespec = '-';
            color = '#EDB120';
        end
        
        sub_chiXY    = temp_chiXY(:, si);
        sub_chiYX    = temp_chiYX(:, si);

        smooth_on = 1;
        X1 = [1:92];
        
        if smooth_on
             plotXY = csaps(X1,sub_chiXY(:), 0.5);
             plotYX = csaps(X1,sub_chiYX(:), 0.5);
             
             subplot(1,2,1)
             fnplt(plotXY, 2, linespec); hold on

             subplot(1,2,2)
             fnplt(plotYX, 2, linespec); hold on
        else
            subplot(1,2,1)
            plot(sub_chiXY(:))
            subplot(1,2,2)
            plot(sub_chiYX(:))
        end
    end
    subplot(1,2,1)
    legend('lag 1', 'lag 7', 'lag 30'); 
    xlim([1 92])
    ylim([0 1])
    title('1-pvalue HR --> Entropy NLP2','fontweight','bold')

    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',14)

    subplot(1,2,2)
    legend('lag 1', 'lag 7', 'lag 30'); 
    xlim([1 92])
    ylim([0 1])
    title('1-pvalue Entropy NLP2 --> HR','fontweight','bold')

    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',14)
    
end



%% plot NLP2 and Transfer Entropy, lags 1 and 7

% lag 1
load('./data/finance/causality_data/experiments_data/cl14_TE_for_NLP2_BTC_Ttot/test1_TEbin_NLP2_BTC_Ttot_prc_lag1_len91_from1_to92_time607915.mat');
load('./data/finance/causality_data/experiments_data/cl08_NLP2_fin_BTC_Ttot/test1_optim_predmlin_Matern_BTC_Ttot_prc_meancov_lag1_SENTIMENT2_dpa_len91_from1_to92_time502980.mat', 'hyperparameters', 'all_data', 'length_data');
% lag 7
% load('./data/finance/causality_data/experiments_data/cl14_TE_for_NLP2_BTC_Ttot/test1_TEbin_NLP2_BTC_Ttot_prc_lag7_len91_from1_to92_time608295.mat');
% load('./data/finance/causality_data/experiments_data/cl08_NLP2_fin_BTC_Ttot/test1_optim_predmlin_Matern_BTC_Ttot_prc_meancov_lag7_SENTIMENT2_dpa_len91_from1_to92_time502707.mat', 'hyperparameters', 'all_data', 'length_data');
load('Lois31_NLP_financial_data_aligned_lag.mat', 'dates_cryptodata', 'dates_sentiment')


floor_switch = true
covf = 'Matern'
causal_placement_switch = 'all'
from_loop = 1
to_loop = 92
meanf = 'lin'
likf = 'Gauss'
lag = 1
name = 'NLP2_TE_Ctot_lag1'

% mkdir . convolutional_structure

[Causality_XY_vec, Causality_YX_vec, XY_chi2cdf_vec, YX_chi2cdf_vec] = ...
         causal_testing_with_covariance_structure(name, all_data, hyperparameters, dates_cryptodata, dates_sentiment, length_data, from_loop, to_loop,...
                                          meanf, likf, lag, floor_switch, covf, causal_placement_switch)
                                      
                                      

lag_set = [lag];
counter = 0;

temp_cXY = zeros(92,size(lag_set,2));
temp_cYX = zeros(92,size(lag_set,2));
temp_chiXY = zeros(92,size(lag_set,2));
temp_chiYX = zeros(92,size(lag_set,2));

repeating_name_part1 =  './data/finance/convolutional_structure/conv_struct_lin_Matern_NLP2_TE_Ctot_lag';
repeating_name_part2 =  '_len91_from1_to92.mat';

for si = 1:size(lag_set,2)
    
    counter = counter+1;

    file_name_pre = [repeating_name_part1, num2str(lag_set(si)), repeating_name_part2]
    try load(file_name_pre, 'length_data',...
        'Causality_XY_vec','Causality_YX_vec',...
        'XY_chi2cdf_vec','YX_chi2cdf_vec')
    catch 
        disp('beep!');
        file_name_pre
    end

    points_nr = length(Causality_XY_vec);
    start_nr  = 92-points_nr+1;
    temp_cXY(start_nr:end,counter)=Causality_XY_vec;
    temp_cYX(start_nr:end,counter)=Causality_YX_vec;
    temp_chiXY(start_nr:end,counter)=XY_chi2cdf_vec;
    temp_chiYX(start_nr:end,counter)=YX_chi2cdf_vec;
    
end


% plot smoothed test statistic


window_length = 91; % that's roughly half a year
data_length  = length(dates_cryptodata);
dates_points = [1: 7: data_length-window_length-1];
dates_points_nr = length(dates_points);

window_dates = dates_cryptodata(dates_points+window_length);
C = window_dates([1,30,60,length(window_dates)]);

labels_cryptodata = ['BTC';	'ETH';	'LTC';	'TRX';	'FNG';	'XRP'];
labels_NLP = ['Ttot';'Tpos';'Tneu';'Tneg';'Ctot';'Cpos';'Cneu';'Cneg'];

% choose all crypto assets, no FnG, entropy sentiment construction
for pairs_string_which = [15]

    pairs_string_w  = num2str(pairs_string_which);
    pair_i1_which  = str2double(pairs_string_w(1));
    pair_i2_which  = str2double(pairs_string_w(2));

    name_p1_which  = labels_cryptodata(pair_i1_which,:);
    name_p2_which  = labels_NLP(pair_i2_which,:);
    
    for si = 1:size(lag_set,2)
        figure();
        if si==1
            linespec = '-.';
            color = '#0072BD';
        elseif si==2
            linespec = '-';
            color =  '#D95319';
        elseif si==3
            linespec = '-';
            color = '#EDB120';
        end
        
        sub_chiXY    = temp_chiXY(:, si);
        sub_chiYX    = temp_chiYX(:, si);

        smooth_on = 1;
        X1 = [1:92];
        
        if smooth_on
             plotXY = csaps(X1,sub_chiXY(:), 0.5);
             plotYX = csaps(X1,sub_chiYX(:), 0.5);             
             
             subplot(1,2,1)
             fnplt(plotXY, 2, linespec); hold on

             subplot(1,2,2)
             fnplt(plotYX, 2, linespec); hold on
             
             % TE
             plotTEXY = csaps(X1,pval_XY, 0.5);
             plotTEYX = csaps(X1,pval_YX, 0.5);             
             
             subplot(1,2,1)
             fnplt(plotTEXY, 2, '-', 'r'); hold on

             subplot(1,2,2)
             fnplt(plotTEYX, 2, '-' , 'r'); hold on
             
        else
            subplot(1,2,1)
            plot(sub_chiXY(:))
            subplot(1,2,2)
            plot(sub_chiYX(:))
        end

        subplot(1,2,1)
        legend('GPC', 'TE'); 
        xlim([1 92])
        ylim([0 1])
        title(['1-pvalue, BTC --> Entropy NLP2, lag', num2str(lag_set(si))],'fontweight','bold')

        set(gca, 'XTick', [1,30,60,length(window_dates)]);
        set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
        set(gca,'FontSize',14)

        subplot(1,2,2)
        legend('GPC', 'TE'); 
        xlim([1 92])
        ylim([0 1])
        title(['1-pvalue, Entropy NLP2 --> BTC, lag', num2str(lag_set(si))],'fontweight','bold')

        set(gca, 'XTick', [1,30,60,length(window_dates)]);
        set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
        set(gca,'FontSize',14)
    end
    
end



%% plot NLP2 for all lags

load('Lois31_NLP_financial_data_aligned_lag.mat', 'dates_cryptodata', 'dates_sentiment')

% lag 1
% load('./data/finance/causality_data/experiments_data/cl08_NLP2_fin_BTC_Ttot/test1_optim_predmlin_Matern_BTC_Ttot_prc_meancov_lag1_SENTIMENT2_dpa_len91_from1_to92_time502980.mat', 'hyperparameters', 'all_data', 'length_data');
% lag 7
% load('./data/finance/causality_data/experiments_data/cl08_NLP2_fin_BTC_Ttot/test1_optim_predmlin_Matern_BTC_Ttot_prc_meancov_lag7_SENTIMENT2_dpa_len91_from1_to92_time502707.mat', 'hyperparameters', 'all_data', 'length_data');
% lag 30
load('./data/finance/causality_data/experiments_data/cl08_NLP2_fin_BTC_Ttot/test1_optim_predmlin_Matern_BTC_Ttot_prc_meancov_lag30_SENTIMENT2_dpa_len91_from1_to92_time502933.mat', 'hyperparameters', 'all_data', 'length_data');

floor_switch = true
covf = 'Matern'
causal_placement_switch = 'all'
from_loop = 1
to_loop = 92
meanf = 'lin'
likf = 'Gauss'
lag = 30
name = 'NLP2_BTC_Ctot_lag30'

% mkdir . convolutional_structure

[Causality_XY_vec, Causality_YX_vec, XY_chi2cdf_vec, YX_chi2cdf_vec] = ...
         causal_testing_with_covariance_structure(name, all_data, hyperparameters, dates_cryptodata, dates_sentiment, length_data, from_loop, to_loop,...
                                          meanf, likf, lag, floor_switch, covf, causal_placement_switch)
                                        
                                      
lag_set = [1, 7, 30];
counter = 0;

temp_cXY = zeros(92,size(lag_set,2));
temp_cYX = zeros(92,size(lag_set,2));
temp_chiXY = zeros(92,size(lag_set,2));
temp_chiYX = zeros(92,size(lag_set,2));

repeating_name_part1 =  './data/finance/convolutional_structure/conv_struct_lin_Matern_NLP2_BTC_Ctot_lag';
repeating_name_part2 =  '_len91_from1_to92.mat';

for si = 1:size(lag_set,2)
    
    counter = counter+1;

    file_name_pre = [repeating_name_part1, num2str(lag_set(si)), repeating_name_part2]
    try load(file_name_pre, 'length_data',...
        'Causality_XY_vec','Causality_YX_vec',...
        'XY_chi2cdf_vec','YX_chi2cdf_vec')
    catch 
        disp('beep!');
        file_name_pre
    end

    points_nr = length(Causality_XY_vec);
    start_nr  = 92-points_nr+1;
    temp_cXY(start_nr:end,counter)=Causality_XY_vec;
    temp_cYX(start_nr:end,counter)=Causality_YX_vec;
    temp_chiXY(start_nr:end,counter)=XY_chi2cdf_vec;
    temp_chiYX(start_nr:end,counter)=YX_chi2cdf_vec;
    
end


% plot smoothed test statistic


window_length = 91; % that's roughly half a year
data_length  = length(dates_cryptodata);
dates_points = [1: 7: data_length-window_length-1];
dates_points_nr = length(dates_points);

window_dates = dates_cryptodata(dates_points+window_length);
C = window_dates([1,30,60,length(window_dates)]);

labels_cryptodata = ['BTC';	'ETH';	'LTC';	'TRX';	'FNG';	'XRP'];
labels_NLP = ['Ttot';'Tpos';'Tneu';'Tneg';'Ctot';'Cpos';'Cneu';'Cneg'];

% choose all crypto assets, no FnG, entropy sentiment construction
for pairs_string_which = [15]

    pairs_string_w  = num2str(pairs_string_which);
    pair_i1_which  = str2double(pairs_string_w(1));
    pair_i2_which  = str2double(pairs_string_w(2));

    name_p1_which  = labels_cryptodata(pair_i1_which,:);
    name_p2_which  = labels_NLP(pair_i2_which,:);
    figure();
    for si = 1:size(lag_set,2)
       
        if si==1
            linespec = '-.';
            color = '#0072BD';
        elseif si==2
            linespec = '-';
            color =  '#D95319';
        elseif si==3
            linespec = '-';
            color = '#EDB120';
        end
        
        sub_chiXY    = temp_chiXY(:, si);
        sub_chiYX    = temp_chiYX(:, si);

        smooth_on = 1;
        X1 = [1:92];
        
        if smooth_on
             plotXY = csaps(X1,sub_chiXY(:), 0.5);
             plotYX = csaps(X1,sub_chiYX(:), 0.5);
             
             subplot(1,2,1)
             fnplt(plotXY, 2, linespec); hold on

             subplot(1,2,2)
             fnplt(plotYX, 2, linespec); hold on
        else
            subplot(1,2,1)
            plot(sub_chiXY(:))
            subplot(1,2,2)
            plot(sub_chiYX(:))
        end
    end
    subplot(1,2,1)
    legend('lag 1', 'lag 7', 'lag 30'); 
    xlim([1 92])
    ylim([0 1])
    title('1-pvalue BTC --> Entropy NLP2','fontweight','bold')

    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',14)

    subplot(1,2,2)
    legend('lag 1', 'lag 7', 'lag 30'); 
    xlim([1 92])
    ylim([0 1])
    title('1-pvalue Entropy NLP2 --> BTC','fontweight','bold')

    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',14)
    
end



%% plot NLP2 - NLP4 for lag 7

load('Lois31_NLP_financial_data_aligned_lag.mat', 'dates_cryptodata', 'dates_sentiment')

% NLP2
% lag 7
load('./data/finance/causality_data/experiments_data/cl08_NLP2_fin_BTC_Ttot/test1_optim_predmlin_Matern_BTC_Ttot_prc_meancov_lag7_SENTIMENT2_dpa_len91_from1_to92_time502707.mat', 'hyperparameters', 'all_data', 'length_data');

floor_switch = true
covf = 'Matern'
causal_placement_switch = 'all'
from_loop = 1
to_loop = 92
meanf = 'lin'
likf = 'Gauss'
lag = 7
name = 'NLP2_BTC_Ctot_lag7'

% mkdir . convolutional_structure

[Causality_XY_vec, Causality_YX_vec, XY_chi2cdf_vec, YX_chi2cdf_vec] = ...
         causal_testing_with_covariance_structure(name, all_data, hyperparameters, dates_cryptodata, dates_sentiment, length_data, from_loop, to_loop,...
                                          meanf, likf, lag, floor_switch, covf, causal_placement_switch)
                                        
clear all

load('Lois31_NLP_financial_data_aligned_lag.mat', 'dates_cryptodata', 'dates_sentiment')

% NLP4
% lag 7
load('test_optim_multimod_lin_Matern_NLP4_BTC_prc_meancov_lag7_winL_91_NLP4_correct_composition_len91_from1_to92_time480604.mat')


floor_switch = true
covf = 'Matern'
causal_placement_switch = 'all'
from_loop = 1
to_loop = 92
meanf = 'lin'
likf = 'Gauss'
lag = 7
name = 'NLP4_BTC_Ctot_lag7'

% mkdir . convolutional_structure

[Causality_XY_vec, Causality_YX_vec, XY_chi2cdf_vec, YX_chi2cdf_vec] = ...
         causal_testing_with_covariance_structure(name, all_data, hyperparameters, dates_cryptodata, dates_sentiment, length_data, from_loop, to_loop,...
                                          meanf, likf, lag, floor_switch, covf, causal_placement_switch)
                                        



                          
