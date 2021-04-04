%##############################################################################
% "Sentiment-driven statistical causality in multimodal systems"
%
%  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
%
%  Anna Zaremba
%  April 2021
%
%##############################################################################

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%% combine TE and GPC results in 1 plot    
%  lag 7


% assume I'm already in the appropriate folder:
% \cl14_TE_for_NLP2_BTC_Ttot


% the test statistic for TE is here:
load('test1_TEbin_NLP2_BTC_Ttot_prc_lag7_len91_from1_to92_time608295',...
                           'pval_XY','pval_YX')
pval_BTC_NLP2_lag7_TE = pval_XY;
pval_NLP2_BTC_lag7_TE = pval_YX;

% assume the structure is as I've sent it, change if needed:
% the test statistic for NLP2 is here:
load('./cl08_NLP2_fin_BTC_Ttot/test1_optim_predmlin_Matern_BTC_Ttot_prc_meancov_lag7_SENTIMENT2_dpa_len91_from1_to92_time502707',...
                           'XY_chi2cdf_vec','YX_chi2cdf_vec')
pval_BTC_NLP2_lag7 = XY_chi2cdf_vec;
pval_NLP2_BTC_lag7 = YX_chi2cdf_vec;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('Lois62_NLP_fin_data_sentiment2_decay_per_asset.mat')

window_length = 91; % that's roughly half a year
data_length  = length(dates_cryptodata);
dates_points = [1: 7: data_length-window_length-1];
dates_points_nr = length(dates_points);

window_dates = dates_cryptodata(dates_points+window_length);
C = window_dates([1,30,60,length(window_dates)]);


X1 = [1:92];
figure();

            plotXY7_GPC = csaps(X1,pval_BTC_NLP2_lag7, 0.5);
            plotYX7_GPC = csaps(X1,pval_NLP2_BTC_lag7, 0.5);  
            
            plotXY7_TE = csaps(X1,pval_BTC_NLP2_lag7_TE, 0.5);
            plotYX7_TE = csaps(X1,pval_NLP2_BTC_lag7_TE, 0.5);                 

            
subplot(1,2,1);
fnplt(plotXY7_GPC);     hold on;   
fnplt(plotXY7_TE);      
    xlim([1 92]);    ylim([0 1])
    title('1-pvalue, BTC --> Entropy NLP2, 7th lag')
    legend('GPC', 'TE')
    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',12)

    

subplot(1,2,2);
fnplt(plotYX7_GPC);     hold on;   
fnplt(plotYX7_TE);      
    xlim([1 92]);    ylim([0 1])
    title('1-pvalue, Entropy NLP2 --> BTC, 7th lag')
    legend('GPC', 'TE')
    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',12)
    
    
    

