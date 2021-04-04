
%##############################################################################
% "Sentiment-driven statistical causality in multimodal systems"
%
%  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
%
%  Anna Zaremba
%  April 2021
%
%##############################################################################

load('conv_struct_lin_Matern_NLP4_BTC_Ctot_lag7_len91_from1_to92.mat')
       
% yes, I know that typically NLP would be denotes as Y
% but here the results use the name "_NLP4_BTC_" which is automatically
% given based on the order of the data inputed
pval_BTC_NLP4_lag7 = YX_chi2cdf_vec;
pval_NLP4_BTC_lag7 = XY_chi2cdf_vec;

load('conv_struct_lin_Matern_NLP2_BTC_Ctot_lag7_len91_from1_to92.mat',...
                           'XY_chi2cdf_vec','YX_chi2cdf_vec')
pval_BTC_NLP2_lag7 = XY_chi2cdf_vec;
pval_NLP2_BTC_lag7 = YX_chi2cdf_vec;

    
    load('Lois62_NLP_fin_data_sentiment2_decay_per_asset.mat')
    window_length = 91; % that's roughly half a year
    data_length  = length(dates_cryptodata);
    dates_points = [1: 7: data_length-window_length-1];
    dates_points_nr = length(dates_points);

    points = [1:dates_points_nr];
    
    window_dates = dates_cryptodata(dates_points+window_length);
    % note, here NLP4 was used as X (and X1, X2, X3)
    % BTC is Y, so if I want to compare to others, I'll put BTC->NLP first,
    % which is Y->X first
        % smooth_on
%         for jj = 1:3
%             load(data_file{jj})

figure()

            plotN2B_NLP4 = csaps(points,pval_NLP4_BTC_lag7, 0.5);
            plotB2N_NLP4 = csaps(points,pval_BTC_NLP4_lag7, 0.5);  
            
            plotN2B_NLP2 = csaps(points,pval_NLP2_BTC_lag7, 0.5);
            plotB2N_NLP2 = csaps(points,pval_BTC_NLP2_lag7, 0.5);            

            subplot(1,2,1); hold on;
            fnplt(plotB2N_NLP4, '--'); hold on  
box on
            fnplt(plotB2N_NLP2); 

            subplot(1,2,2); hold on;
            fnplt(plotN2B_NLP4, '--'); hold on
box on
            fnplt(plotN2B_NLP2); 

%         end

        subplot(1,2,1); hold on;
    %     legend('prc meancov', 'prc mean', 'ret meancov', 'ret mean')
        xlim([1 dates_points_nr])
        ylim([0 1.05])
        title(['1-pvalue BTC --> NLP; lag ',num2str(lag)])
        legend('NLP4','NLP2')

        set(gca, 'XTick', [1,30,60,length(window_dates)]);
        set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
        set(gca,'FontSize',12)

        subplot(1,2,2); hold on;
    %     legend('prc meancov', 'prc mean', 'ret meancov', 'ret mean')
        xlim([1 dates_points_nr])
        ylim([0 1.05])
        title(['1-pvalue NLP --> BTC; lag ',num2str(lag)])
        legend('NLP4','NLP2')

        set(gca, 'XTick', [1,30,60,length(window_dates)]);
        set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
        set(gca,'FontSize',12)
