%##############################################################################
% "Sentiment-driven statistical causality in multimodal systems"
%
%  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
%
%  Anna Zaremba
%  April 2021
%
%##############################################################################

% parameters I looped through:
% for if_caus_in_mean in 0 1
% 	for if_returns in 0 1
% 		for lag in 1 7 30
% 			for pairs in 11 12 13 14 21 22 23 24 31 32 33 34 15 16 17 18 25 26 27 28 35 36 37 38
% 			for pairs in 41 42 43 44 45 46 47 48 61 62 63 64 65 66 67 68



multiple_results = 0

%%%%%%%%%%%%%%%%%%%
if multiple_results
%%%%%%%%%%%%%%%%%%%


labels_cryptodata = ['HR'];
labels_NLP = ['Ttot';'Tpos';'Tneu';'Tneg';'Ctot';'Cpos';'Cneu';'Cneg'];

repeating_name_part =  'test1_optim_predmlin_Matern_';
% test1_optim_predmlin_Matern_HR_Cneg_HR_meancov_lag1_SENTIMENT2_dpa_len91_from1_to92_time134932.mat
                
                % eval(['save(',char(39), 'test1_optim_predm',meanf,'_',covf,'_',name,...
                %  '_len',num2str(length_data),...
                %  '_from',num2str(from_loop),'_to' ,num2str(to_loop),...
                %  '_time',num2str(starting_time_name),'.mat',char(39), ')']);
   
                % covf = 'Matern'
                % meanf = 'lin'

            

pairs_set = [11 12 13 14 15 16 17 18 ];
lag_set = [1 7 30];
ifret_set = [0];
cm_set = [0];

counter=0;

 temp_cXY = zeros(92,length(pairs_set)*length(lag_set)*length(ifret_set)*length(cm_set));
 temp_cYX = zeros(92,length(pairs_set)*length(lag_set)*length(ifret_set)*length(cm_set));
 temp_chiXY = zeros(92,length(pairs_set)*length(lag_set)*length(ifret_set)*length(cm_set));
 temp_chiYX = zeros(92,length(pairs_set)*length(lag_set)*length(ifret_set)*length(cm_set));
 
 params_used = zeros(4,length(pairs_set)*length(lag_set)*length(ifret_set)*length(cm_set));
     
for pair_ii = 1:length(pairs_set)
    pairs_string = num2str(pairs_set(pair_ii));
    pair_i1  = str2double(pairs_string(1));
    pair_i2  = str2double(pairs_string(2));

    name_p1  = labels_cryptodata(pair_i1,:);
    name_p2  = labels_NLP(pair_i2,:);
    for lag_ii = 1:length(lag_set)
        lag = lag_set(lag_ii);
        for ifret_ii = 1:length(ifret_set)
%             if ifret_ii-1
%                 name_data = 'ret';
%             else
%                 name_data = 'prc';
%             end
            name_data = 'HR';
            
            for cm_ii = 1:length(cm_set)
                
                counter = counter+1; 
                if cm_ii-1
                    name_cause  = 'mean';
                    name = [name_p1,'_',name_p2,'_',name_data,'_',name_cause,...
                            '_lag',num2str(lag),'_meanchi'];
                
                else
                    name_cause  = 'meancov';
                    name = [name_p1,'_',name_p2,'_',name_data,'_',name_cause, ...
                            '_lag',num2str(lag),'_SENTIMENT2_dpa'];
                
                end
                % test1_optim_predmlin_Matern_HR_Cneg_HR_meancov_lag1_SENTIMENT2_dpa_len91_from1_to92_time134932.mat
                file_name_pre = [repeating_name_part,name];
                
                matFiles = dir([file_name_pre,'*']); 
                matFile_cell = struct2cell(matFiles);
                try load(matFile_cell{1,end}, 'length_data',...
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
                params_used(:,counter) = [pairs_set(pair_ii); lag; ifret_ii-1; cm_ii-1 ];


                
            end
        end
    end
end
   
save('Lois72_GPC_hashRate_n_NLP2_dpa_plotter_from_L68')


%%%%%%%%%%%%%%%%%%%%%%
end % multiple_results
%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('Lois72_GPC_hashRate_n_NLP2_dpa_plotter_from_L68')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
load('Lois62_NLP_fin_data_sentiment2_decay_per_asset.mat')

window_length = 91; % that's roughly half a year
data_length  = length(dates_cryptodata);
dates_points = [1: 7: data_length-window_length-1];
dates_points_nr = length(dates_points);

window_dates = dates_cryptodata(dates_points+window_length);
C = window_dates([1,30,60,length(window_dates)]);


labels_cryptodata = ['HR'];

for pairs_string_which = [11 12 13 14]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pairs_string_w  = num2str(pairs_string_which);
pair_i1_which  = str2double(pairs_string_w(1));
pair_i2_which  = str2double(pairs_string_w(2));

name_p1_which  = labels_cryptodata(pair_i1_which,:);
name_p2_which  = labels_NLP(pair_i2_which,:);

indices      = find(params_used(1,:)==pairs_string_which);
sub_params   = params_used(:,indices);
sub_chiXY    = temp_chiXY(:, indices);
sub_chiYX    = temp_chiYX(:, indices);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

smooth_on = 1;
X1 = [1:92];
figure();
for ii = 1:3
    lag = lag_set(ii);
    subindices   = find(sub_params(2,:)==lag); 
    sub_sub_params = sub_params(:,subindices)
    
    if smooth_on
         
        for jj = 1:length(subindices)
            plotXY = csaps(X1,sub_chiXY(:,subindices(jj)), 0.5);
            plotYX = csaps(X1,sub_chiYX(:,subindices(jj)), 0.5);           
            
            if sub_sub_params(3,jj)
                subplot(1,2,1); hold on;
                fnplt(plotXY,'--',1); hold on        

                subplot(1,2,2); hold on;
                fnplt(plotYX,'--',1); hold on
            else
                subplot(1,2,1); hold on;
                fnplt(plotXY); hold on        

                subplot(1,2,2); hold on;
                fnplt(plotYX); hold on
            end

        end
    else
        subplot(1,2,2); hold on;
        plot(sub_chiXY(:,subindices))
        subplot(1,2,2); hold on;
        plot(sub_chiYX(:,subindices))
    end
    subplot(1,2,1); hold on;
%     legend('prc meancov', 'prc mean', 'ret meancov', 'ret mean')
    xlim([1 92])
    ylim([0 1])
    title(['1-pvalue, NLP2 ', name_p1_which, ' ', name_p2_which,' lag ', num2str(lag)])
    
    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',12)
    
    subplot(1,2,2); hold on;
%     legend('prc meancov', 'prc mean', 'ret meancov', 'ret mean')
    xlim([1 92])
    ylim([0 1])
    title(['1-pvalue, NLP2 ', name_p2_which, ' ', name_p1_which,' lag ', num2str(lag)])
    
    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',14)
    

end
subplot(1,2,1); hold on;
     legend('lag 1', 'lag 7', 'lag 30')
subplot(1,2,2); hold on;
     legend('lag 1', 'lag 7', 'lag 30')

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% for_excel_data_fin_NLP = [params_used;temp_chiYX];

for pairs_string_which = [11 12 13 14 15 16 17 18]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pairs_string_w  = num2str(pairs_string_which);
pair_i1_which  = str2double(pairs_string_w(1));
pair_i2_which  = str2double(pairs_string_w(2));

name_p1_which  = labels_cryptodata(pair_i1_which,:);
name_p2_which  = labels_NLP(pair_i2_which,:);

indices      = find(params_used(1,:)==pairs_string_which);
sub_params   = params_used(:,indices);
sub_chiXY    = temp_chiXY(:, indices);
sub_chiYX    = temp_chiYX(:, indices);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

smooth_on = 1;
X1 = [1:92];
figure();
for ii = 1:3
    lag = lag_set(ii);
    subindices   = find(sub_params(2,:)==lag); 
    sub_sub_params = sub_params(:,subindices)
    
    if smooth_on
         
        for jj = 1:length(subindices)
            plotXY = csaps(X1,sub_chiXY(:,subindices(jj)), 0.5);
            plotYX = csaps(X1,sub_chiYX(:,subindices(jj)), 0.5);           
            
            if sub_sub_params(3,jj)
                subplot(2,3,ii)
                fnplt(plotXY,'--',1); hold on        

                subplot(2,3,ii+3)
                fnplt(plotYX,'--',1); hold on
            else
                subplot(2,3,ii)
                fnplt(plotXY); hold on        

                subplot(2,3,ii+3)
                fnplt(plotYX); hold on
            end

        end
    else
        subplot(2,3,ii)
        plot(sub_chiXY(:,subindices))
        subplot(2,3,ii+3)
        plot(sub_chiYX(:,subindices))
    end
    subplot(2,3,ii)
    legend('prc meancov', 'prc mean', 'ret meancov', 'ret mean')
    xlim([1 92])
    ylim([0 1])
    title(['1-pvalue, NLP2 ', name_p1_which, ' ', name_p2_which,' lag ', num2str(lag)])
    
    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',12)
    
    subplot(2,3,ii+3)
    legend('prc meancov', 'prc mean', 'ret meancov', 'ret mean')
    xlim([1 92])
    ylim([0 1])
    title(['1-pvalue, NLP2 ', name_p2_which, ' ', name_p1_which,' lag ', num2str(lag)])
    
    set(gca, 'XTick', [1,30,60,length(window_dates)]);
    set(gca, 'XTickLabel', datestr(window_dates([1,30,60,length(window_dates)]),'mmmyy'));
    set(gca,'FontSize',14)
    

end


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
