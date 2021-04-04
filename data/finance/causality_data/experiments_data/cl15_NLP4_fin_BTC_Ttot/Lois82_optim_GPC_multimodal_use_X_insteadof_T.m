function [Causality_Y_vec, Causality_X_vec, chi2cdf_X_vec, chi2cdf_Y_vec,...
                            hyperparameters] = ...
         Lois82_optim_GPC_multimodal_use_X_insteadof_T(all_data,from_loop, to_loop, ...
                            name, lag, causal_placement_switch)


%##############################################################################
% "Sentiment-driven statistical causality in multimodal systems"
%
%  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
%
%  Anna Zaremba
%  April 2021
%
%##############################################################################


time_now = now;
starting_time = floor(1.e+06 * rem(time_now,1));
starting_time_name = num2str(starting_time);
% important to distinguish the jobs!

floor_switch= 'true';
    switch floor_switch
        case 'true'
            floor_n = 0;
        case 1
            floor_n = 0;
        case 0
            floor_n = -Inf;
        case 'false'
            floor_n = -Inf;
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if nargin<3
    to_loop = 100;
    if nargin<2 
        from_loop = 1;
    end
end

length_data = max(size(all_data(:,:,1)));
%     lag = 1;


%% 2. specify the model - here Matern kernel, additive Gaussian iid noise
% use the fixed version of Matern


if nargin < 6
    causal_placement_switch = 'all';
end


meanfunc = {@meanLinear}; 
covfunc  = {@covSum, {{@covMaternard_modified,3}, @covNoise}};
likfunc  = {@likGauss}; 
inffunc = @infExact;
hypA.lik  = log(0); hypB.lik  = log(0);

meanf = 'lin';
covf  = 'Matern'
 


%% run many tests to compare!

multirun_number = 100;
if from_loop == 1
    multirun_number = to_loop
end
step_nr = -30; % in optimisation
requested_nr = 5; % how many best starting points
grid_nr = 5; % grid for the starting points


nlmlA_X_multiruns = zeros(multirun_number,1);
nlmlB_X_multiruns = zeros(multirun_number,1);
nlmlA_Y_multiruns = zeros(multirun_number,1);
nlmlB_Y_multiruns = zeros(multirun_number,1);


Causality_X_vec = zeros(multirun_number,1);
Causality_Y_vec = zeros(multirun_number,1);

chi2cdf_X_vec = zeros(multirun_number,1);
chi2cdf_Y_vec = zeros(multirun_number,1);

%%%%%%% many starting points:
mean_paramA_nr = 1; mean_paramB_nr = 2;
mean_rangesA = [-1,1]; mean_rangesB = [-1,1;-1,1];
    
% assume covMaternard_modified
cov_param_nrA = 3; cov_param_nrB = 4;
cov_rangesA = [-10,0;-10,-2;-4,-1];
cov_rangesB = [-10,0;-10,0;-10,-2;-4,-1];


if strcmp(causal_placement_switch,'mean') %now cater only for causality in mean or all
    % assume covMaternard_modified
    cov_param_nrA = 3; cov_param_nrB = 4;
    cov_rangesA = [-10,0;-10,-2;-4,-1];
    cov_rangesB = [-Inf,-Inf;-10,0;-10,-2;-4,-1];
end
    
    
% preallocate memory for the predicted (fitted) values
mean_predA_X1_vec = zeros(multirun_number,1);
mean_predA_X2_vec = zeros(multirun_number,1);
mean_predA_X3_vec = zeros(multirun_number,1);
mean_predA_Y_vec  = zeros(multirun_number,1);
mean_predB_X1_vec = zeros(multirun_number,1);
mean_predB_X2_vec = zeros(multirun_number,1);
mean_predB_X3_vec = zeros(multirun_number,1);    
mean_predB_Y_vec  = zeros(multirun_number,1);


    
% preallocate memory for saving the parameters
hyps_A_X1_vec = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_X1_vec = nan(multirun_number,mean_paramB_nr+cov_param_nrB);
hyps_A_X2_vec = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_X2_vec = nan(multirun_number,mean_paramB_nr+cov_param_nrB);
hyps_A_X3_vec = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_X3_vec = nan(multirun_number,mean_paramB_nr+cov_param_nrB);
hyps_A_Y_vec  = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_Y_vec  = nan(multirun_number,mean_paramB_nr+cov_param_nrB);

w1 = zeros(size(all_data(:,1,1)));
w2 = zeros(size(all_data(:,1,1)));
w3 = zeros(size(all_data(:,1,1)));

% get the time data, which will be on the same scale as everything else,
% i.e. also studentised!
t_data = 1:max(size(all_data(:,:,1)));
T_ts   = zscore(t_data)';


tic    
for run_ii = from_loop:to_loop
run_ii
    
    % will want to test [X1,X2,X3] --> Y, Y --> [X1,X2,X3] 
    % BUT we build the models for Xi separately, then we combine them 
    Data  = all_data(:,:,run_ii);
    X1_ts = Data(:,1); % remember, for input use 1:end-lag
    X2_ts = Data(:,2);
    X3_ts = Data(:,3);
    Y_ts  = Data(:,4);
    % T_ts
    % remember, for input use 1:end-lag, for target: lag+1:end
    
    
    % FIRST! we need to get w1, w2, w3
    % so, we need A models for Xi
    [hypA_X1_param_out, nA_X1_out] = get_starting_point_less_points_seed_version4NLP(mean_paramA_nr, cov_param_nrA,...
    [mean_rangesA;     cov_rangesA],... % mean and cov
     grid_nr,requested_nr, X1_ts(1:end-lag), X1_ts(lag+1:end),...
     meanfunc, covfunc, likfunc, hypA.lik, inffunc, -10*run_ii);
 
    [hypA_X2_param_out, nA_X2_out] = get_starting_point_less_points_seed_version4NLP(mean_paramA_nr, cov_param_nrA,...
    [mean_rangesA;     cov_rangesA],... % mean and cov
     grid_nr,requested_nr, X2_ts(1:end-lag), X2_ts(lag+1:end),...
     meanfunc, covfunc, likfunc, hypA.lik, inffunc, -10*run_ii);
 
    [hypA_X3_param_out, nA_X3_out] = get_starting_point_less_points_seed_version4NLP(mean_paramA_nr, cov_param_nrA,...
    [mean_rangesA;     cov_rangesA],... % mean and cov
     grid_nr,requested_nr, X3_ts(1:end-lag), X3_ts(lag+1:end),...
     meanfunc, covfunc, likfunc, hypA.lik, inffunc, -10*run_ii);
    
 
    [hyp_best_X1_A] = optimise_parameters(X1_ts(1:end-lag), X1_ts(lag+1:end), ...
                                          hypA_X1_param_out, nA_X1_out, ...
                                          meanfunc, covfunc, likfunc, inffunc, step_nr, ...
                                          mean_paramA_nr, cov_param_nrA, requested_nr);
    [hyp_best_X2_A] = optimise_parameters(X2_ts(1:end-lag), X2_ts(lag+1:end), ...
                                          hypA_X2_param_out, nA_X2_out, ...
                                          meanfunc, covfunc, likfunc, inffunc, step_nr, ...
                                          mean_paramA_nr, cov_param_nrA, requested_nr);
    [hyp_best_X3_A] = optimise_parameters(X3_ts(1:end-lag), X3_ts(lag+1:end), ...
                                          hypA_X3_param_out, nA_X3_out, ...
                                          meanfunc, covfunc, likfunc, inffunc, step_nr, ...
                                          mean_paramA_nr, cov_param_nrA, requested_nr);
    
    k1 = feval(covfunc{:}, hyp_best_X1_A.cov, 0);
    k2 = feval(covfunc{:}, hyp_best_X2_A.cov, 0);
    k3 = feval(covfunc{:}, hyp_best_X3_A.cov, 0);
    
    w1(run_ii) = k1 / (k1+k2+k3);
    w2(run_ii) = k2 / (k1+k2+k3);
    w3(run_ii) = k3 / (k1+k2+k3);


    % save hyperparameters
    hyps_A_X1_vec(run_ii,:) = [hyp_best_X1_A.mean', hyp_best_X1_A.cov'];
    hyps_A_X2_vec(run_ii,:) = [hyp_best_X2_A.mean', hyp_best_X2_A.cov'];
    hyps_A_X3_vec(run_ii,:) = [hyp_best_X3_A.mean', hyp_best_X3_A.cov'];
                                      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    % SECOND! the B models for Xi
    [hypB_X1_param_out, nB_X1_out] = get_starting_point_less_points_seed_version4NLP(mean_paramB_nr, cov_param_nrB,...
    [mean_rangesB;     cov_rangesB],... % mean and cov
     grid_nr,requested_nr, [Y_ts(1:end-lag),X1_ts(1:end-lag)], X1_ts(lag+1:end),...
     meanfunc, covfunc, likfunc, hypB.lik, inffunc, -10*run_ii); 
    
    [hypB_X2_param_out, nB_X2_out] = get_starting_point_less_points_seed_version4NLP(mean_paramB_nr, cov_param_nrB,...
    [mean_rangesB;     cov_rangesB],... % mean and cov
     grid_nr,requested_nr, [Y_ts(1:end-lag),X2_ts(1:end-lag)], X2_ts(lag+1:end),...
     meanfunc, covfunc, likfunc, hypB.lik, inffunc, -10*run_ii); 
    
    [hypB_X3_param_out, nB_X3_out] = get_starting_point_less_points_seed_version4NLP(mean_paramB_nr, cov_param_nrB,...
    [mean_rangesB;     cov_rangesB],... % mean and cov
     grid_nr,requested_nr, [Y_ts(1:end-lag),X3_ts(1:end-lag)], X3_ts(lag+1:end),...
     meanfunc, covfunc, likfunc, hypB.lik, inffunc, -10*run_ii); 

 
    [hyp_best_X1_B] = optimise_parameters([Y_ts(1:end-lag),X1_ts(1:end-lag)], X1_ts(lag+1:end), ...
                                          hypB_X1_param_out, nB_X1_out, ...
                                          meanfunc, covfunc, likfunc, inffunc, step_nr, ...
                                          mean_paramB_nr, cov_param_nrB, requested_nr);
    [hyp_best_X2_B] = optimise_parameters([Y_ts(1:end-lag),X2_ts(1:end-lag)], X2_ts(lag+1:end), ...
                                          hypB_X2_param_out, nB_X2_out, ...
                                          meanfunc, covfunc, likfunc, inffunc, step_nr, ...
                                          mean_paramB_nr, cov_param_nrB, requested_nr);
    [hyp_best_X3_B] = optimise_parameters([Y_ts(1:end-lag),X3_ts(1:end-lag)], X3_ts(lag+1:end), ...
                                          hypB_X3_param_out, nB_X3_out, ...
                                          meanfunc, covfunc, likfunc, inffunc, step_nr, ...
                                          mean_paramB_nr, cov_param_nrB, requested_nr);
    
    % save hyperparameters
    hyps_B_X1_vec(run_ii,:) = [hyp_best_X1_B.mean', hyp_best_X1_B.cov'];
    hyps_B_X2_vec(run_ii,:) = [hyp_best_X2_B.mean', hyp_best_X2_B.cov'];
    hyps_B_X3_vec(run_ii,:) = [hyp_best_X3_B.mean', hyp_best_X3_B.cov'];
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % now! I can put the time series X together!
    % X = w1 * X1 + w2 * X2 + w3 * X3
    X_ts = w1(run_ii) * X1_ts + w2(run_ii) * X2_ts + w3(run_ii) * X3_ts;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % now we can model Y, because we got X!
    [hypA_Y_param_out, nA_Y_out] = get_starting_point_less_points_seed_version4NLP(mean_paramA_nr, cov_param_nrA,...
        [mean_rangesA;     cov_rangesA],... % mean and cov
        grid_nr,requested_nr, Y_ts(1:end-lag), Y_ts(lag+1:end),...
        meanfunc, covfunc, likfunc, hypA.lik, inffunc, -10*run_ii);
    [hypB_Y_param_out, nB_Y_out] = get_starting_point_less_points_seed_version4NLP(mean_paramB_nr, cov_param_nrB,...
        [mean_rangesB;     cov_rangesB],... % mean and cov
        grid_nr,requested_nr, [X_ts(1:end-lag),Y_ts(1:end-lag)], Y_ts(lag+1:end),...
        meanfunc, covfunc, likfunc, hypB.lik, inffunc, -10*run_ii);  
 
    [hyp_best_Y_A] = optimise_parameters(Y_ts(1:end-lag), Y_ts(lag+1:end),...
                                         hypA_Y_param_out, nA_Y_out, ...
                                         meanfunc, covfunc, likfunc, inffunc, step_nr, ...
                                         mean_paramA_nr, cov_param_nrA, requested_nr);
 
    [hyp_best_Y_B] = optimise_parameters([X_ts(1:end-lag),Y_ts(1:end-lag)], Y_ts(lag+1:end),...
                                         hypB_Y_param_out, nB_Y_out, ...
                                         meanfunc, covfunc, likfunc, inffunc, step_nr, ...
                                         mean_paramB_nr, cov_param_nrB, requested_nr);
    
    % save hyperparameters
    hyps_A_Y_vec(run_ii,:) = [hyp_best_Y_A.mean', hyp_best_Y_A.cov'];
    hyps_B_Y_vec(run_ii,:) = [hyp_best_Y_B.mean', hyp_best_Y_B.cov'];
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % NOW, the causality!
    
    %%%%%%%%% for X --> Y   
    % first get the additional predictive means:
    mean_predA_Y_vec(run_ii) = gp(hyp_best_Y_A, inffunc, meanfunc, covfunc, likfunc, Y_ts(1:end-lag), Y_ts(lag+1:end), Y_ts(end-lag+1));
    mean_predB_Y_vec(run_ii) = gp(hyp_best_Y_B, inffunc, meanfunc, covfunc, likfunc, [X_ts(1:end-lag),Y_ts(1:end-lag)], Y_ts(lag+1:end), [X_ts(end-lag+1),Y_ts(end-lag+1)]);
    % then the usual negative log marginal likelihoods:
    nlmlA_Y_multiruns(run_ii) = gp(hyp_best_Y_A, inffunc, meanfunc, covfunc, likfunc, Y_ts(1:end-lag), Y_ts(lag+1:end));
    nlmlB_Y_multiruns(run_ii) = gp(hyp_best_Y_B, inffunc, meanfunc, covfunc, likfunc, [X_ts(1:end-lag),Y_ts(1:end-lag)], Y_ts(lag+1:end));
    % nlml is negative log likelihood, so need to put "-" in front
    Causality_Y_vec(run_ii) = - (nlmlB_Y_multiruns(run_ii) - nlmlA_Y_multiruns(run_ii));
    
    %%%%%%%%% for Y --> X   
    % first get the additional predictive means:
    mean_predA_X1_vec(run_ii) = gp(hyp_best_X1_A, inffunc, meanfunc, covfunc, likfunc, X1_ts(1:end-lag), X1_ts(lag+1:end), X1_ts(end-lag+1));
    mean_predB_X1_vec(run_ii) = gp(hyp_best_X1_B, inffunc, meanfunc, covfunc, likfunc, [Y_ts(1:end-lag),X1_ts(1:end-lag)], X1_ts(lag+1:end), [Y_ts(end-lag+1),T_ts(end-lag+1)]);
    mean_predA_X2_vec(run_ii) = gp(hyp_best_X2_A, inffunc, meanfunc, covfunc, likfunc, X2_ts(1:end-lag), X2_ts(lag+1:end), X2_ts(end-lag+1));
    mean_predB_X2_vec(run_ii) = gp(hyp_best_X2_B, inffunc, meanfunc, covfunc, likfunc, [Y_ts(1:end-lag),X2_ts(1:end-lag)], X2_ts(lag+1:end), [Y_ts(end-lag+1),T_ts(end-lag+1)]);
    mean_predA_X3_vec(run_ii) = gp(hyp_best_X3_A, inffunc, meanfunc, covfunc, likfunc, X3_ts(1:end-lag), X3_ts(lag+1:end), X3_ts(end-lag+1));
    mean_predB_X3_vec(run_ii) = gp(hyp_best_X3_B, inffunc, meanfunc, covfunc, likfunc, [Y_ts(1:end-lag),X3_ts(1:end-lag)], X3_ts(lag+1:end), [Y_ts(end-lag+1),T_ts(end-lag+1)]);
    % then the usual negative log marginal likelihoods:
    nlmlA_X1(run_ii) = gp(hyp_best_X1_A, inffunc, meanfunc, covfunc, likfunc, X1_ts(1:end-lag), X1_ts(lag+1:end));
    nlmlB_X1(run_ii) = gp(hyp_best_X1_B, inffunc, meanfunc, covfunc, likfunc, [Y_ts(1:end-lag),X1_ts(1:end-lag)], X1_ts(lag+1:end));
    nlmlA_X2(run_ii) = gp(hyp_best_X2_A, inffunc, meanfunc, covfunc, likfunc, X2_ts(1:end-lag), X2_ts(lag+1:end));
    nlmlB_X2(run_ii) = gp(hyp_best_X2_B, inffunc, meanfunc, covfunc, likfunc, [Y_ts(1:end-lag),X2_ts(1:end-lag)], X2_ts(lag+1:end));
    nlmlA_X3(run_ii) = gp(hyp_best_X3_A, inffunc, meanfunc, covfunc, likfunc, X3_ts(1:end-lag), X3_ts(lag+1:end));
    nlmlB_X3(run_ii) = gp(hyp_best_X3_B, inffunc, meanfunc, covfunc, likfunc, [Y_ts(1:end-lag),X3_ts(1:end-lag)], X3_ts(lag+1:end));
    % nlml is negative log likelihood, so need to put "-" in front
    
    nlml_X1_multiruns(run_ii) = - (nlmlB_X1(run_ii) - nlmlA_X1(run_ii));
    nlml_X2_multiruns(run_ii) = - (nlmlB_X2(run_ii) - nlmlA_X2(run_ii));
    nlml_X3_multiruns(run_ii) = - (nlmlB_X3(run_ii) - nlmlA_X3(run_ii));
    
    Causality_X_vec(run_ii) = w1(run_ii)*nlml_X1_multiruns(run_ii) + w2(run_ii)*nlml_X2_multiruns(run_ii) + w3(run_ii)*nlml_X3_multiruns(run_ii);
    
    
    if strcmp(causal_placement_switch,'mean') 
        chi2cdf_X_vec(run_ii) = chi2cdf(2*Causality_X_vec(run_ii),1);
        chi2cdf_Y_vec(run_ii) = chi2cdf(2*Causality_Y_vec(run_ii),1);
    else
        chi2cdf_X_vec(run_ii) = chi2cdf(2*Causality_X_vec(run_ii),2);
        chi2cdf_Y_vec(run_ii) = chi2cdf(2*Causality_Y_vec(run_ii),2);
    end
    
    chi2cdf_X_vec(run_ii) = max(floor_n, chi2cdf_X_vec(run_ii));
    chi2cdf_Y_vec(run_ii) = max(floor_n, chi2cdf_Y_vec(run_ii));
        
end


%%

hyperparameters{1} = hyps_A_X1_vec;
hyperparameters{2} = hyps_B_X1_vec;
hyperparameters{3} = hyps_A_X2_vec;
hyperparameters{4} = hyps_B_X2_vec;
hyperparameters{5} = hyps_A_X3_vec;
hyperparameters{6} = hyps_B_X3_vec;
hyperparameters{7} = hyps_A_Y_vec;
hyperparameters{8} = hyps_B_Y_vec;

toc
    eval(['save(',char(39), 'test_optim_multimod_',meanf,'_',covf,'_',name,...
     '_len',num2str(length_data),...
     '_from',num2str(from_loop),'_to' ,num2str(to_loop),...
     '_time',num2str(starting_time_name),'.mat',char(39), ')']);

 
disp('Hej!')

%%
% 
%     F = figure('visible','off')
figure()
subplot(1,2,1); 
plot(Causality_Y_vec(from_loop:to_loop)); hold on
plot(Causality_X_vec(from_loop:to_loop));
title('causality from evidence')
legend('X-->Y','Y-->X')


subplot(1,2,2);
plot(chi2cdf_Y_vec(from_loop:to_loop),'-o'); 
hold on
plot(chi2cdf_X_vec(from_loop:to_loop),'-*'); 
title(['chi2cdf', name])
legend('X-->Y','Y-->X')

%     savefig(F,['GPC',name,'.fig'])



function [hyp_best] = optimise_parameters(input, target, ...
                                          hyp_starting_out, n_starting_out, ...
                                          meanfunc, covfunc, likfunc, inffunc, step_nr, ...
                                          mean_param_nr, cov_param_nr, requested_nr)

n_best = n_starting_out(1);
hyp_best.lik =  log(0); % well, that's actually always the starting point



mean_nums = [1:mean_param_nr];
cov_nums  = [mean_param_nr+1:mean_param_nr+cov_param_nr]; 


hyp_best.mean=hyp_starting_out(mean_nums,1);
hyp_best.cov =hyp_starting_out(cov_nums ,1);

hyp_opt.lik =  log(0);
  

for h_ii = 1:requested_nr

    hyp_opt.mean=hyp_starting_out(mean_nums,h_ii);
    hyp_opt.cov =hyp_starting_out(cov_nums ,h_ii);

    [hyp_opt] = minimize_modified(hyp_opt, @gp, step_nr, ...
       inffunc, meanfunc, covfunc, likfunc, input, target);

    nlml_opt = gp(hyp_opt, inffunc, meanfunc, ...
    covfunc, likfunc, input, target);
      % now replace the best ones:
   if nlml_opt<n_best
       hyp_best = hyp_opt;
       n_best = nlml_opt;
   end
end
