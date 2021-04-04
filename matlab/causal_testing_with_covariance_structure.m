function [Causality_XY_vec, Causality_YX_vec, XY_chi2cdf_vec, YX_chi2cdf_vec] = ...
         causal_testing_with_covariance_structure(name, all_data, hyperparameters, dates_cryptodata, dates_sentiment, length_data, from_loop, to_loop,...
                                          meanf, likf, lag, floor_switch, covf, causal_placement_switch)

%##############################################################################
% "Sentiment-driven statistical causality in multimodal systems"
%
%  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
%
%  Anna Zaremba
%  April 2021
%
% specify the model - here Matern kernel, additive Gaussian iid noise
% use the fixed version of Matern
%##############################################################################
                                      

if nargin < 8
    floor_n = -Inf;
else
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
end


if nargin < 9
    covfunc  = {@covSum, {{@covMaternard_modified_conv_structure,3}, @covNoise}};
else
    switch covf
        case 'Matern'
            covfunc = {@covSum, {{@covMaternard_modified_conv_structure,3}, @covNoise}};
        case {'poly', 'poly-cov'}
            covfunc = {@covSum, {{@covPoly,2}, @covNoise}};
        otherwise
            covfunc = {@covSum, {{@covMaternard_modified_conv_structure, 3}, @covNoise}};
    end
end

if nargin < 10
    causal_placement_switch = 'all';
end


if nargin < 4
    meanfunc = {@meanLinear}; likfunc  = {@likGauss}; 
else
    switch meanf
        case {'lin', 'linear'}
            meanfunc = {@meanLinear};
        case {'poly', 'polynomial'}
            meanfunc = {@meanPoly,2};
        otherwise
            meanfunc = {@meanLinear};
    end
    if nargin < 5
    likfunc  = {@likGauss};
    else
        switch likf
            case {'Gauss', 'Gaus', 'Gaussian', 'likGauss'}
                likfunc  = {@likGauss}; 
                inffunc = @infExact;
                hypA.lik  = log(0); hypB.lik  = log(0);
            case {'t-student', 'student-t', 't', 'likT'}
                likfunc = {@likT};
                inffunc = @infLaplace;
                hypA.lik  = [log(0); log(1)]; hypB.lik  = [log(0); log(1)]; % for starting points...

            otherwise
                likfunc  = {@likGauss}; 
                inffunc = @infExact;
                hypA.lik  = log(0); hypB.lik  = log(0);
        end

    end
end

if strcmp(char(meanfunc{1}),'meanLinear') 
    mean_paramA_nr = 1; mean_paramB_nr = 2;
    mean_rangesA = [-1,1]; mean_rangesB = [-1,1;-1,1];
else %assume poly
    mean_paramA_nr = 2; mean_paramB_nr = 4;
    mean_rangesA = [-1,1;-1,1]; mean_rangesB = [-1,1;-1,1;-1,1;-1,1];
end

if strcmp(char(covfunc{2}{1}{1}),'covPoly') 
    cov_param_nrA = 3; cov_param_nrB = 3; % c, sf, sn
    cov_rangesA = [-10,0;-10,-2;-4,-1];
    cov_rangesB = [-10,0;-10,-2;-4,-1];
else % assume covMaternard_modified
    cov_param_nrA = 3; cov_param_nrB = 4;
    cov_rangesA = [-10,0;-10,-2;-4,-1];
    cov_rangesB = [-10,0;-10,0;-10,-2;-4,-1];
end

if strcmp(causal_placement_switch,'mean') %now cater only for causality in mean or all
    if strcmp(char(covfunc{2}{1}{1}),'covPoly') 
        cov_param_nrA = 3; cov_param_nrB = 3; % c, sf, sn
        cov_rangesA = [-10,0;-10,-2;-4,-1];
        cov_rangesB = [-10,0;-10,-2;-4,-1];
    else % assume covMaternard_modified
        cov_param_nrA = 3; cov_param_nrB = 4;
        cov_rangesA = [-10,0;-10,-2;-4,-1];
        cov_rangesB = [-Inf,-Inf;-10,0;-10,-2;-4,-1];
    end
end

%% load required data and re-compute test statistic


hyps_A_XY_vec = hyperparameters{1};
hyps_B_XY_vec = hyperparameters{2};
hyps_A_YX_vec = hyperparameters{3};
hyps_B_YX_vec = hyperparameters{4};


multirun_number = 100
if from_loop == 1
    multirun_number = to_loop
end
% preallocate memory for the predicted (fitted) values
mean_predA_XY_vec = zeros(multirun_number,1);
mean_predB_XY_vec = zeros(multirun_number,1);
mean_predA_YX_vec = zeros(multirun_number,1);
mean_predB_YX_vec = zeros(multirun_number,1);

nlmlA_XY_multiruns = zeros(multirun_number,1);
nlmlB_XY_multiruns = zeros(multirun_number,1);
nlmlA_YX_multiruns = zeros(multirun_number,1);
nlmlB_YX_multiruns = zeros(multirun_number,1);

Causality_XY_vec = zeros(multirun_number,1);
Causality_YX_vec = zeros(multirun_number,1);

XY_chi2cdf_vec = zeros(multirun_number,1);
YX_chi2cdf_vec = zeros(multirun_number,1);

% each run_ii corresponds to one window

for run_ii = from_loop:to_loop
    run_ii
    
    Data = all_data(:,:,run_ii);
    % will want to test X --> Y | Z and Y --> X | Z
    %%%%%%%%% for X --> Y | Z
    fromXY = 1; toXY = 2; sideXY = [];
    inputA_XY = Data(1:end-lag,[toXY, sideXY]); %remember: nonnested here
    inputB_XY = Data(1:end-lag,[fromXY, toXY, sideXY]);
    target_XY = Data(lag+1:end,toXY);

    % load hyperparameters - only valid for non-FnG data
    % Model A
    params_window_tmp = hyps_A_XY_vec(run_ii,:)
    hypA_XY_best.mean = params_window_tmp(1:mean_paramA_nr)';
    hypA_XY_best.cov  = params_window_tmp(mean_paramA_nr + 1:mean_paramA_nr + cov_param_nrA)';
    
    % Model B
    params_window_tmp = hyps_B_XY_vec(run_ii,:)
    hypB_XY_best.mean = params_window_tmp(1:mean_paramB_nr)';
    hypB_XY_best.cov  = params_window_tmp(mean_paramB_nr + 1:mean_paramB_nr + cov_param_nrB)';
  
    %%%%%%%%% for Y --> X | Z
    fromYX = 2; toYX = 1; sideYX = [];
    inputA_YX = Data(1:end-lag,[toYX, sideYX]); %remember: nonnested here
    inputB_YX = Data(1:end-lag,[fromYX, toYX, sideYX]);
    target_YX = Data(lag+1:end,toYX);
    
    inputA_XY_test = Data(end-lag+1,[toXY, sideXY]);
    inputB_XY_test = Data(end-lag+1,[fromXY, toXY, sideXY]);
    inputA_YX_test = Data(end-lag+1,[toYX, sideYX]); 
    inputB_YX_test = Data(end-lag+1,[fromYX, toYX, sideYX]);

    % load hyperparameters - only valid for non-FnG data
    % Model A
    params_window_tmp = hyps_A_YX_vec(run_ii,:)
    hypA_YX_best.mean = params_window_tmp(1:mean_paramA_nr)';
    hypA_YX_best.cov  = params_window_tmp(mean_paramA_nr + 1:mean_paramA_nr + cov_param_nrA)';
    
    % Model B
    params_window_tmp = hyps_B_YX_vec(run_ii,:)
    hypB_YX_best.mean = params_window_tmp(1:mean_paramB_nr)';
    hypB_YX_best.cov  = params_window_tmp(mean_paramB_nr + 1:mean_paramB_nr + cov_param_nrB)';

    % add the lik parameter:
    hypA_XY_best.lik =  hypA.lik;
    hypB_XY_best.lik =  hypA.lik;
    hypA_YX_best.lik =  hypA.lik;
    hypB_YX_best.lik =  hypA.lik;
    
    % compute test statistic

    %%%%%%%%% for X --> Y | Z    
    % first get the additional predictive means:
    mean_predA_XY_vec(run_ii) = gp(hypA_XY_best, inffunc, meanfunc, covfunc, likfunc, inputA_XY, target_XY, ... 
                                                                            inputA_XY_test);
    mean_predB_XY_vec(run_ii) = gp(hypB_XY_best, inffunc, meanfunc, covfunc, likfunc, inputB_XY, target_XY, ...
                                                                            inputB_XY_test);
    % then the usual negative log marginal likelihoods:
      
    nlmlA_XY_multiruns(run_ii) = gp(hypA_XY_best, inffunc, meanfunc, covfunc, likfunc, inputA_XY, ...
                                                target_XY);
    nlmlB_XY_multiruns(run_ii) = gp(hypB_XY_best, inffunc, meanfunc, covfunc, likfunc, inputB_XY, ...
                                                target_XY);
    
    % nlml is negative log likelihood, so need to put "-" in front
    Causality_XY_vec(run_ii) = - (nlmlB_XY_multiruns(run_ii) - nlmlA_XY_multiruns(run_ii));
    
    %%%%%%%%% for Y --> X | Z
    % first get the additional predictive means:
    mean_predA_YX_vec(run_ii) = gp(hypA_YX_best, inffunc, meanfunc, covfunc, likfunc, inputA_YX, target_YX, ...
                                                            inputA_YX_test);
    mean_predB_YX_vec(run_ii) = gp(hypB_YX_best, inffunc, meanfunc, covfunc, likfunc, inputB_YX, target_YX, ...
                                                           inputB_YX_test);
    % then the usual negative log marginal likelihoods:
    nlmlA_YX_multiruns(run_ii) = gp(hypA_YX_best, inffunc, meanfunc, covfunc, likfunc, inputA_YX, target_YX);
    nlmlB_YX_multiruns(run_ii) = gp(hypB_YX_best, inffunc, meanfunc, covfunc, likfunc, inputB_YX, target_YX);
    
    % nlml is negative log likelihood, so need to put "-" in front
    Causality_YX_vec(run_ii) = - (nlmlB_YX_multiruns(run_ii) - nlmlA_YX_multiruns(run_ii));

    
    if strcmp(causal_placement_switch,'mean') 
        XY_chi2cdf_vec(run_ii) = chi2cdf(2*Causality_XY_vec(run_ii),1);
        YX_chi2cdf_vec(run_ii) = chi2cdf(2*Causality_YX_vec(run_ii),1);
    else
        XY_chi2cdf_vec(run_ii) = chi2cdf(2*Causality_XY_vec(run_ii),2);
        YX_chi2cdf_vec(run_ii) = chi2cdf(2*Causality_YX_vec(run_ii),2);
    end

    XY_chi2cdf_vec(run_ii) = max(floor_n, XY_chi2cdf_vec(run_ii));
    YX_chi2cdf_vec(run_ii) = max(floor_n, YX_chi2cdf_vec(run_ii));
    
end       


eval(['save(', char(39), '/tmp/convolutional_structure/conv_struct_',meanf,'_',covf,'_',name,...
     '_len',num2str(length_data),...
     '_from',num2str(from_loop),'_to' ,num2str(to_loop),'.mat',char(39), ')']);


end % function end
