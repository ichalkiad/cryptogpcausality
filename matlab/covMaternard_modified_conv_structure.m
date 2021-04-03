function K = covMaternard_modified_conv_structure(d, hyp, x, z, i)

%##############################################################################
% "Sentiment-driven statistical causality in multimodal systems"
%
%  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
%
%  Anna Zaremba
%  April 2021
%
% fixed for errors in derivatives, modified to change division by lengthscale 
% to multiplication by it, so that lengthscale can be equal zero
% hard-coded convolutional component - parameters to be set in lines 41-44
%##############################################################################


% Matern covariance function with nu = d/2 and with Automatic Relevance
% Determination (ARD) distance measure. For d=1 the function is also known as
% the exponential covariance function or the Ornstein-Uhlenbeck covariance 
% in 1d. The covariance function is:
%
%   k(x^p,x^q) = sf^2 * f( sqrt(d)*r ) * exp(-sqrt(d)*r)
%
% with f(t)=1 for d=1, f(t)=1+t for d=3 and f(t)=1+t+t²/3 for d=5.
% Here r is the distance sqrt((x^p-x^q)'*inv(P)*(x^p-x^q)), where the P matrix
% is diagonal with ARD parameters ell_1^2,...,ell_D^2, where D is the dimension
% of the input space and sf2 is the signal variance. The hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          ..
%         log(ell_D)
%         log(sf) ]
%
% Copyright (c) by Hannes Nickisch, 2013-10-13.
%
% See also COVFUNCTIONS.M.



long_memory_d = 0.15
cyclic_freq_u = 1.0
sigma = 1.0
Nmax = 5000

if nargin<3, K = '(D+1)'; return; end              % report number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

[n,D] = size(x);
ell = exp(hyp(1:D));
sf2 = exp(2*hyp(D+1));
if all(d~=[1,3,5]), error('only 1, 3 and 5 allowed for d'), end         % degree

switch d
  case 1, f = @(t) 1;               df = @(t) 1./t;     % df(t) = (f(t)-f'(t))/t
  case 3, f = @(t) 1 + t;           df = @(t) 1;
  case 5, f = @(t) 1 + t.*(1+t/3);  df = @(t) (1+t)/3;
end
          m = @(t,f) f(t).*exp(-t); dm = @(t,f) df(t).*exp(-t);

% precompute distances
if dg                                                               % vector kxx
  K = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = sq_dist(diag(sqrt(d).*ell)*x');
  else                                                   % cross covariances Kxz
    K = sq_dist(diag(sqrt(d).*ell)*x',diag(sqrt(d).*ell)*z');
  end
end

if nargin<5                                                        % covariances
  K = sf2*m(sqrt(K),f);

  % add convolutional component
  [m, n] = size(K);
  K_conv = discrete_long_memory_filter(m, n, long_memory_d, cyclic_freq_u, sigma, Nmax);
  K = K + K_conv;
 
%    dbstack;
else                                                               % derivatives
  if i<=D                                               % length scale parameter
%        dbstack;
    if dg
      Ki = zeros(size(x,1),1);
    else
      if xeqz
        Ki = sq_dist(sqrt(d)*ell(i)*x(:,i)');
      else
        Ki = sq_dist(sqrt(d)*ell(i)*x(:,i)',sqrt(d)*ell(i)*z(:,i)');
      end
    end
    K = (1*ell(i))*sf2*dm(sqrt(K),f).*Ki;
    K(Ki<1e-12) = 0;                                    % fix limit case for d=1
  elseif i==D+1                                            % magnitude parameter
    K = 2*sqrt(sf2)*m(sqrt(K),f);     %CORRECTED THE DERIVATIVE!
  else
    error('Unknown hyperparameter')
  end
end
