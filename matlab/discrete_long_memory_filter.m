function [K] = discrete_long_memory_filter(m, n, d, u, sigma, Nmax)

%##############################################################################
% "Sentiment-driven statistical causality in multimodal systems"
%
%  Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler
%
%  Ioannis Chalkiadakis  ic14@hw.ac.uk
%  April 2021
%##############################################################################

    if ~exist('Nmax','var')
        Nmax = 5000;
    end
    if d < 0 || d > 0.5
        error('Long memory exponent is between 0 and 0.5')
    end
        
    % following computation can be moved outside the function,
    % computed only once, and re-use vector psi
    % compute Gegenbauer coefficients
    psi = zeros(Nmax-1, 1);
    for j = 1:Nmax-1
       psi(j) = gegenbauer_coeff(j, d, u, psi); 
    end
    psi = [1; psi];
    
    % fill in matrix of Gegenbauer products
    psiprod = zeros(Nmax, Nmax);
    for i = 1:Nmax
       for j = 1:i           
           psiprod(i, j) = psi(i)*psi(j);
           psiprod(j, i) = psiprod(i, j);
       end        
    end 
 
    if (n==1 && m > 1)
        % for vector k(x_*, x) - WHAT DO WE USE FOR TIMES t, s?
        K = zeros(m, 1);
        t = m + 1 % t = m + 1 since we only evaluate on 1 test case per window?
        for s = 1:m
                K(s) = (sigma^2)*sum(diag(psiprod, s-t));
        end
    else
        K = zeros(m, n);
        for s = 1:m
            for t = 1:s
                K(s, t) = (sigma^2)*sum(diag(psiprod, s-t));
                K(t, s) = K(s, t);
            end
        end    
    end
end

function [psij] = gegenbauer_coeff(j, d, u, psis)
    % compute Gegenbauer coefficient at time j with long memory d
    if j == 0
        psij = 1;
    elseif j == 1
        psij = 2*d*u;
    elseif j == 2
        psij = -d + 2*d*(1 + d)*u^2;
    else
        psij = 2*u*((d-1)/j + 1)*psis(j-1) - (2*(d-1)/j + 1)*psis(j-2);
    end    
end
