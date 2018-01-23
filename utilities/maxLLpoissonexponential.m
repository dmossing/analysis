function tau = maxLLpoissonexponential(k,lambda0)
% k is an N trial by T time point array of spike counts
% lambda0 the mean spike count in the "baseline" time bins
% returns tau in units of # of time bins
t = repmat(1:size(k,2),size(k,1),1);
    function L = computeLL(tau)
        L = sum(sum(k.*t/tau + lambda0*exp(-t/tau)));
    end
tau = fminunc(@computeLL,1);
end