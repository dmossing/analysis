function [dp,p] = dprimechance(n,k1,k2,pcutoff)
% k1 hits out of n GO trials; k2 false alarms out of n NO GO trials\
% dp: dprime at chance level
% p: probability that you'd get a difference at least as large as in your
% data, if the mouse were licking at equal probability for both trial types
if ~exist('pcutoff','var') || isempty(pcutoff)
    pcutoff = 0.05; % you can change this
end
dprime_meas = norminv(k1/n)-norminv(k2/n); % dprime 
q = 0.5*(k1+k2)/n; % maximum likelihood model with uniform lick rate for both
prob = binopdf(0:n,n,q);
probsq = prob'*prob; % probsq(i,j) = prob. of i-1 hits and j-1 false alarms
z = norminv((0:n)/n);
dprimes = repmat(z',1,n+1)-repmat(z,n+1,1); 
% dprimes(i,j) = dprime value for i-1 hits, j-1 false alarms
prob_sort_by_dprime = probsq(:);
prob_sort_by_dprime = prob_sort_by_dprime(sort_by(dprimes(:))); % sort by ascending d'
dprime_sort = sort(dprimes(:));
cdf = cumsum(prob_sort_by_dprime);
% cdf(dprime_sort==whatever) = prob. of getting a worse d' than whatever, by chance
dp = dprime_sort(find(cdf>1-pcutoff,1)); % d' corresponding to just above your p-cutoff
p = 1-cdf(find(dprime_sort>dprime_meas,1));

function aux = sort_by(idx)
if size(idx,1)==1 || size(idx,2)==1
    idx = idx(:);
end
aux = sortrows([idx [1:size(idx,1)]']);
aux = aux(:,end);