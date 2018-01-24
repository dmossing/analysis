function [corrected,baseline,neuropilMultiplier] = neuropilRR(Data,Neuropil)
% weighting fn. penalizes large negative residuals more than large positive
% ones
wfun = @(r)((r<0)+(r>=0).*(r<1).*(1 - r.^2).^2);
N = size(Data,1);
corrected = zeros(size(Data));
neuropilMultiplier = zeros(N,1);
baseline = zeros(N,1);
for i=1:N
    b = robustfit(Neuropil(i,:),Data(i,:),wfun);
    neuropilMultiplier(i) = b(2);
    baseline(i) = b(1);
    corrected(i,:) = Data(i,:)-neuropilMultiplier(i)*Neuropil(i,:);
end