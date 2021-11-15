function [mnNorm] = norm01(mn,alldims)
if nargin<2
    alldims = 0;
end
if ~alldims
    if min(size(mn))==1
        mn = mn(:)';
    end
    mnNorm = zeros(size(mn));
    for i=1:size(mn,1)
        normfact = (max(mn(i,:))-min(mn(i,:)));
        mnNorm(i,:) = (mn(i,:)-min(mn(i,:)))/normfact;
    end
else
    mxm = max(mn(:));
    mnm = min(mn(:));
    mnNorm = (mn-mnm)/(mxm-mnm);
end