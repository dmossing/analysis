function [Y,new_beginds] = filter_by_beginds(X,beginds,gd)
[~,bins] = histc(1:size(X,1),beginds);
Y = X(gd(bins),:);
d = diff(beginds);
new_beginds = cumsum([1; d(gd)]);
end