function [bm,bs,bsem] = binxy(x,y,binno)
[~,bind] = histc(x,linspace(min(x),max(x),binno));
[bm,bs,bsem] = deal(zeros(size(unique(bind))));
for i=1:max(bind)
    bm(i) = mean(y(bind==i));
    bs(i) = std(y(bind==i));
    bsem(i) = bs(i)/sqrt(sum(bind==i));
end