function [bm,bl,bu] = binxy_boot(x,y,binno,pctL,pctU)
[~,bind] = histc(x,linspace(min(x),max(x),binno));
[bm,bl,bu] = deal(zeros(numel(unique(bind)),size(y,2)));
for i=1:max(bind)
    bootstat = bootstrp(100,@mean,y(bind==i,:));
    bm(i,:) = prctile(bootstat,50);
    bl(i,:) = prctile(bootstat,pctL);
    bu(i,:) = prctile(bootstat,pctU);
end