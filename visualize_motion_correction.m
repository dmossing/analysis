function [offsets,h] = visualize_motion_correction(T)
offmax = prctile(abs(T(:)),99);
offsets = -offmax:offmax;
h = hist3(T,{offsets,offsets});
figure
imagesc(offsets,offsets,log10(h))