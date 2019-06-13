function save_suite2p_mean_images(foldname)
%%
old_fold = pwd;
cd(foldname)
%%
load('Fall.mat','ops')
% subplot(1,2,1)
% imagesc(ops.meanImg)
% subplot(1,2,2)
% imagesc(ops.meanImg_chan2)
%%
figure
r = ops.meanImg_chan2;
r = r/max(r(:));
g = ops.meanImg;
g = g/max(g(:));
rgb = cat(3,r,g,zeros(size(r)));
imshow(rgb)
%%
green_vals = ops.meanImg(:);
red_vals = ops.meanImg_chan2(:);
% figure
% scatter(green_vals,red_vals)
p = polyfit(green_vals,red_vals,1);
% hold on
% extrema = [min(green_vals) max(green_vals)];
% plot(extrema,p(1)*extrema)
% hold off
bleedthru = p(1);
%%
r = ops.meanImg_chan2 - ops.meanImg*bleedthru;
r = r/max(r(:));
g = ops.meanImg;
g = g/max(g(:));
rgb = cat(3,r,g,zeros(size(r)));
imshow(rgb)
%%
red_mean = ops.meanImg_chan2 - ops.meanImg*bleedthru;
green_mean = ops.meanImg;
save('mean_images','red_mean','green_mean')
%%
cd(old_fold)