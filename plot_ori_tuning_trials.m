function plot_ori_tuning_trials(response)
ncell = size(response,1);
nori = size(response,3);
for i=1:ncell
    r = response(i,:,:,:);
    r = r(:);
    for j=1:nori
        subplot(ncell,nori,(i-1)*nori+j)
        plot(squeeze(response(i,:,j,:)))
        ylim([min(r) max(r)])
        axis off
    end
end