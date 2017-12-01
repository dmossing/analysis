
t = 0:size(stimresponse,2)-1;
N = size(stimresponse,1);
%% get stim info
% stimfoldname = '//adesnik2.ist.berkeley.edu/Excitation/mossing/visual_stim/170726/M7199/';
stimfoldbase = '/home/mossing/excitation/visual_stim/';
stimfoldname = [stimfoldbase subfold];
load([stimfoldname filebase '.mat'],'result')
ori = result.stimParams(1,:);













































































contrast = result.stimParams(5,:);
% gcontrast = result.stimParams(6,:);
% for k=1:N
%     subplot(10,13,k)
%     imagesc(squeeze(trialwise(k,sort_by([gcontrast(:) ori(:)]),:)))
%     hold on
%     plot([1 1],[1 120],'m')
%     plot([16 16],[1 120],'m')
%     hold off
% end
 % ,'gcontrast'
%% save masks
msk = zeros(numel(ROIdata.rois),512,796);
for i=1:numel(ROIdata.rois)
    msk(i,:,:) = ROIdata.rois(i).mask;
end
%% save running data
runfoldbase = '/home/mossing/excitation/running/'; 
runfoldname = [runfoldbase subfold];
[dx_dt,triggers] = process_run_data([runfoldname filebase '.bin']);
%% align running data to ca
dxdt = resamplebytrigs(dx_dt,size(Data,2),find(triggers),info.frame);
trialrunning = trialize(dxdt',[info.frame(1:2:end) info.frame(2:2:end)],15,31);
%% load roi data to save
trigroi = load('roi_info_004');
trigmsk = trigroi.msk;
%% fuck MATLAB; using Python instead
save([imfoldname 'trialwise159'],'trialwise','ori','contrast','Data','Neuropil','neuropilMultiplier','dxdt','trialrunning') % ,'msk','trigmsk') % 'dx_dt','triggers'