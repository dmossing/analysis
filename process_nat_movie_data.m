%% load natural movie data
filebase = 'M7194_153_000';
roifold = '/home/mossing/excitation/2P/170907/M7194/';
roifile = [roifold filebase '.rois'];
load(roifile,'-mat','ROIdata','Data','Neuropil')
%% load stim timing data
resultfold = '/home/mossing/excitation/visual_stim/170907/M7194/';
infofold = '/home/mossing/excitation/2P/170907/M7194/';
resultfile = [resultfold filebase '.mat'];
infofile = [infofold filebase '.mat'];
load(resultfile)
load(infofile)
%% load running data
runfold = '/home/mossing/excitation/running/170907/M7194/';
[dx_dt,triggers] = process_run_data([runfold filebase '.bin']);
dxdt = resamplebytrigs(dx_dt,size(Data,2),find(triggers),info.frame);
%% trialize
info.frame = info.frame(5:40); % leave out initial trials and photobleached later trials
[corrected,baseline,neuropilMultiplier] = neuropilRR(Data,Neuropil);
trialwise = trialize(corrected,[info.frame(1:2:end) info.frame(2:2:end)],15,31);
trialrunning = trialize(dxdt,[info.frame(1:2:end) info.frame(2:2:end)],15,31);
%% save
save('trialwise000','trialwise','Data','Neuropil','neuropilMultiplier','trialrunning')