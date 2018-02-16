function save_trialwise(foldname)
fnames = dir([foldname '/*.rois']);
fnames = {fnames(:).name};
for i=1:numel(fnames)
    name = fnames{i};
    load(name,'-mat','corrected')
    load(strrep(name,'.rois','.mat'),'info')
    trialwise = trialize(corrected,info.frame(info.event_id==1),15,30);
    save(name,'-mat','-append','trialwise')
end