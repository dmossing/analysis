function save_roi_files(foldname)
addpath(genpath('/home/mossing/Documents/code/downloads/EvansCode/'))
d = dir([foldname '/F_*_proc.mat']);
fnames = {d(:).name};
for i=1:numel(fnames)
    fnames{i} = [foldname '/' fnames{i}(1:end-9)]; % leave off "_proc.mat"
end
for i=1:numel(fnames)
    ROIdata = suite2P2ROIdata([fnames{i} '.mat']);
    Data = {ROIdata.rois(:).rawdata};
    Data = cell2mat(Data');
    Neuropil = {ROIdata.rois(:).rawneuropil};
    Neuropil = cell2mat(Neuropil');
    save([fnames{i} '.rois'],'-mat','ROIdata','Data','Neuropil')
end