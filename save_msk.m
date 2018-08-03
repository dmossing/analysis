function save_msk(foldname)
d = dir([foldname '/*.rois']);
for i=1:numel(d)
    roifile = load([foldname '/' d(i).name],'-mat');
    msk = make_msk(roifile.ROIdata.rois);
    save([foldname '/' d(i).name],'msk','-mat','-append')
end