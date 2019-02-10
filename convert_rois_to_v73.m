function convert_rois_to_v73(foldname)
d = dir([foldname '/*.rois']);
for i=1:numel(d)
    i
    roifile = load([foldname '/' d(i).name],'-mat');
    save([foldname '/' d(i).name],'-struct','roifile','-v7.3')
end