function save_roilines(foldname)
d = dir([foldname '/*.rois']);
for i=1:numel(d)
    roifile = load([foldname '/' d(i).name],'-mat');
    msk = roifile.msk;
    ctr = center_of_mass(msk);
    save([foldname '/' d(i).name],'ctr','-mat','-append')
end