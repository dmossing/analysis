function msk = make_np_msk(rois)
% generates a d1xd2xN array of binary masks, indexed by ROI along the 3rd
% dim.
msk = [rois(:).neuropilmask];
msk = reshape(full(msk),size(rois(1).neuropilmask,1),size(rois(1).neuropilmask,2),[]);