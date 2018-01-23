function msk = make_msk(rois)
% generates a d1xd2xN array of binary masks, indexed by ROI along the 3rd
% dim.
msk = [rois(:).mask];
msk = reshape(full(msk),size(rois(1).mask,1),size(rois(1).mask,2),[]);