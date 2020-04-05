function save_eye_tracking(eyefoldname,roifoldname,msk)
% eyefoldname an eye_tracking_00x.mat file
% roifoldname a .rois filename
load(eyefoldname,'ctr_sm','area_sm')
msk_area = sum(msk(:));
msk_width = 2*sqrt(msk_area/pi); % diameter, assuming a circular mask
[xx,yy] = meshgrid(1:size(msk,2),1:size(msk,1));
msk_ctrx = mean(xx.*msk);
msk_ctry = mean(yy.*msk);
msk_ctr = [msk_ctrx msk_ctry];
if exist('ctr_sm','var')
    pupil_ctr = ctr_sm';
    pupil_frac_ctr = (ctr_sm - repmat(msk_ctr,size(ctr_sm,1),1))/msk_width;
    pupil_area = area_sm';
    pupil_frac_area = area_sm/msk_area;
    save(roifoldname,'-mat','pupil_ctr','pupil_frac_ctr','pupil_area','pupil_frac_area','-append')
else
    disp('info not saved')
end