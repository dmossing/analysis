function save_eye_tracking(eyefoldname,roifoldname)
% eyefoldname an eye_tracking_00x.mat file
% roifoldname a .rois filename
load(eyefoldname,'ctr_sm','area_sm')
if exist('ctr_sm','var')
    pupil_ctr = ctr_sm;
    pupil_area = area_sm;
    save(roifoldname,'-mat','pupil_ctr','pupil_area','-append')
else
    disp('info not saved')
end