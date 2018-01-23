function [x,y] = tuning_cv_xy(orilist,ori_cv);
ROIno = size(ori_cv,1);
r = max([ori_cv ori_cv(:,1)],0)';
% r = max([ori_cv ori_cv(:,1)],0)';
x = r.*repmat(cos(pi/180*[orilist orilist(1)]),ROIno,1)';
y = r.*repmat(sin(pi/180*[orilist orilist(1)]),ROIno,1)';