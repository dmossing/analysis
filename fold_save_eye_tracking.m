function fold_save_eye_tracking(foldname_eyes,foldname_rois,dry_run)
if nargin < 3
    dry_run = false;
end
d = dir(foldname_eyes);
forbidden = {'.','..','Duplicate','.DS_Store','new_tree'};
if ~dry_run
    for i=1:numel(d)
        drois = dir([foldname_rois '/ot/M*_*_' d(i).name '_ot_000.rois']);
        if ~ismember(d(i).name,forbidden) && ~contains(d(i).name,'eye_tracking')
            eyefoldname = [foldname_eyes '/eye_tracking_' d(i).name '.mat'];
            if exist(eyefoldname)
                i
                roifoldname = [foldname_rois '/ot/' drois(i).name];
                save_eye_tracking(eyefoldname,roifoldname);
            end
        end
    end
end