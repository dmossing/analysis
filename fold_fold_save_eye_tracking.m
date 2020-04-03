function fold_fold_save_eye_tracking(basefold,roibasefold,rg)
d = dir(basefold);
forbidden = {'.','..','Duplicate','.DS_Store','new_tree'};
dry_run = true;
for i=1:numel(d)
    foldname = d(i).name;
    if ~isnan(str2double(foldname))%~ismember(foldname,forbidden) %&& ~contains(foldname,'eye_tracking')
        in_bounds = str2num(foldname) >= rg(1) && str2num(foldname) <= rg(2);
        if in_bounds
            subfold_save_tracks([basefold '/' foldname],[roibasefold '/' foldname],dry_run)
        end
    end
end
dry_run = false;
for i=1:numel(d)
    foldname = d(i).name;
    if ~isnan(str2double(foldname))%~ismember(foldname,forbidden) %&& ~contains(foldname,'eye_tracking')
        in_bounds = str2num(foldname) >= rg(1) && str2num(foldname) <= rg(2);
        if in_bounds
            subfold_save_tracks([basefold '/' foldname],dry_run)
        end
    end
end

function subfold_save_tracks(basefold,roibasefold,dry_run)
d = dir(basefold);
forbidden = {'.','..','Duplicate','.DS_Store','new_tree'};
for i=1:numel(d)
    foldname = d(i).name;
    if ~ismember(foldname,forbidden)
        fold_save_eye_tracking([basefold '/' foldname],[roibasefold '/' foldname],dry_run)
    end
end