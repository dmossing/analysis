function fold_fold_track_eyes_kmeans_convex(basefold,rg)
d = dir(basefold);
forbidden = {'.','..','Duplicate'};
dry_run = true;
for i=1:numel(d)
    foldname = d(i).name;
    if ~ismember(foldname,forbidden) %&& ~contains(foldname,'eye_tracking')
        in_bounds = str2num(foldname) >= rg(1) && str2num(foldname) <= rg(2);
        if in_bounds
            subfold_track_eyes_kmeans_convex([basefold '/' foldname],dry_run)
        end
    end
end
dry_run = false;
for i=1:numel(d)
    foldname = d(i).name;
    if ~ismember(foldname,forbidden) %&& ~contains(foldname,'eye_tracking')
        in_bounds = str2num(foldname) >= rg(1) && str2num(foldname) <= rg(2);
        if in_bounds
            subfold_track_eyes_kmeans_convex([basefold '/' foldname],dry_run)
        end
    end
end

function subfold_track_eyes_kmeans_convex(basefold,dry_run)
d = dir(basefold);
forbidden = {'.','..','Duplicate'};
for i=1:numel(d)
    foldname = d(i).name;
    if ~ismember(foldname,forbidden)
        fold_track_eyes_kmeans_convex([basefold '/' foldname],dry_run)
    end
end