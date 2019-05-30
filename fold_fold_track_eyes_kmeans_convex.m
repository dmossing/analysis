function fold_fold_track_eyes_kmeans_convex(basefold,rg)
d = dir(basefold);
forbidden = {'.','..','Duplicate'};
for i=1:numel(d)
    foldname = d(i).name;
    if ~ismember(foldname,forbidden) %&& ~contains(foldname,'eye_tracking')
        in_bounds = str2num(foldname) >= rg(1) && str2num(foldname) <= rg(2);
        if in_bounds
            subfold_track_eyes_kmeans_convex([basefold '/' foldname])
        end
    end
end

function subfold_track_eyes_kmeans_convex(basefold)
d = dir(basefold);
forbidden = {'.','..','Duplicate'};
for i=1:numel(d)
    foldname = d(i).name;
    if ~ismember(foldname,forbidden)
        fold_track_eyes_kmeans_convex([basefold '/' foldname])
    end
end