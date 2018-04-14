function fold_getSignals(foldname)
d = dir([foldname '/*.segment']);
fnames = {d(:).name};
for i=1:numel(fnames)
    filebase = strsplit(fnames{i},'.segment');
    filebase = filebase{1};
    getSignals([foldname '/' filebase]);
end