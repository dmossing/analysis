function fold_getSignals(foldname)
d = dir([foldname '/*.sbx']);
fns = {d(:).name};
for i=1:numel(fns)
    getSignals(fns{i}(1:end-4));
end