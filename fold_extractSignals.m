function fold_getSignals(foldname)
d = dir([foldname '/*.sbx']);
fns = {d(:).name};
for i=1:numel(fns)
    ROIFile = strrep(fns{i},'.sbx','.rois');
    [~, Data, Neuropil, ~] = extractSignals(fns{i}, ROIFile, 'all', 'Save', 'SaveFile', ROIFile, 'MotionCorrect', [fns{i},'.align'], 'Frames', Frames);
    save(ROIFile, 'Data', 'Neuropil', '-mat', '-append');
end