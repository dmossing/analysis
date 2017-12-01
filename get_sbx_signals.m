function get_sbx_signals(filename)
ROIFile = [filename, '.rois'];
ROIdata = sbxDistribute(filename, 'Save', 'SaveFile', ROIFile); % intialize struct
createMasks(ROIdata, 'Save', 'SaveFile', ROIFile); % create ROI masks
config = load2PConfig([filename, '.sbx']);
[~, Data, Neuropil, ~] = extractSignals([filename,'.sbx'], ROIFile, 'all', 'Save', 'SaveFile', ROIFile, 'MotionCorrect', [filename,'.align'], 'Frames', 1:config.Frames-1);
save(ROIFile, 'Data', 'Neuropil', '-mat', '-append');