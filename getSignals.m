function getSignals(fname)
addpath(genpath('~/Documents/code/downloads/EvansCode'))
addpath(genpath('~/Documents/code/downloads/ROISelection'))
i = 1;
fns{i} = fname;
ROIFile = [fns{i}, '.rois'];
ROIdata = sbxDistribute(fns{i}, 'Save'); % , 'SaveFile', ROIFile); % intialize struct
createMasks(ROIdata, 'Save', 'SaveFile', ROIFile); % create ROI masks
if isempty(strfind(fns{i},'depth'))
    config = load2PConfig([fns{i}, '.sbx']);
    [~, Data, Neuropil, ~] = extractSignals([fns{i},'.sbx'], ROIFile, 'all', 'Save', 'MotionCorrect', [fns{i},'.align'], 'Frames', 1:config.Frames-1); % 'SaveFile', ROIFile, 
else
    strbase = strsplit(fns{i},'_depth');
    Depth = str2num(strbase{2}(1));
    strbase = strbase{1};
    config = load2PConfig([strbase, '.sbx']);
%     Frames = idDepth([strbase,'.sbx'],[],'Depth',1)';
    [~, Data, Neuropil, ~] = extractSignals([strbase,'.sbx'], ROIFile, 'all', 'Save', 'MotionCorrect', [fns{i},'.align'], 'Depth',Depth); % 'SaveFile', ROIFile, % 'Frames', 1:config.Frames-1,
end
save(ROIFile, 'Data', 'Neuropil', '-mat', '-append');