function ROIdata = computeDFoF(ROIdata, varargin)

saveOut = false;
saveFile = '';

NeuropilWeight = [];
ROIindex = [1 inf];
numFramesBaseline = [];
directory = cd;

%% Check input arguments
if ~exist('ROIdata','var') || isempty(ROIdata)
    [ROIdata, p] = uigetfile({'*.rois;*.mat'},'Choose ROI file',directory);
    if isnumeric(ROIdata)
        return
    end
    ROIdata = fullfile(p,ROIdata);
end

index = 1;
while index<=length(varargin)
    try
        switch varargin{index}
            case 'NeuropilWeight'
                NeuropilWeight = varargin{index+1};
                index = index + 2;
            case 'numFramesBaseline'
                numFramesBaseline = varargin{index+1};
                index = index + 2;
            case 'ROIindex'
                ROIindex = varargin{index+1};
                index = index + 2;
            case {'Save', 'save'}
                saveOut = true;
                index = index + 1;
            case {'SaveFile', 'saveFile'}
                saveFile = varargin{index+1};
                index = index + 2;
            otherwise
                warning('Argument ''%s'' not recognized',varargin{index});
                index = index + 1;
        end
    catch
        warning('Argument %d not recognized',index);
        index = index + 1;
    end
end

fprintf('Calculating trial-wise dF/F...');


%% Load ROI data
if ischar(ROIdata)
    ROIFile = ROIdata;
    load(ROIFile, 'ROIdata', '-mat');
    if saveOut && isempty(saveFile)
        saveFile = ROIFile;
    end
end
if saveOut && isempty(saveFile)
    warning('Cannot save output as no file specified');
    saveOut = false;
end
numROIs = numel(ROIdata.rois);
[H, W] = size(ROIdata.rois(1).data);


%% Determine ROIs to analyze
if ischar(ROIindex)
    switch ROIindex
        case {'all', 'All'}
            ROIindex = 1:numel(ROIdata.rois);
        case {'new', 'New'}
            ROIindex = find(arrayfun(@(x) (isempty(x.rawdata)), ROIdata.rois));
    end
elseif isnumeric(ROIindex) && ROIindex(end) == inf
    ROIindex = [ROIindex(1:end-1), ROIindex(end-1)+1:numel(ROIdata.rois)];
end
if iscolumn(ROIindex)
    ROIindex = ROIindex';
end


%% Determine Neuropil Weight (fluorescence can never be negative)
if isempty(NeuropilWeight)
    NeuropilWeight = determineNeuropilWeight(ROIdata, ROIindex);
elseif numel(NeuropilWeight)==1
    NeuropilWeight = repmat(NeuropilWeight, numROIs, 1);
end

% Record to struct
if ~isfield(ROIdata.DataInfo, 'NeuropilWeight')
    ROIdata.DataInfo.NeuropilWeight = NeuropilWeight;
else
    ROIdata.DataInfo.NeuropilWeight(ROIindex) = NeuropilWeight(ROIindex);
end


%% Determine number of frames to average for baseline
if isempty(numFramesBaseline)
    numFramesBaseline = ROIdata.DataInfo.numFramesBefore;
elseif numFramesBaseline > ROIdata.DataInfo.numFramesBefore
    numFramesBaseline = ROIdata.DataInfo.numFramesBefore;
end


%% Calculate dF/F

% Initialize output
[ROIdata.rois(ROIindex).dFoF] = deal(zeros(H,W));

% Compute dF/F for each trial for each ROI
for rindex = ROIindex
    
    % Extract all trials for current stimulus
    data = ROIdata.rois(rindex).data;
    if NeuropilWeight(rindex) % remove weighted Neuropil signal
        data = data - NeuropilWeight(rindex)*ROIdata.rois(rindex).neuropil;
    end
    
    % Compute Fluorescence baseline for each trial
    baseline = nanmedian(data(:, ROIdata.DataInfo.numFramesBefore - numFramesBaseline + 1:ROIdata.DataInfo.numFramesBefore), 2);
    
    % Compute dF/F signal for each trial
    ROIdata.rois(rindex).dFoF = bsxfun(@rdivide, bsxfun(@minus, data, baseline), baseline);
    
end

fprintf('\tComplete\n');


%% Save to file
if saveOut
    if ~exist(saveFile, 'file')
        save(saveFile, 'ROIdata', '-mat', '-v7.3');
    else
        save(saveFile, 'ROIdata', '-mat', '-append');
    end
    fprintf('\tROIdata saved to: %s\n', saveFile);
end