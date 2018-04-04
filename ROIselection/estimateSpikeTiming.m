function [ROIdata, Spikes, ROIindex] = estimateSpikeTiming(ROIdata, NeuropilWeight, varargin)

saveOut = false;
saveFile = '';

ROIindex = [1 inf];
frameRate = 15.45;
directory = cd;


%% Check input arguments
if ~exist('ROIdata','var') || isempty(ROIdata)
    [ROIdata, p] = uigetfile({'*.mat'},'Choose ROI file',directory);
    if isnumeric(ROIdata)
        return
    end
    ROIdata = fullfile(p,ROIdata);
end

if ~exist('NeuropilWeight', 'var') || isempty(NeuropilWeight)
    NeuropilWeight = 0.65;
end

index = 1;
while index<=length(varargin)
    try
        switch varargin{index}
            case 'frameRate'
                frameRate = varargin{index+1};
                index = index + 2;
            case 'ROIindex'
                ROIindex = varargin{index+1};
                index = index + 2;
            case {'save', 'Save'}
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


%% Load ROI data
if ischar(ROIdata)
    ROIFile = ROIdata;
    load(ROIdata, 'ROIdata');
    if saveOut && isempty(saveFile)
        saveFile = ROIFile;
    end
end
if isempty(saveFile)
    saveOut = false;
end

if ~isfield(ROIdata.rois, 'rawneuropil')
    NeuropilWeight = false;
end

%% Determine ROIs to compute
if ischar(ROIindex)
    switch ROIindex
        case {'all', 'All'}
            ROIindex = 1:numel(ROIdata.rois);
        case {'new', 'New'}
            ROIindex = find(arrayfun(@(x) (isempty(x.rawspikes)), ROIdata.rois));
    end
elseif isnumeric(ROIindex) && ROIindex(end) == inf
    ROIindex = [ROIindex(1:end-1), ROIindex(end-1)+1:numel(ROIdata.rois)];
end
numROIs = numel(ROIindex);


%% Determine Neuropil Weight
if isempty(NeuropilWeight)
    NeuropilWeight = determineNeuropilWeight(ROIdata, ROIindex);
elseif numel(NeuropilWeight)==1
    NeuropilWeight = repmat(NeuropilWeight, numROIs, 1);
end


%% Estimate spike timing

% Initialize variables
V.dt = 1/frameRate;  % time step size
V.NCells = 1; % number of cells in each ROI

% Cycle through ROIs estimating spike timing
fprintf('Estimating spikes for %d neurons...\n\tFinished ROI:', numROIs);
Spikes = zeros(numROIs, numel(ROIdata.rois(1).rawdata));
for rindex = 1:numROIs
    if NeuropilWeight(rindex)
        Spikes(rindex,:) = fast_oopsi(ROIdata.rois(ROIindex(rindex)).rawdata - NeuropilWeight(rindex)*ROIdata.rois(ROIindex(rindex)).rawneuropil, V);
    else
        Spikes(rindex,:) = fast_oopsi(ROIdata.rois(ROIindex(rindex)).rawdata, V);
    end
    ROIdata.rois(rindex).rawspikes = Spikes(rindex,:);
    
    if ~mod(rindex, 5)
        fprintf('\t%d', rindex)
        if ~mod(rindex, 100)
            fprintf('\n\tFinished ROI:');
        end
    end
end
fprintf('\n');

%% Save to file
if saveOut
    if ~exist(saveFile, 'file')
        save(saveFile, 'ROIdata', '-mat', '-v7.3');
    else
        save(saveFile, 'ROIdata', '-mat', '-append');
    end
    if isequal(ROIindex, 1:numel(ROIdata.rois))
        save(saveFile, 'Spikes', '-mat', '-append');
    end
    fprintf('\tROIdata saved to: %s\n', saveFile);
end
