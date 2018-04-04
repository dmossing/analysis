function [Data, FileIndex, ROIindex, Labels, Tag, ROIs] = gatherROIdata(ROIs, FieldName, Index, Label, ROIindex, FileIndex)
% ROIs is a cell array of strings specifying the filenames of the ROI
% files, or it is a cell array of ROIdata objects
% FieldName is a field name of 'ROIdata.rois'
% Index is a number indexing in that field (e.g. '1' or use '':'' for all elements)
% Label is a cell array of strings of ROIs to gather data from, or false if
% gathering data from ROIs with any label (use 'none' for unlabeled)
% ROIindex is a list of roi indices, correspondings to the files listed in
% FileIndex

%% Check input arguments
if ~exist('ROIs', 'var') || isempty(ROIs)
    directory = cd;
    [ROIs, p] = uigetfile({'*.mat'},'Choose ROI file(s)', directory, 'MultiSelect', 'on');
    if isnumeric(ROIs)
        return
    elseif iscell(ROIs)
        for findex = 1:numel(ROIs)
            ROIs{findex} = fullfile(p, ROIs{findex});
        end
    elseif ischar(ROIs)
        ROIs = {fullfile(p, ROIs)};
    end
end

if ~exist('FieldName', 'var') || isempty(FieldName)
    error('Requires specific field name to be specified to extract information');
end

if ~exist('Index', 'var') || isempty(Index)
    Index = ':';
end

if ~exist('Label', 'var') || isempty(Label)
    Label = false;
end

if ~exist('ROIindex', 'var') || isempty(ROIindex)
    ROIindex = 'all';
end

%% Load data
numFiles = numel(ROIs);
if iscellstr(ROIs)
    ROIFiles = ROIs;
    ROIs = cell(numFiles, 1);
    for findex = 1:numFiles
        load(ROIFiles{findex}, 'ROIdata', '-mat');
        ROIs{findex} = ROIdata;
    end
end

%% Label unlabeled ROIs
for findex = 1:numFiles
    unlabeledIndex = cellfun(@isempty,{ROIs{findex}.rois(:).label});
    if any(unlabeledIndex)
        [ROIs{findex}.rois(unlabeledIndex).label] = deal({'none'});
    end
end

%% Determine ROIs to gather data from
numROIs = zeros(numFiles, 1);
if ischar(ROIindex) && strcmp(ROIindex, 'all')
    numROIs = cellfun(@(x) (numel(x.rois)), ROIs); % vector of number of rois in each element of cell array
    ROIindex = [];
    FileIndex = [];
    for findex = 1:numFiles
        ROIindex = cat(2, ROIindex, 1:numROIs(findex));
        FileIndex = cat(2, FileIndex, findex*ones(1,numROIs(findex)));
    end
else
    for findex = 1:numFiles
        numROIs(findex) = sum(FileIndex==findex);
    end
end
totalROIs = sum(numROIs);

%% Determine data size & initialize outputs
temp = ROIs{FileIndex(1)}.rois(ROIindex(1)).(FieldName)(Index);
if isnumeric(temp)
    Data = nan(totalROIs, numel(temp));
    extraction = 'numeric';
else
    Data = cell(totalROIs, 1);
    if ischar(temp) || (iscell(temp) && numel(temp)>1)
        extraction = 'cell';
    elseif iscell(temp)
        extraction = 'numeric';
    end
end

%% Pull out data
Labels = cell(totalROIs, 1);
Tag = cell(totalROIs, 1);
for rindex = 1:totalROIs
    switch extraction
        case 'numeric'
            Data(rindex,:) = ROIs{FileIndex(rindex)}.rois(ROIindex(rindex)).(FieldName)(Index);
        case 'cell'
            Data{rindex} = ROIs{FileIndex(rindex)}.rois(ROIindex(rindex)).(FieldName)(Index);
    end
    Labels{rindex} = ROIs{FileIndex(rindex)}.rois(ROIindex(rindex)).label{1};
    Tag{rindex} = ROIs{FileIndex(rindex)}.rois(ROIindex(rindex)).tag;
end

%% Remove unwanted ROIs
if ~islogical(Label)
    unwanted = ~strcmp(Labels, Label);
    switch extraction
        case 'numeric'
            Data(unwanted,:) = [];
        case 'cell'
            Data(unwanted) = [];
    end
    FileIndex(unwanted) = [];
    ROIindex(unwanted) = [];
    Labels(unwanted) = [];
    Tag(unwanted) = [];
end