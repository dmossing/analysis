function [Image, Origin, hA] = spatialOverlay(ROIs, Data, ROIindex, FileIndex, ColorIndex, Labels, Colors, Brightness, varargin)
%spatialOverlay    Overlay ROI data onto an image
% [Image, Origin, hA] = spatialOverlay(ROIs, Data, ROIindex, FileIndex, ColorIndex, Labels, Colors, Brightness)
%
% Labels - for discrete data, labels is a cell array of N strings
% corresponding to the <= N unique values in ColorIndex, and the N unique
% colors.
% Labels - for continuous data, labels is a 2x1 vector specifying the lower
% and upper bounds of the colors (the same thing as CLim)

% Data
DataType = 'continuous'; % 'discrete' or 'continuous'
Radius = [];

% Display Properties
transparency = 1; % value from 0 to 1
ImgToDisplay = 'average'; % 'average' or 'none' or 'var' or 'max' or 'min'
mergetype = 'quick'; % 'quick' or 'pretty'
showColorBar = false;
colorbarLabel = '';
Crop = false;
Title = '';
AdjustEdgeBrightness = true;

% Default variables
Origin = [];
hA = [];
Image = [];
CMap = [];
Map = [];

directory = cd;

%% Parse input arguments
if ~exist('ROIs','var') || isempty(ROIs)
    directory = CanalSettings('DataDirectory');
    [ROIs, p] = uigetfile({'*.mat'},'Choose ROI file(s)', directory, 'MultiSelect', 'on');
    if isnumeric(ROIs)
        return
    end
    if iscell(ROIs)
        for findex = 1:numel(ROIs)
            ROIs{findex} = fullfile(p,ROIs{findex});
        end
    elseif ischar(ROIs)
        ROIs = {fullfile(p,ROIs)};
    end
elseif ischar(ROIs)
    ROIs = {fullfile(p,ROIs)};
end

if ~exist('Data','var') || isempty(Data)
    [Data, p] = uigetfile({'*.mat;*.ciexp'},'Choose Experiment file(s)', directory, 'MultiSelect', 'on');
    if isnumeric(Data)
        return
    end
    if iscell(Data)
        for findex = 1:numel(Data)
            Data{findex} = fullfile(p,Data{findex});
        end
    elseif ischar(Data)
        Data = {fullfile(p,Data)};
    end
elseif ischar(Data)
    Data = {Data};
end

index = 1;
while index<=length(varargin)
    switch varargin{index}
        case 'DataType'
            DataType = varargin{index+1};
            index = index + 2;
        case 'Radius'
            Radius = varargin{index+1};
            index = index + 2;
        case 'Crop'
            Crop = varargin{index+1};
            index = index + 2;
        case 'Image'
            Image = varargin{index+1};
            index = index + 2;
        case 'Colormap'
            CMap = varargin{index+1};
            index = index + 2;
        case 'Map'
            Map = varargin{index+1};
            index = index + 2;
        case 'Origin'
            Origin = varargin{index+1};
            index = index + 2;
        case 'transparency'
            transparency = varargin{index+1};
            index = index + 2;
        case 'ImgToDisplay'
            ImgToDisplay = varargin{index+1};
            index = index + 2;
        case 'mergetype'
            mergetype = varargin{index+1};
            index = index + 2;
        case 'colorbarLabel'
            colorbarLabel = varargin{index+1};
            index = index + 2;
        case 'axes'
            hA = varargin{index+1};
            index = index + 2;
        case 'showColorBar'
            showColorBar = true;
            index = index + 1;
        case 'Title'
            Title = varargin{index+1};
            index = index + 2;
        otherwise
            warning('Argument ''%s'' not recognized',varargin{index});
            index = index + 1;
    end
end


%% Load ROI data
numFiles = numel(ROIs);
if iscellstr(ROIs)
    ROIFiles = ROIs;
    ROIs = cell(numFiles, 1);
    for findex = 1:numFiles
        load(ROIFiles{findex}, 'ROIdata', '-mat');
        ROIs{findex} = ROIdata;
    end
end


%% Determine ROIs to overlay
if ~exist('ROIindex', 'var') || isempty(ROIindex) || (ischar(ROIindex) && strcmp(ROIindex, 'all'))
    ROIindex = [1, inf];
end
if ROIindex(end) == inf
    totalROIs = 0;
    for findex = 1:numFiles
        totalROIs = totalROIs + numel(ROIs{findex}.rois);
    end
    ROIindex = cat(2, ROIindex(1:end-1), ROIindex(1:end-1)+1:totalROIs);
end
numROIs = numel(ROIindex);

if ~exist('FileIndex', 'var') || isempty(FileIndex)
    FileIndex = ones(numROIs, 1);
end


%% Build colormap
if ~exist('Colors', 'var') || isempty(Colors)
    [Vals, ~, ColorIndex] = unique(ColorIndex);
    Colors = jet(numel(Vals));
end
if ~exist('Brightness', 'var') || isempty(Brightness)
    Brightness = ones(numROIs, 3);
else
    Brightness = bsxfun(@times, ones(numROIs, 3), Brightness);
end


%% Determine colormap labels
switch DataType
    case 'discrete'
        if ~exist('Labels', 'var') || isempty(Labels)
            Labels = flip(cellstr(num2str((1:size(Colors,1))')));
        else
            Labels = flip(Labels);
        end
        Colors = flip(Colors);
        ColorIndex = abs(ColorIndex - max(ColorIndex) - 1);
    case 'continuous'
        if ~exist('Labels', 'var') || isempty(Labels)
            Labels = [min(ColorIndex), max(ColorIndex)];
        end
        Labels = cellstr(num2str((Labels(1):(Labels(2)-Labels(1))/10:Labels(2))'));
end


%% Load Image data
if iscellstr(Data)
    ExperimentFiles = Data;
    clear Data;
    if ~iscell(ImgToDisplay) && any(strcmp(ImgToDisplay, {'average', 'variance', 'max', 'min'}))
        for findex = 1:numFiles
            Data(findex) = load(ExperimentFiles{findex}, 'Map', 'ImageFiles', '-mat');
        end
    else
        for findex = 1:numFiles
            Data(findex) = load(ExperimentFiles{findex}, 'Map', '-mat');
        end
    end
    for findex = 1:numFiles
        Data(findex).file = ExperimentFiles{findex};
        if ~isfield(Data, 'Map')
            Data(findex).Map = [];
        end
        Data(findex).origMap = Data(findex).Map;
        Data(findex).cropped = false;
    end
end


%% Build Image
if isempty(Image)
    [Image, newOrigin, ~, Data] = createMultiFoVImage(Data, ImgToDisplay, mergetype, Crop, Map);
    if newOrigin ~= false
        Origin = newOrigin;
    end
end


%% Determine Offsets & Shift ROIs
Shift = zeros(numFiles, 2);
if numFiles > 1
    for findex = 1:numFiles
        Shift(findex,:) = [Data(findex).Map.XWorldLimits(1)-Origin(1), Data(findex).Map.YWorldLimits(1)-Origin(2)];
    end
    for rindex = 1:numROIs
        ROIs{FileIndex(rindex)}.rois(ROIindex(rindex)).vertices =...
            ROIs{FileIndex(rindex)}.rois(ROIindex(rindex)).vertices +...
            repmat(Shift(FileIndex(rindex),:), size(ROIs{FileIndex(rindex)}.rois(ROIindex(rindex)).vertices,1), 1);
    end
end


%% Display Image

% Select axes
if isempty(hA)
    figure();
    hA = axes();
else
    axes(hA);
end

% Display image
switch ImgToDisplay
    case 'none'
        imshow(Image);
    otherwise
        if size(Image,3)==1
            if isempty(CMap)
                CMap = gray(250);
            end
            Image = gray2ind(mat2gray(Image), size(CMap,1));
        end
        image(Image);
        if size(Image,3)==1
            colormap(CMap)
        end
end

% Plot overlay
hold on
for rindex = 1:numROIs
    if isempty(Radius)
        vertices = ROIs{FileIndex(rindex)}.rois(ROIindex(rindex)).vertices;
    else
        vertices = circle(ROIs{FileIndex(rindex)}.rois(ROIindex(rindex)).centroid, Radius(rindex));
    end
    hP = patch(vertices(:,1),...
        vertices(:,2),...
        Colors(ColorIndex(rindex),:).*Brightness(rindex,:),...
        'FaceAlpha',transparency,...
        'EdgeAlpha',transparency);
    if AdjustEdgeBrightness
        hP.EdgeColor = Colors(ColorIndex(rindex),:).*Brightness(rindex,:);
    else
        hP.EdgeColor = Colors(ColorIndex(rindex),:);
    end
end
axis off

% Plot colorbar
if showColorBar
    cmap = colormap;            % determine colormap of image
    cbH = colorbar;             % place colorbar
    YLim = get(cbH, 'Limits');  % determine colorbar limits
    NewCMap = cat(1,cmap,Colors); % concatenate plot colors to colormap
    colormap(NewCMap);          % set new colormap
    HeightNewCMapInColorbar = size(Colors,1)*(YLim(2)/size(NewCMap,1)); % determine portion of colorbar taken by colors that were added
    NewYLim = [YLim(2)-HeightNewCMapInColorbar, YLim(2)]; % limit the colormap to this range
    Split = (NewYLim(2)-NewYLim(1))/numel(Labels); % determine the distance on the colorbar between two colors
    switch DataType
        case 'discrete'
            set(cbH, 'Limits', NewYLim, 'FontSize', 20, 'Ticks', NewYLim(1)+Split/2:Split:NewYLim(2), 'YTickLabel', Labels);
        case 'continuous'
            set(cbH, 'Limits', NewYLim, 'FontSize', 20, 'Ticks', NewYLim(1):(NewYLim(2)-NewYLim(1))/(numel(Labels)-1):NewYLim(2), 'YTickLabel', Labels);
    end
    % set(cbH, 'Ticks', YLim(1):(YLim(2)-YLim(1))/(numel(Labels)-1):YLim(2), 'YTickLabel', Labels);
    if ~isempty(colorbarLabel)
        ylabel(cbH, colorbarLabel, 'FontSize', 20);
    end
    
end

% Display title
if ~isempty(Title)
    title(Title);
end

end

function vertices = circle(centroid,r)
th = (0:pi/50:2*pi)';
vertices = [r * cos(th) + centroid(1), r * sin(th) + centroid(2)];
end
