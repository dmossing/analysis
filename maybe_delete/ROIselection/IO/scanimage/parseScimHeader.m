function config = parseScimHeader(TifFile)
warning('off','MATLAB:imagesci:tiffmexutils:libtiffWarning');

% config.version = 1;

%% Check input arguments
narginchk(0,1);
if ~exist('TifFile', 'var') || isempty(TifFile) % Prompt for file selection
    [TifFile, p] = uigetfile({'*.tif'}, 'Select ''tif'' files to load header info from:', directory);
    if isnumeric(TifFile)
        return
    end
    TifFile = fullfile(p, TifFile);
end


%% Set identifying info
config.type = 'scim';
config.FullFilename = TifFile;
[~, config.Filename, ~] = fileparts(TifFile);

%% Identify header information from file

% Load header
header = scim_openTif(TifFile);
config.header = {header};

% Save important information
if isfield(header,'scanimage') && header.scanimage.VERSION_MAJOR == 5;
    config.Height = header.scanimage.linesPerFrame;
    config.Width = header.scanimage.pixelsPerLine;
    config.Depth = header.scanimage.stackNumSlices;
    config.ZStepSize = header.scanimage.stackZStepSize;
    config.Channels = size(header.scanimage.channelsSave,1);
    config.FrameRate = 1 / header.scanimage.scanFramePeriod;
    config.ZoomFactor = header.scanimage.zoomFactor;
    config.Frames = header.scanimage.acqNumFrames;
else
    config.Height = header.acq.linesPerFrame;
    config.Width = header.acq.pixelsPerLine;
    config.Depth = header.acq.numberOfZSlices;
    config.ZStepSize = header.acq.zStepSize;
    config.Channels = header.acq.numberOfChannelsSave;
    config.FrameRate = header.acq.frameRate;
    config.ZoomFactor = header.acq.zoomFactor;
    
    % Determine number of frames
    info = imfinfo(TifFile);
    config.Frames = numel(info)/(config.Channels*config.Depth);
end

%% DEFAULTS
% config.MotionCorrected = false;
% config.info = [];
config.Precision = 'uint16';
config.DimensionOrder = {'Height','Width','Channels','Frames'}; % default
config.Colors = {'green', 'red'};
config.size = sizeDimensions(config);


