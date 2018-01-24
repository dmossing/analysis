function config = parseSbxHeader(File)
% File can be a .mat info file, a .sbx file, a directory to prompt from, or
% an empty matrix to initiate file selection from the default directory

% config.version = 1;

%% Check input arguments
narginchk(0,1);
if ~exist('File', 'var') || isempty(File)
    directory = loadCanalSettings('DataDirectory');
    [File, p] = uigetfile({'.sbx;.mat'}, 'Choose sbx file', directory);
    if isnumeric(File)
        return
    end
    File = fullfile(p, File);
elseif isdir(File)
    [File, p] = uigetfile({'.sbx;.mat'}, 'Choose sbx file', File);
    if isnumeric(File)
        return
    end
    File = fullfile(p, File);
end


%% Load in header
[~, ~, e] = fileparts(File);
switch e
    case '.mat'
        ConfigFile = File;
        temp = sbxIdentifyFiles(File);
        SbxFile = temp{1};
    case '.sbx'
        SbxFile = File; % assumes sbx file to have same name and be located on same path
        temp = sbxIdentifyFiles(File);
        ConfigFile = temp{1};
end


%% Set identifying info
config.type = 'sbx';
config.FullFilename = SbxFile;
[~, config.Filename, ~] = fileparts(config.FullFilename);


%% Identify header information from file

% Load header
load(ConfigFile, 'info');

% Save header
config.header = {info};

% Save frame dimensions
if info.scanmode
    config.Height = info.recordsPerBuffer;
else
    config.Height = 2*info.recordsPerBuffer;
end
if isfield(info,'scanbox_version') && info.scanbox_version >= 2
    try
        config.Width = info.sz(2);
    catch
        config.Width = 796;
    end
else
    info.scanbox_version = 1;
    info.Width = info.postTriggerSamples;
end

% Determine # of channels
switch info.channels
    case 1
        config.Channels = 2;      % both PMT0 & 1
    case 2
        config.Channels = 1;      % PMT 0
    case 3
        config.Channels = 1;      % PMT 1
end

% Determine # of frames
d = dir(SbxFile);
config.Frames =  d.bytes/(config.Height*config.Width*config.Channels*2); % "2" b/c assumes uint16 encoding => 2 bytes per sample
    
% Determine magnification
config.ZoomFactor = info.config.magnification;

% Determine frame rate
if info.scanbox_version>=2
    try
        if info.scanmode == 1
            config.FrameRate = 15.49; % dependent upon mirror speed
        else
            config.FrameRate = 30;
        end
    catch
        config.FrameRate = 15.49;
    end
else
    config.FrameRate = 15.49;
end
            
%% DEFAULTS
% config.Processing = {};
% config.info = [];
% config.MotionCorrected = false;
try 
    config.Depth = info.otparam(3); %should be (3), testing what happens if this goes back
catch
    config.Depth = 1; % current default
end
config.ZStepSize = 0; % current default
config.Precision = 'uint16'; % default
config.DimensionOrder = {'Channels','Width','Height','Frames','Depth'}; % default
config.Colors = {'green', 'red'};
config.size = [config.Height, config.Width, config.Depth, config.Channels, config.Frames];
