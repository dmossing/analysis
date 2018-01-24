function Config = load2PConfig(DataFiles)

directory = cd;

%% Check input arguments
narginchk(0,1);
if ~exist('DataFiles', 'var') || isempty(DataFiles)
    [DataFiles,p] = uigetfile({'*.sbx;*.tif;*.imgs'}, 'Choose images file(s) to load', directory, 'MultiSelect', 'on');
    if isnumeric(DataFiles)
        Images = []; return
    elseif iscellstr(DataFiles)
        for index = 1:numel(DataFiles)
            DataFiles{index} = fullfile(p,DataFiles{index});
        end
    else
        DataFiles = {fullfile(p,DataFiles)};
    end
elseif ~iscell(DataFiles)
    DataFiles = {DataFiles};
end

%% Load in config information
numFiles = numel(DataFiles);
ext = cell(numFiles, 1);
for index = 1:numFiles
    [~,~,ext{index}] = fileparts(DataFiles{index});
    switch ext{index}
        case '.sbx'
            Config(index) = parseSbxHeader(DataFiles{index});
        case '.tif'
            Config(index) = parseScimHeader(DataFiles{index});
        case '.imgs'
            Config(index) = parseImgsHeader(DataFiles{index});
        otherwise
            warning('File type %s not recognized...', ext{index});
            Config = [];
    end
end

% %% Determine lumped data values
% if numFiles == 1
%     
%     Config = Headers;
%     Config.header = [];
%     
% else
%     
%     Config.header = Headers;
%     Config.Filename = 'multiple';
%     Config.FullFilename = 'multiple';
%     
%     % Number of Frames
%     Config.Frames = sum([Headers(:).Frames]);
%     
%     % Frame Rate
%     temp = [Headers(:).FrameRate];
%     if ~all(temp == temp(1))
%         warning('Files do not have the same frame rates, using most common frame rate');
%     end
%     Config.FrameRate = mode(temp);
%     
%     % Height
%     temp = [Headers(:).Height];
%     if ~all(temp == temp(1))
%         warning('Files do not have the same height. Loading these files will cause an error.');
%     end
%     Config.Height = temp(1);
%     
%     % Width
%     temp = [Headers(:).Width];
%     if ~all(temp == temp(1))
%         warning('Files do not have the same width. Loading these files will cause an error.');
%     end
%     Config.Width = temp(1);
%     
%     % Depth
%     temp = [Headers(:).Depth];
%     if ~all(temp == temp(1))
%         warning('Files do not have the same depth. Loading these files will cause an error.');
%     end
%     Config.Depth = temp(1);
%     
%     % Channels
%     temp = [Headers(:).Channels];
%     if ~all(temp == temp(1))
%         warning('Files do not have the same number of channels. Loading these files will cause an error.');
%     end
%     Config.Channels = temp(1);
% 
%     % Z-stack info
%     Config.ZoomFactor = [Headers(:).ZoomFactor]; % current default
%     Config.ZStepSize = [Headers(:).ZStepSize]; % current default
%     
%     % Update file info (assumes all files are of same type)
%     Config.type = Headers(1).type;
%     Config.Precision = Headers(1).Precision;
%     Config.DimensionOrder = Headers(1).DimensionOrder;
%     
%     % Overall dimensions
%     Config.size = sizeDimensions(Config);
%     
%     % Config.Processing = {};
%     % Config.info = [];
%     % Config.MotionCorrected = {Headers(:).MotionCorrected};
%     % Config.Colors = [];
% end