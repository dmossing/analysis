function [Images, loadObj, Config] = load2P(ImageFiles, varargin)
% Loads 'Frames' of single .sbx, .tif, or .imgs file. Requires
% corresponding information file ('InfoFile').
% INPUTS:
% 'ImageFiles' -> cell array of strings of filenames to load images from,
% string specifying single filename to load images from, or blank to prompt
% for file selection.
% ARGUMENTS:
% 'Type' -> follow with 'Direct' to load frames directly to RAM, or
% 'MemMap' to use memory mapping. 
% 'Frames' -> follow with vector specifying the frame indices to load, cell
% array of a single vector specifying frame indices to load from each file,
% or cell array of N vectors specifying the exact frame indices to load
% from each of the N filenames input. For any vector, if last value is
% 'inf' then it will load all frames. (default is [1, inf] which will load
% all frames)
% 'Channels' -> follow with vector of channel indices to load
% 'Depths' -> follow with vector of depth indices to load
% 'Double' -> makes output Images of class double

LoadType = 'Direct'; % 'MemMap' or 'Direct' or 'Buffered' 
Frames = [1, inf]; % indices of frames to load in 'Direct' mode, or 'all'
Channels = inf;
Double = false;
SaveToMat = false;
Verbose = true;

directory = cd;

%% Initialize Parameters
if ~exist('ImageFiles', 'var') || isempty(ImageFiles)
    [ImageFiles,p] = uigetfile({'*.sbx;*.tif;*.imgs'}, 'Choose images file(s) to load', directory, 'MultiSelect', 'on');
    if isnumeric(ImageFiles)
        Images = []; return
    elseif iscellstr(ImageFiles)
        for index = 1:numel(ImageFiles)
            ImageFiles{index} = fullfile(p,ImageFiles{index});
        end
    else
        ImageFiles = {fullfile(p,ImageFiles)};
    end
elseif ischar(ImageFiles)
    ImageFiles = {ImageFiles};
elseif isstruct(ImageFiles)
    loadObj = ImageFiles;
end
numFiles = numel(ImageFiles);

index = 1;
while index<=length(varargin)
    try
        switch varargin{index}
            case isstruct(varargin{index})
                loadObj = varargin{index};
                index = index + 1;
            case {'Type','type'}
                LoadType = varargin{index+1};
                index = index + 2;
            case {'Frames','frames','Frame','frame'} % indices of frames to load in 'Direct' mode
                Frames = varargin{index+1};
                index = index + 2;
            case {'Channels', 'channels', 'Channel', 'channel'}
                Channels = varargin{index+1};
                index = index + 2;
            case {'Depths', 'depths', 'Depth', 'depth'}
                Depths = varargin{index+1};
                index = index + 2;
            case {'Double', 'double'}
                Double = true;
                index = index + 1;
            case {'Verbose', 'verbose'}
                Verbose = varargin{index+1};
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

if ischar(Frames) && strcmp(Frames, 'all')
    Frames = [1 inf];
elseif isnumeric(Frames) && isvector(Frames) && size(Frames,1)~=1
    Frames = Frames';
end

if ~exist('loadObj', 'var')
    % Initialize LoadObj
    for findex = 1:numFiles
        loadObj.files(findex).FullFilename = ImageFiles{findex};
        loadObj.files(findex).Frames = [];
        loadObj.files(findex).ext = '';
    end
else
    LoadType = loadObj.Type; % set load type based on loadObj
end

% Update LoadObj
loadObj.Type = LoadType;
loadObj.FrameIndex = [];


%% Load in header information
Config = load2PConfig(ImageFiles);
for findex = 1:numFiles
    loadObj.files(findex).Frames = Config(findex).Frames;
end


%% Load images
switch LoadType
    
%% Load Direct
    case 'Direct'

        % Determine Dimensions are Equal Across Files
        if range([Config(:).Height]) ~= 0 || range([Config(:).Width]) ~= 0 || range([Config(:).Depth]) ~= 0
            error('Data need to be the same size...');
        end
        % Scanbox version 1 fix
        if strcmp(Config(1).type,'sbx') && Config(1).header{1}.scanbox_version == 1
            Config.Width = 796;
        end
        
        % Determine number of frames in each file and which frames to load
        % from each file
        numFrames = [Config(:).Frames];
        FrameIndex = cell(numFiles, 1);
        if iscell(Frames) % specific frames from each file are designated
            if numel(Frames) == 1 && numFiles > 1 % single cell entry specified
                Frames = repmat(Frames, numFiles, 1);
            elseif numel(Frames) ~= numFiles
                error('Must specify which frames to load from each file');
            end
            for findex = 1:numFiles
                FrameIndex{findex} = Frames{findex};
                if FrameIndex{findex}(end) == inf
                    try
                        FrameIndex{findex} = cat(2, FrameIndex{findex}(1:end-1), FrameIndex{findex}(end-1)+1:numFrames(findex));
                    catch % only 'inf' input (numel(Frames)==1)
                        FrameIndex{findex} = 1:numFrames(findex);
                    end
                end
            end
        elseif isnumeric(Frames) % take all files to be one experiment and select relative frames
            cumsumFrames = [0,cumsum(numFrames)];
            totalFrames = cumsumFrames(end);
            cumsumFrames(end) = [];
            if Frames(end) == inf
                try
                    Frames = cat(2, Frames(1:end-1), Frames(end-1)+1:totalFrames);
                catch % only 'inf' input (numel(Frames)==1)
                    Frames = 1:totalFrames;
                end
            end
            temp = bsxfun(@minus, Frames, cumsumFrames');
            for findex = 1:numFiles
                FrameIndex{findex} = temp(findex, temp(findex,:)>=1 & temp(findex,:)<=numFrames(findex));
            end
        end
        for findex = 1:numFiles
            numFrames(findex) = numel(FrameIndex{findex});
        end
        
        % Determine Channels to Load (FILES ALL MUST HAVE DESIRED CHANNEL
        % OR WILL ERROR)
        if ischar(Channels) || (numel(Channels)==1 && Channels == inf)
            Channels = 1:min([Config(:).Channels]); % load all channels (based on file with minimum # of frames)
        elseif Channels(end) == inf
            Channels = [Channels(1:end-2),Channels(end-1):min([Config(:).Channels])];
        end
        
        % Load Images
%         Images = zeros(Config(1).Height, Config(1).Width, Config(1).Depth, numel(Channels), sum(numFrames), 'uint16');
        Images = zeros(Config(1).Height, Config(1).Width, 1, numel(Channels), sum(numFrames), 'uint16');
        startFrame = cumsum([1,numFrames(1:end-1)]);
        for index = 1:numFiles
            [~,~,loadObj.files(index).ext] = fileparts(ImageFiles{index});
            if ~isempty(FrameIndex{index})
                switch loadObj.files(index).ext
                    case '.sbx'
                        Images(:,:,:,:,startFrame(index):startFrame(index)+numFrames(index)-1)...
                            = readSbx(ImageFiles{index}, [], 'Type', 'Direct', 'Frames', FrameIndex{index}, 'Channels', Channels, 'Verbose', Verbose);
                    case '.tif'
                        Images(:,:,:,:,startFrame(index):startFrame(index)+numFrames(index)-1)...
                            = readScim(ImageFiles{index}, 'Frames', FrameIndex{index}, 'Channels', Channels, 'Verbose', Verbose);
                    case '.imgs'
                        Images(:,:,:,:,startFrame(index):startFrame(index)+numFrames(index)-1)...
                            = readImgs(ImageFiles{index}, 'Type', 'Direct', 'Frames', FrameIndex{index}, 'Channels', Channels);
                end
            end
            loadObj.FrameIndex = cat(1, loadObj.FrameIndex, cat(2, repmat(index, numFrames(index), 1), FrameIndex{index}'));
        end
        
        if Double && ~isa(Images, 'double')
            Images = double(Images);
        end

        
%% Load MemMap
    case 'MemMap'
        
        if numFiles > 1
            warning('Cannot load more than one file with MemMap. Loading first file...');
            numFiles = 1;
            Config = Config(1);
            ImageFiles = ImageFiles(1);
            loadObj.files(2:end) = [];
        end
        
        [~,~,loadObj.files.ext] = fileparts(ImageFiles{1});
        switch loadObj.files.ext
            case '.sbx'
                Images = readSbx(ImageFiles{1}, [], 'Type', LoadType);
            case '.tif'
                Images = readScim(ImageFiles, 'Type', LoadType);
            case '.imgs'
                loadObj.files.memmap = readImgs(ImageFiles, 'Type', LoadType);
                Images = loadObj.files.memmap.Data.Frames;
        end
        loadObj.FrameIndex = cat(2, ones(Config.Frames, 1), (1:Config.Frames)');
end


%% Update data information
loadObj.Precision = class(Images(1));
[loadObj.Height, loadObj.Width, loadObj.Depth, loadObj.Channels, loadObj.Frames] = size(Images);
loadObj.DimensionOrder = {'Height', 'Width', 'Depth', 'Channels', 'Frames'};
loadObj.size = [loadObj.Height, loadObj.Width, loadObj.Depth, loadObj.Channels, loadObj.Frames];

loadObj.FrameRate = mode([Config(:).FrameRate]);

%% Save Images
if SaveToMat
    [p,f,~] = fileparts(ImageFiles{index});
    SaveFile = fullfile(p, [f,'.mat']); % automatically create filename to save to
    if exist('SaveFile', 'file') % if file previously exists, prompt for filename
        [SaveFile, p] = uiputfile({'.mat'}, 'Save images as:', p);
        SaveFile = fullfile(p,SaveFile);
    end
    [~,~,ext] = fileparts(SaveFile);
    switch ext
        case '.mat'
            save(SaveFile, 'Images', 'Config', 'numFrames', 'info', '-v7.3');
    end
end