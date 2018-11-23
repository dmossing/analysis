function Images = readTiff(TifFile, varargin)
% Loads 'Frames' of single .sbx file ('SbxFile'). Requires
% corresponding information file ('InfoFile').

Frames = [1,inf]; % indices of frames to load in 'Direct' mode, or 'all'
% Channel = 1;
% Depths = 1;
Verbose = true;

warning('off','MATLAB:imagesci:tiffmexutils:libtiffWarning');

%% Initialize Parameters
index = 1;
while index<=length(varargin)
    try
        switch varargin{index}
            case {'Frames','frames'} % indices of frames to load in 'Direct' mode
                Frames = varargin{index+1};
                index = index + 2;
%             case {'Depths', 'depths'}
%                 Depths = varargin{index+1};
%                 index = index + 2;
%             case {'Channel','channel'}
%                 Channel = varargin{index+1};
%                 index = index + 2;
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

if ~exist('TifFile', 'var') || isempty(TifFile)
    [TifFile,p] = uigetfile({'*.tif'}, 'Choose scanbox file to load');
    if isnumeric(TifFile)
        Images = []; return
    end
    TifFile = fullfile(p,TifFile);
end


%% Load in header
warning('off', 'MATLAB:imagesci:tifftagsread:nextIfdPointerOutOfRange');
info = imfinfo(TifFile,'tif');
totalFrames = numel(info);


%% Determine frames to load
if ischar(Frames) || (numel(Frames)==1 && Frames == inf)
    Frames = 1:totalFrames;
elseif Frames(end) == inf
    Frames = [Frames(1:end-2),Frames(end-1):totalFrames];
end
numFrames = numel(Frames);


%% Load in frames
if Verbose
    fprintf('Loading\t%d\tframe(s) from\t%s...', numFrames, TifFile);
    parfor_progress(numFrames);
end

switch info(1).ColorType
    case 'truecolor'
        Images = zeros(info(1).Height, info(1).Width, 3, numFrames);
    otherwise
        Images = zeros(info(1).Height, info(1).Width, 1, numFrames);
end
tif=Tiff(TifFile,'r');
findex = 1;
while findex < numFrames
    if tif.currentDirectory ~= Frames(findex)
        tif.nextDirectory
    else
        Images(:,:,:,findex)=tif.read();
        findex=findex+1;
        if Verbose
            parfor_progress;
        end
    end
end
tif.close;

if Verbose
    parfor_progress(0);
end

