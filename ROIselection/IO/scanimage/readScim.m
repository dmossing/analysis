function [Images, Config] = readScim(TifFile, varargin)
% Loads 'Frames' of single .sbx file ('SbxFile'). Requires
% corresponding information file ('InfoFile').

LoadType = 'Direct'; % 'MemMap' or 'Direct'
Frames = 1:20; % indices of frames to load in 'Direct' mode, or 'all'
Channels = 1;
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
            case {'Channels','channels'}
                Channels = varargin{index+1};
                index = index + 2;
            case {'Type','type'}
                LoadType = varargin{index+1}; %'Direct' or 'MemMap'
                index = index + 2;
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
    [TifFile,p] = uigetfile({'*.sbx'}, 'Choose scanbox file to load');
    if isnumeric(TifFile)
        Images = []; return
    end
    TifFile = fullfile(p,TifFile);
end

%% Load In Acquisition Information
Config = parseScimHeader(TifFile);

%% Load In Images

switch LoadType
    case 'MemMap'
        
        warning('MemMap TIFF won''t work for functions looking for a ''Height'',''Width'',''Depth'',''Channels'',''Frames'' ordering');
        Images = TIFFStack(TifFile);
        
    case 'Direct'
        
        % Determine frames to load
        if ischar(Frames) || (numel(Frames)==1 && Frames == inf)
            Frames = 1:Config.Frames;
        elseif Frames(end) == inf
            Frames = [Frames(1:end-2),Frames(end-1):info.numFrames];
        end
        
        if Verbose
            fprintf('Loading\t%d\tframe(s) from\t%s...', numel(Frames), TifFile);
        end
        
        [~,Images] = scim_openTif(TifFile, 'frames', Frames, 'channels', Channels);
        
        if Config.Depth == 1
            Images = permute(Images, [1,2,5,3,4]);
        else
            Images = permute(Images, [1,2,4,3,5]);
        end
        
        % tif=Tiff(ImgFiles{f},'r');
        % go=0;
        % while tif.currentDirectory~=Channel
        %     tif.nextDirectory
        % end
        % while ~go
        %     Images(:,:,i)=tif.read();
        %     metadata{i} = ImgFiles{f};
        %     i=i+1;
        %     if tif.lastDirectory %frame just read was last frame
        %         go=1;
        %     end
        %     if ~go
        %         tif.nextDirectory
        %         if tif.lastDirectory && rem(tif.currentDirectory-Channel,nChannels)~=0 %last frame, and frame isn't a part of the current channel
        %             go=1;
        %         end
        %         while ~go && rem(tif.currentDirectory-Channel,nChannels)~=0 %while not a frame in current channel and haven't hit last frame, go to next frame
        %             tif.nextDirectory
        %             if tif.lastDirectory && rem(tif.currentDirectory-Channel,nChannels)~=0 %last frame, and frame isn't a part of the current channel
        %                 go=1;
        %             end
        %         end
        %     end
        % end
        % tif.close;
        % waitbar(f/nFiles,wb);
        % end
        %
        % Images=double(Images);
        % close(wb);
        
        if Verbose
            fprintf('\tComplete\n');
        end
        
end

