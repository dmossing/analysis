function Images = applyMotionCorrection(Images, MCdata, MCindex, frameIndex, depthIndex)


directory = cd;

%% Parse input arguments
if ~exist('Images','var') || isempty(Images) % Prompt for file selection
    [ImageFile,p] = uigetfile({'*.imgs;*.sbx;*.tif'},'Select images:',directory);
    if isnumeric(ImageFile)
        return
    end
    ImageFile = fullfile(p, ImageFile);
    [Images, loadObj] = load2P(ImageFile, 'Type', 'Direct', 'Double');
elseif ischar(Images)
    [Images, loadObj] = load2P(Images, 'Type', 'Direct', 'Double');
end
[~,~, numDepth, numChannels, numFrames] = size(Images);

if ~exist('MCdata','var') || isempty(MCdata)
    [ExperimentFile, p] = uigetfile({'*.mat'},'Choose Experiment file',directory);
    if isnumeric(ExperimentFile)
        return
    end
    ExperimentFile = fullfile(p,ExperimentFile);
    load(ExperimentFile, 'MCdata');
elseif ischar(MCdata)
    load(MCdata, 'MCdata');
end

if ~exist('MCindex', 'var') || isempty(MCindex)
    MCindex = cat(2, ones(numFrames, 1), (1:numFrames)');
elseif isstruct(MCindex)
    loadObj = MCindex;
    if exist('frameIndex', 'var')
        loadObj.FrameIndex = loadObj.FrameIndex(frameIndex, :);
    end
elseif isvector(MCindex)
    if isrow(MCindex)
        MCindex = MCindex';
    end
    MCindex = cat(2, ones(numFrames, 1), MCindex);
end

%% Determine frame indices
if exist('loadObj', 'var')
    % Determine ImageFiles in loadObj
    ImageFiles = {loadObj.files(:).FullFilename}; % gather filenames
    numFiles = numel(ImageFiles);
    
    % Match MCdata to ImageFiles
    MCdataIndex = nan(numFiles, 1);
    MCFiles = {MCdata(:).FullFilename};
    for findex = 1:numFiles
        try
            MCdataIndex(findex) = find(strcmp(ImageFiles{findex}, MCFiles));
        catch
            warning('Image file not found in motion correction object. Assuming motion correction corresponds to first file input');
            MCdataIndex(findex) = 1;
        end
    end
    
    % Create frame indices vector
    MCindex = loadObj.FrameIndex;
    for findex = 1:numFiles
        MCindex(MCindex(:,1)==findex,1) = MCdataIndex(findex);
    end
    if any(isnan(MCindex(:,1)))
        error('Filenames contained in MCdata do not match image files loaded -> most likely because ImageFile was moved and MCdata was not updated');
    end
end

%% Apply motion correction
switch MCdata(1).type
    
    case 'doLucasKanade'
        for iF = 1:numFrames
            for iC = 1:numChannels
                for iD = 1:numDepth
                    
                    % to specify depth of image passed in
                    if ~exist('depthIndex', 'var')
                        Di = iD;
                    else
                        Di = depthIndex;
                    end
                    
                    if iF == 1 % initialize variables
                        [Images(:,:,iD,iC,iF), B, xi, yi] = applyDoLucasKanade(...
                            Images(:,:,iD,iC,iF),...
                            MCdata(MCindex(iF,1)).dpx(:,MCindex(iF,2),Di),...
                            MCdata(MCindex(iF,1)).dpy(:,MCindex(iF,2),Di));
                    else % use previously created variables
                        Images(:,:,iD,iC,iF) = applyDoLucasKanade(...
                            Images(:,:,iD,iC,iF),...
                            MCdata(MCindex(iF,1)).dpx(:,MCindex(iF,2),Di),...
                            MCdata(MCindex(iF,1)).dpy(:,MCindex(iF,2),Di), B, xi, yi);
                    end
                end
            end
        end
        
    case 'Translation'
        for iF = 1:numFrames
            for iC = 1:numChannels
                for iD = 1:numDepth
                    
                    % to specify depth of image passed in
                    if ~exist('depthIndex', 'var')
                        Di = iD;
                    else
                        Di = depthIndex;
                    end
                    
                    try
                        Images(:,:,iD,iC,iF) = circshift(Images(:,:,iD,iC,iF), MCdata(MCindex(iF,1)).T(MCindex(iF,2),:));
                    catch ME
                        if MCindex(iF,2) ~= size(MCdata(MCindex(iF,1)).T,1)+1 % old bug where code doesn't motion correct last image
                            rethrow(ME)
                        end
                    end        
                    
                end
            end
        end
end