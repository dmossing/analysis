function [Index, ROIMasks, newBoolean] = autoMatchROIs(ROIMasks, Maps, Centroids, varargin)
% ROIMasks - cell array of HxWxN ROI masks or cell array of ROI files or
% cell array of ROIdata structs
% Maps - cell array of imref2d objects or cell array of filenames
% Centroids - cell array of centroids of ROIs or empty

% Index - matrix of size numUniqueROIs x numFiles where each index is the
% index of that unique ROI in that file's ROIMasks matrix, 0 if that ROI is
% not found in that file, or NaN if that ROI shouldn't exist in that FoV.

% ROIMasks - cell array of HxWxR ROI masks or cell array of ROI files

% newROIs - logical array equal in size to index specifying whether the ROI
% of that index is a new ROI (added via this function)

saveOut = false;
saveFile = {''};
saveType = 'new'; % 'new' or 'all' (determines which files to save to)

distanceThreshold = 10; % pixels
overlapThreshold = .7; % percentage
addNewROIs = false; % add ROIs that don't match across files

directory = cd;

%% Parse input arguments
index = 1;
while index<=length(varargin)
    try
        switch varargin{index}
            case 'AddROIs'
                addNewROIs = true;
                index = index + 1;
            case 'ROIindex'
                ROIindex = varargin{index+1};
                index = index + 2;
            case {'Save', 'save'}
                saveOut = true;
                index = index + 1;
            case {'SaveFile', 'saveFile'}
                saveFile = varargin{index+1};
                index = index + 2;
            case {'SaveType', 'saveType'}
                saveType = varargin{index+1};
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

if ~exist('ROIMasks', 'var') || isempty(ROIMasks)
    [ROIMasks,p] = uigetfile({'*.rois;*.mat'}, 'Select ROI files:', directory, 'MultiSelect', 'on');
    if isnumeric(ROIMasks)
        return
    elseif iscellstr(ROIMasks)
        ROIMasks = fullfile(p, ROIMasks);
    else
        ROIMasks = {fullfile(p, ROIMasks)};
    end
end
numFiles = numel(ROIMasks);

if ~exist('Maps', 'var')
    [Maps,p] = uigetfile({'*.exp;*.align'}, 'Select files containing maps:', directory, 'MultiSelect', 'on');
    if isnumeric(Maps)
        return
    elseif iscellstr(Maps)
        Maps = fullfile(p, Maps);
    else
        Maps = {fullfile(p, Maps)};
    end
end


%% Load in ROIMasks
if iscellstr(ROIMasks)
    ROIFiles = ROIMasks;
    InitialROIdata = cell(numFiles, 1);
    ROIMasks = cell(numFiles, 1);
    Centroids = cell(numFiles, 1);
    for findex = 1:numFiles
        [~,~,ext] = fileparts(ROIFiles{findex});
        switch ext
            case '.segment'
                load(ROIFiles{findex}, 'mask', 'dim', '-mat');
                if issparse(mask)
                    ROIMasks{findex} = reshape(full(mask), dim(1), dim(2), size(mask,2));
                else
                    ROIMasks{findex} = mask;
                end
                Centroids{findex} = zeros(size(ROIMasks{findex},3), 2);
                for rindex = 1:size(ROIMasks{findex},3)
                    temp = regionprops(ROIMasks{findex}(:,:,rindex), 'centroid');
                    Centroids{findex}(rindex,:) = temp.Centroid;
                end
            case '.rois'
                load(ROIFiles{findex}, 'ROIdata', '-mat');
                InitialROIdata{findex} = ROIdata;
                ROIMasks{findex} = reshape(full([ROIdata.rois(:).pixels]), size(ROIdata.rois(1).pixels,1), size(ROIdata.rois(1).pixels,2), numel(ROIdata.rois));
                Centroids{findex} = reshape([ROIdata.rois(:).centroid], 2, numel(ROIdata.rois))';
        end
    end
elseif isstruct(ROIMasks{1}) % ROIs cell input
    ROIFiles = strcat('file', {' '}, num2str((1:numFiles)'));
    InitialROIdata = cell(numFiles, 1);
    ROIMasks = cell(numFiles, 1);
    Centroids = cell(numFiles, 1);
    for findex = 1:numFiles
        InitialROIdata{findex} = ROIMasks{findex};
        ROIMasks{findex} = reshape(full([InitialROIdata{findex}.rois(:).pixels]), size(InitialROIdata{findex}.rois(1).pixels,1), size(InitialROIdata{findex}.rois(1).pixels,2), numel(InitialROIdata{findex}.rois));
        Centroids{findex} = reshape([InitialROIdata{findex}.rois(:).centroid], 2, numel(InitialROIdata{findex}.rois))';
    end
else
    ROIFiles = strcat('file', {' '}, num2str((1:numFiles)'));
end
[Height,Width,numROIs] = cellfun(@size, ROIMasks);

% Compute centroids (if not given as input argument or loaded)
if ~exist('Centroids', 'var') || isempty(Centroids)
    Centroids = cell(numFiles, 1);
    for findex = 1:numFiles
        Centroids{findex} = zeros(numROIs(findex), 2);
        for rindex = 1:numROIs(findex)
            temp = regionprops(ROIMasks{findex}(:,:,rindex), 'centroid');
            Centroids{findex}(rindex,:) = temp.Centroid;
        end
    end
end

fprintf('Matching ROIs between %d files\n', numFiles);
temp = strcat(num2str(numROIs), {' rois from '}, ROIFiles');
fprintf('\t%s\n', temp{:});


%% Load in Maps
if iscellstr(Maps)
    MapFiles = Maps;
    Maps = imref2d();
    for findex = 1:numFiles
        temp = load(MapFiles{findex}, 'Map', '-mat');
        if isfield(temp, 'Map')
            Maps(findex) = temp.Map;
        else
            warning('No map found in file %s, assuming file starts at origin', MapFiles{findex});
            Maps(findex) = imref2d([Height(findex), Width(findex)]);
        end
    end
elseif isempty(Maps)
    warning('No maps input, assuming all files start at origin');
    Maps = cell(numFiles, 1);
    for findex = 1:numFiles
        Maps(findex) = imref2d([Height(findex), Width(findex)]);
    end
end


%% Build Map
[offsets, refMap, indMap] = mapFoVs(Maps);
[H,W,~] = size(indMap);


%% Translate ROIs
ActualMasks = cell(numFiles,1);
ROIindex = cell(numFiles,1);
for findex = 1:numFiles
    ROIindex{findex} = 1:numROIs(findex);
    Centroids{findex} = bsxfun(@plus, Centroids{findex}, offsets(findex,1:2));
    ActualMasks{findex} = false(H, W, numROIs(findex));
    for rindex = 1:numROIs(findex)
        ActualMasks{findex}(:,:,rindex) = imwarp(ROIMasks{findex}(:,:,rindex), Maps(findex), affine2d(), 'OutputView', refMap);
    end
end

% Reshape variables
indMap = reshape(indMap, H*W, numFiles);
ActualMasks = cellfun(@(x) reshape(x, size(x,1)*size(x,2), size(x,3)), ActualMasks, 'UniformOutput', false);


%% Determine distances between ROI centroids in the different datasets and the amount of overlap between the datasets
combinations = combnk(1:numFiles, 2);
[~,I] = sort(combinations, 1);
combinations = combinations(I(:,1), :); % sort order
numCombinations = size(combinations, 1);
distances = cell(numCombinations, 1);
overlapMasks = zeros(H*W, numCombinations);
for cindex = 1:numCombinations
    distances{cindex} = pdist2(Centroids{combinations(cindex, 1)}, Centroids{combinations(cindex, 2)}); % distance between all ROIs in current 2 files
    overlapMasks(:, cindex) = all(indMap(:,[combinations(cindex, 1), combinations(cindex, 2)]),2);             % region of overlap between the current 2 files
end


%% Match ROIs
Index = [];
numUniqueROIs = 0;
for findex = 1:numFiles
    numCurrentROIs = numel(ROIindex{findex});               % unmatched ROIs left in current dataset
    Index = cat(1, Index, nan(numCurrentROIs, numFiles));   % expand list of unique ROIs
    Index(numUniqueROIs+1:end, findex) = ROIindex{findex};  % record ROI identifiers for current file
    
    % Cycle through remaining ROIs matching each to the other files
    for rindex = ROIindex{findex}
        
        % Determine what files current ROI should be found in
        X = min(max(round(Centroids{findex}(rindex,1)), 1), W);
        Y = min(max(round(Centroids{findex}(rindex,2)), 1), H);
        FileIndices = find(indMap(sub2ind([H,W], Y, X), :)); % centroid in FoV
        % FileIndices = find(all(Map(ActualMasks{findex}(:,rindex),:), 1)); % ROI mask completely in FoV
        FileIndices(FileIndices==findex) = [];
        
        % Match ROIs in each of the matched Files
        for mfindex = FileIndices
            
            if ~isempty(ROIindex{mfindex}) % Determine if matched file has ROIs available
                
                % Sort ROIs in matched file by distance from current ROI
                cindex = ismember(combinations,[findex, mfindex],'rows');
                if any(cindex)
                    currentDistances = distances{cindex}(rindex, :);
                else
                    cindex = ismember(combinations,[mfindex, findex],'rows');
                    currentDistances = distances{cindex}(:, rindex);
                end
                [~,distIndices] = sort(currentDistances(ROIindex{mfindex}));
                
                % Determine if any of the ROIs within the distance threshold
                % overlap more than the overlap threshold
                for mrindex = ROIindex{mfindex}(distIndices)
                    distThresh = currentDistances(mrindex) <= distanceThreshold;
                    overlapThresh = nnz(all([ActualMasks{findex}(:,rindex),ActualMasks{mfindex}(:,mrindex),overlapMasks(:,cindex)],2))/nnz(all([ActualMasks{findex}(:,rindex),overlapMasks(:,cindex)],2)) >= overlapThreshold;
                    if ~distThresh                                          % moved onto ROIs too far away
                        Index(numUniqueROIs+1, mfindex) = 0;                % therefore ROI doesn't exist in matched file
                        break
                    elseif overlapThresh                                    % ROI matches
                        Index(numUniqueROIs+1, mfindex) = mrindex;          % record matching ROI
                        ROIindex{mfindex}(ROIindex{mfindex}==mrindex) = []; % remove matching ROI from list
                        break
                    end
                end %mrindex
                
            else
                Index(numUniqueROIs+1, mfindex) = 0;                        % no more available ROIs in current file -> ROI doesn't exist
            end
            
        end %mfindex
        
        ROIindex{findex}(ROIindex{findex}==rindex) = [];    % remove current ROI from list so it can't be matched later
        numUniqueROIs = numUniqueROIs + 1;                  % update ROI counter
        
    end %rindex
        
end % findex


%% Add unmatched ROIs
newBoolean = false(size(Index));
if addNewROIs
    fprintf('Adding new ROIs across files...\n');
    
    % Determine what ROIs are missing
    needsROI = Index == 0;
    numNew = sum(needsROI);
    
    % Cycle through files adding new ROIs to each file
    for findex = find(numNew)
        
        % Determine indices of missing ROIs
        uindex = find(needsROI(:,findex));
        fprintf('\tadding %d ROI(s) to %s', numNew(findex), ROIFiles{findex});
        
        % Cycle through new ROIs adding each to the output
        ROIMasks{findex} = cat(3, ROIMasks{findex}, zeros(Height(findex), Width(findex), numNew(findex)));
        for rindex = 1:numNew(findex)
            [~, mfindex, mrindex] = find(Index(uindex(rindex),:), 1);                                           % find matched ROI in another file
            ROIMasks{findex}(:,:,numROIs(findex)+rindex) = transferROIs(ROIMasks{mfindex}(:,:,mrindex), Maps(mfindex), Maps(findex));   % translate and add ROI                            % add new ROI
            Index(uindex(rindex), findex) = numROIs(findex)+rindex;                                             % record index of new ROI
            newBoolean(uindex(rindex), findex) = true;
        end
        
    end
    fprintf('\nComplete\n');
end


%% Save output
if saveOut && ~isempty(saveFile)
    
    % Determine what files to save to
    switch saveType
        case 'all'
            findices = 1:numFiles;
        case 'new'
            findices = find(numNew);
    end
    
    % Save to each file
    for findex = findices
        [~,~,ext] = fileparts(saveFile{findex});
        switch ext
            
            case '.segment'
                
                % Save variable to file
                mask = sparse(reshape(ROIMasks{findex}, Height(findex)*Width(findex), size(ROIMasks{findex},3)));
                if ~exist(saveFile{findex}, 'file')
                    save(saveFile{findex}, 'mask', '-mat', '-v7.3');
                else
                    save(saveFile{findex}, 'mask', '-mat', '-append');
                end
                
            case '.rois'
                
                % Determine if previous variable exists
                if isempty(InitialROIdata{findex}) && exist(saveFile{findex}, 'file')
                    load(saveFile{findex}, 'ROIdata', '-mat');
                    if exist('ROIdata', 'var')
                        InitialROIdata{findex} = ROIdata;
                    end
                end
                
                % Create/update struct
                if isempty(InitialROIdata{findex})
                    ROIdata = createROIdata(ROIMasks{findex}); % update whole struct
                else
                    ROIdata = createROIdata(ROIMasks{findex}(:,:,numROIs(findex)+1:end), 'ROIdata', InitialROIdata{findex}); % only update new ROIs
                end
                
                % Save to file
                if ~exist(saveFile{findex}, 'file')
                    save(saveFile{findex}, 'ROIdata', '-mat', '-v7.3');
                else
                    save(saveFile{findex}, 'ROIdata', '-mat', '-append');
                end
                
        end %switch ext
        fprintf('(%d of %d) Saved rois to file: %s\n', findex, numel(findices), saveFile{findex});
    end %findex
end