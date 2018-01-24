function [Images, Maps, rect] = crop(Images, rect, Maps)
% rect - [x, y, width, height] (bottom left corner->x,y)

%% Parse input arguments
if iscell(Images)
    output = 'cell';
elseif ~isempty(Images)
    Images = {Images};
    output = 'numeric';
else
    output = 'empty';
end

if ~exist('rect', 'var') || isempty(rect)
    rect = true;
%     rect = repmat([32.51, 0, 729.98, 512], numFiles, 1);
end
if isequal(rect,true);
    UI = true;
else
    UI = false;
end

if ~exist('Maps', 'var') || isempty(Maps)
    Maps = [];
end


%% Crop images
if ~isempty(Images) 
    numFiles = numel(Images);
    if UI
        rect = nan(numFiles,4);
    end
    
    % Fix input
    if ~islogical(rect) && size(rect, 1) == 1 && numFiles>1
        rect = repmat(rect, numFiles, 1);
    end
    
    % Crop images
    for findex = 1:numFiles
        
        % UI-crop
        if UI
            [~, rect(findex,:)] = imcrop(Images{findex}(:,:,1,1,1));
        end
        
        % Make rect integers
        rect(findex,[1,2]) = floor(rect(findex,[1,2]));
        rect(findex,[3,4]) = ceil(rect(findex,[3,4]));
        
        % Crop image
        if ~iscell(Images{findex})
            Images{findex} = Images{findex}(rect(findex,2)+1:rect(findex,2)+rect(findex,4), rect(findex,1)+1:rect(findex,1)+rect(findex,3),:,:,:);
        else
            for cindex = 1:numel(Images{findex})
                Images{findex}{cindex} = Images{findex}{cindex}(rect(findex,2)+1:rect(findex,2)+rect(findex,4), rect(findex,1)+1:rect(findex,1)+rect(findex,3),:,:,:);
            end
        end
        
    end
    
end

% Close figure (UI-crop only)
if UI
    close gcf;
end

% Fix output
if strcmp(output, 'numeric')
    Images = Images{1};
end


%% Crop maps
if ~isempty(Maps)
    numFiles = numel(Maps);
    
    % Fix input
    if ~islogical(rect) && size(rect, 1) == 1 && numFiles>1
        rect = repmat(rect, numFiles, 1);
    end
    
    for findex = 1:numFiles
        
        % Make rect integers
        rect(findex,[1,2]) = floor(rect(findex,[1,2]));
        rect(findex,[3,4]) = ceil(rect(findex,[3,4]));
        
        % Crop map
        Maps(findex).XWorldLimits(1) = Maps(findex).XWorldLimits(1)+rect(findex,1);
        Maps(findex).XWorldLimits(2) = Maps(findex).XWorldLimits(1)+rect(findex,3);
        Maps(findex).YWorldLimits(1) = Maps(findex).YWorldLimits(1)+rect(findex,2);
        Maps(findex).YWorldLimits(2) = Maps(findex).YWorldLimits(1)+rect(findex,4);
        Maps(findex).ImageSize = rect(findex,[4,3]);
    end
end



