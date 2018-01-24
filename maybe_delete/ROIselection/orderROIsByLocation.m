function [Order, Distances, ROIindex, ROIs] = orderROIsByLocation(ROIs, ExperimentFiles, ROIindex, FileIndex, Label)



%% Parse input arguments
if ~exist('ROIindex', 'var') || isempty(ROIindex)
    ROIindex = 'all';
end

if ~exist('FileIndex', 'var')
    FileIndex = [];
end

if ~exist('Label', 'var')
    Label = [];
end

%% Load in data
[Centroid, FileIndex, ROIindex, ~, ~, ROIs] = gatherROIdata(ROIs, 'centroid', ':', Label, ROIindex, FileIndex);
numFiles = numel(ROIs);
numROIs = numel(ROIindex);


%% Shift ROIs
if numel(ROIs)>1

end


%% Find distances between ROIs and given location
Distances = nan(numROIs,1);
switch type
    case 'frame'
        for findex = 1:numFiles
            index = find(FileIndex==findex);
            load(ExperimentFiles{findex},'ImageFiles');
            hF = figure;
            imagesc(ImageFiles.Average);
            [X,Y] = ginput(1);
            close(hF);
            Distances(index) = pdist2([X,Y], Centroid(index,:));
        end
        
    case 'mFoV'
        [Image, ~, ~] = createMultiFoVImage(ExperimentFiles, 'average', 'quick');
        hF = figure;
        imagesc(Image);
        [X,Y] = ginput(1);
        close(hF);
        for findex = 1:numFiles
            index = find(FileIndex==findex);
            Centroid(index,:) = bsxfun(@plus, Centroid(index, :), ROIs{findex}.mFoVShift);
            Distances(index) = pdist2([X,Y], Centroid(index,:));
        end
        
end

%% Sort ROIs
[Distances, Order] = sort(Distances);
