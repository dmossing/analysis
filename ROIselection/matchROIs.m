function matchROIs(ExperimentFiles1, ExperimentFiles2, ROIFiles1, ROIFiles2)


%% Check input arguments


%% Configure default settings
gd.Internal.directory = '\\128.32.173.33\Imaging\ResonantScope';
gd.Internal.Settings.ROITextSize = 20;
gd.Internal.Settings.ROILineWidth = 2;
gd.Internal.ID = [];

%% Create & Populate Figure

% FIGURE
gd.fig = figure(...
    'NumberTitle',          'off',...
    'Name',                 'Match ROIs',...
    'ToolBar',              'none',...
    'Units',                'pixels',...
    'Position',             [50, 50, 1400, 800],...
    'KeyPressFcn',          @(hObject,eventdata)KeyPressCallback(hObject,eventdata,guidata(hObject)));

% DATA1
% panel
gd.Data1.panel = uipanel(...
    'Title',                'Display',...
    'Parent',               gd.fig,...
    'Units',                'Normalized',...
    'Position',             [0, .5, .5, .5]);
% experiment load button
gd.Data1.loadExp = uicontrol(...
    'Style',                'pushbutton',...
    'String',               'Load Experiment',...
    'Parent',               gd.Data1.panel,...
    'Units',                'normalized',...
    'Position',             [0,.9,.2,.1],...
    'UserData',             1,...
    'Callback',             @(hObject,eventdata)LoadExperiment(hObject,eventdata,guidata(hObject)),...
    'BackgroundColor',      [.3,.3,.3],...
    'ForegroundColor',      [1,1,1]);
% ROIs load button
gd.Data1.loadROIs = uicontrol(...
    'Style',                'pushbutton',...
    'String',               'Load ROIs',...
    'Parent',               gd.Data1.panel,...
    'Units',                'normalized',...
    'Position',             [.2,.9,.2,.1],...
    'UserData',             1,...
    'Callback',             @(hObject,eventdata)LoadROIs(hObject,eventdata,guidata(hObject)),...
    'BackgroundColor',      [.3,.3,.3],...
    'ForegroundColor',      [1,1,1]);
% axes
gd.Data1.axes = axes(...
    'Parent',               gd.Data1.panel,...
    'Units',                'normalized',...
    'Position',             [0,0,.85,.9]);
axis off
% ROIs listbox
gd.Data1.list = uicontrol(...
    'Style',                'listbox',...
    'String',               [],...
    'Parent',               gd.Data1.panel,...
    'Units',                'normalized',...
    'Position',             [.85,0,.15,.9],...
    'UserData',             1,...
    'Value',                [],...
    'Callback',             @(hObject,eventdata)plotDataAxes(guidata(hObject), hObject));
% copy ROI button
gd.Data1.copy = uicontrol(...
    'Style',                'pushbutton',...
    'String',               'Add ->',...
    'Parent',               gd.Data1.panel,...
    'Units',                'normalized',...
    'Position',             [.85,.9,.15,.1],...
    'UserData',             1,...
    'Callback',             @(hObject,eventdata)AddROI(hObject,eventdata,guidata(hObject)));

% DATA2
% panel
gd.Data2.panel = uipanel(...
    'Title',                'Display',...
    'Parent',               gd.fig,...
    'Units',                'Normalized',...
    'Position',             [.5, .5, .5, .5]);
% experiment load button
gd.Data2.loadExp = uicontrol(...
    'Style',                'pushbutton',...
    'String',               'Load Experiment',...
    'Parent',               gd.Data2.panel,...
    'Units',                'normalized',...
    'Position',             [.6,.9,.2,.1],...
    'UserData',             2,...
    'Callback',             @(hObject,eventdata)LoadExperiment(hObject,eventdata,guidata(hObject)),...
    'BackgroundColor',      [.3,.3,.3],...
    'ForegroundColor',      [1,1,1]);
% ROIs load button
gd.Data2.loadROIs = uicontrol(...
    'Style',                'pushbutton',...
    'String',               'Load ROIs',...
    'Parent',               gd.Data2.panel,...
    'Units',                'normalized',...
    'Position',             [.8,.9,.2,.1],...
    'UserData',             2,...
    'Callback',             @(hObject,eventdata)LoadROIs(hObject,eventdata,guidata(hObject)),...
    'BackgroundColor',      [.3,.3,.3],...
    'ForegroundColor',      [1,1,1]);
% axes
gd.Data2.axes = axes(...
    'Parent',               gd.Data2.panel,...
    'Units',                'normalized',...
    'Position',             [.15,0,.85,.9]);
axis off
% ROIs listbox
gd.Data2.list = uicontrol(...
    'Style',                'listbox',...
    'String',               [],...
    'Parent',               gd.Data2.panel,...
    'Units',                'normalized',...
    'Position',             [0,0,.15,.9],...
    'UserData',             2,...
    'Value',                [],...
    'Callback',             @(hObject,eventdata)plotDataAxes(guidata(hObject), hObject));
% copy ROI button
gd.Data2.copy = uicontrol(...
    'Style',                'pushbutton',...
    'String',               '<- Add',...
    'Parent',               gd.Data2.panel,...
    'Units',                'normalized',...
    'Position',             [0,.9,.15,.1],...
    'UserData',             2,...
    'Callback',             @(hObject,eventdata)AddROI(hObject,eventdata,guidata(hObject)));

% OFFSET IMAGES: overlay images
% panel
gd.Merge.panel = uipanel(...
    'Title',                'Merge',...
    'Parent',               gd.fig,...
    'Units',                'Normalized',...
    'Position',             [0, 0, .5, .5]);
% axes
gd.Merge.axes = axes(...
    'Parent',               gd.Merge.panel,...
    'Units',                'normalized',...
    'Position',             [0,0,1,1]);
axis off
% popupmenu: merge type
% Data table (columns for name, view, move)

% MATCH ROIs
% panel
gd.Match.panel = uipanel(...
    'Title',                'Match',...
    'Parent',               gd.fig,...
    'Units',                'Normalized',...
    'Position',             [.5, 0, .5, .5]);
% axes
gd.Match.axes = axes(...
    'Parent',               gd.Match.panel,...
    'Units',                'normalized',...
    'Position',             [0,0,1,.9]);
axis off
% match button
gd.Match.match = uicontrol(...
    'Style',                'pushbutton',...
    'String',               'Match Current ROIs',...
    'Parent',               gd.Match.panel,...
    'Units',                'normalized',...
    'Position',             [0,.9,.2,.1],...
    'Callback',             @(hObject,eventdata)MatchROIs(hObject,eventdata,guidata(hObject)),...
    'BackgroundColor',      [.7,1,.7],...
    'Enable',               'off');
% mouse ID
gd.Match.ID = uicontrol(...
    'Style',                'edit',...
    'String',               '0000',...
    'Parent',               gd.Match.panel,...
    'Units',                'normalized',...
    'Position',             [.2,.9,.2,.05]);
% base string
gd.Match.base = uicontrol(...
    'Style',                'edit',...
    'String',               strcat('_',datestr(now,'yyyy-mm-dd'),'_'),...
    'Parent',               gd.Match.panel,...
    'Units',                'normalized',...
    'Position',             [.4,.9,.2,.05]);
% index
gd.Match.index = uicontrol(...
    'Style',                'edit',...
    'String',               '1',...
    'Parent',               gd.Match.panel,...
    'Units',                'normalized',...
    'Position',             [.6,.9,.2,.05]);
% pair-wise or all dropdown
gd.Match.type = uicontrol(...
    'Style',                'popupmenu',...
    'String',               {'pair-wise','all'},...
    'Parent',               gd.Match.panel,...
    'Units',                'normalized',...
    'Position',             [.2,.95,.3,.05]);
% right tag, left tag, or new tag dropdown
% gd.Match.tag = uicontrol(...
%     'Style',                'popupmenu',...
%     'String',               {'left','right','new'},...
%     'Parent',               gd.Match.panel,...
%     'Units',                'normalized',...
%     'Position',             [.2,.95,.3,.05]);
% match button
gd.Match.save = uicontrol(...
    'Style',                'pushbutton',...
    'String',               'Save Tags',...
    'Parent',               gd.Match.panel,...
    'Units',                'normalized',...
    'Position',             [.8,.9,.2,.1],...
    'Callback',             @(hObject,eventdata)SaveROIs(hObject,eventdata,guidata(hObject)),...
    'BackgroundColor',      [.7,1,.7],...
    'Enable',               'off');
guidata(gd.fig, gd);
% 
% subplot 4: ROIs overlayed on top of each other 
%   first file red
%   second file blue, overlayed => purple
%   shows only ROIs selected
%   boundaries are shared region
% 2 listboxes: list of ROIs from the two files
%   2 select all ROIs shown above
% 2 listboxes: list of currently selected ROIs from the two files
%   1 match button, pair up the ROIs selected to the most overlapping ROI
%   2 "Add ROIs ->" buttons
% automatch button to automatically pair selected ROIs
%   requires overlapping threshold input
%   ability to accept or don't accept
% toggle autoselect paired ROI

%% CLICK CALLBACKS
function ROIClickCallback(hObject,eventdata,gd)
% if eventdata.Button == 1
%     switch get(hObject, 'Type')
%         case 'patch'
%             index = get(hObject,'UserData');
%             gd.Internal.ROIs.current = index;
%         case 'image'
%             gd.Internal.ROIs.current = false;
%             set([gd.ROI.currentLabels,gd.ROI.info], 'String', '');
%             set(gd.ROI.currentTag, 'String', 'tag');
%     end
%     guidata(hObject, gd);
%     plotmainaxes([],[],gd);
% end

%% MAIN
function LoadExperiment(hObject,eventdata,gd)

% Select Files
if ~iscell(eventdata)
    if isdir(gd.Internal.directory)
        directory = gd.Internal.directory;
    elseif exist('CanalSettings.m', 'file');
        directory = CanalSettings('ExperimentDirectory');
    else
        directory = cd;
    end
    [ExperimentFiles, p] = uigetfile({'*.mat;*.ciexp'}, 'Choose experiment file(s)', directory, 'MultiSelect', 'on');
    if isnumeric(ExperimentFiles)
        return
    elseif iscell(ExperimentFiles)
        for findex = 1:numel(ExperimentFiles)
            ExperimentFiles{findex} = fullfile(p, ExperimentFiles{findex});
        end
    else % ischar(ExperimentFiles)
        ExperimentFiles = {fullfile(p, ExperimentFiles)};
    end
%     ExperimentFiles = uipickfiles('Prompt', 'Choose experiment files', 'FilterSpec', [directory, '*.ciexp']);
%     if isnumeric(ExperimentFiles)
%         return
%     end
else
    ExperimentFiles = eventdata;
end
[gd.Internal.directory,~,~] = fileparts(ExperimentFiles{1});
numFiles = numel(ExperimentFiles);
index = get(hObject, 'UserData');

% Determine place in struct
if isfield(gd, 'Experiment') % previous files loaded
    offset = numel(gd.Experiment); % add files to end
else
    offset = 0;
end

% Load in files
for findex = 1:numFiles
    load(ExperimentFiles{findex}, 'Map', 'ImageFiles', 'ID', '-mat');
    
    % Determine if same mouse as the files previously loaded
    if isempty(gd.Internal.ID)
        gd.Internal.ID = ID;
        set(gd.Match.ID, 'String', ID);
    elseif ~strcmp(gd.Internal.ID, ID)
        error('Mouse being loaded (%s) is not same mouse as the one loaded (%s)', gd.Internal.ID, ID);
    end
    
    % Place file in struct
    gd.Experiment(offset + findex).filename = ExperimentFiles{findex};
    gd.Experiment(offset + findex).index = index;
    if exist('ImageFiles', 'var')
        gd.Experiment(offset + findex).ImageFiles = ImageFiles;
    else
        error('Compute average of data first using ''ComputeProjections''');
    end
    if exist('Map', 'var')
        gd.Experiment(offset + findex).Map = Map;
    else
        gd.Experiment(offset + findex).Map = [];
    end
end

% Build Individual Images
for findex = offset+1:offset+numFiles
    gd.Experiment(findex).Image = permute(gd.Experiment(findex).ImageFiles.Average, [1,2,4,3]);
    [H, W, C, ~] = size(gd.Experiment(findex).Image);
    if C == 1
        gd.Experiment(findex).Image = cat(3, zeros(H, W), gd.Experiment(findex).Image, zeros(H, W)); % green only
    elseif C == 2
        gd.Experiment(findex).Image = cat(3, gd.Experiment(findex).Image(:,:,[2,1]), zeros(H, W)); % green & red data
    end
    if isempty(gd.Experiment(findex).Map)
        gd.Experiment(findex).Map = imref2d([H,W]);
    end
    % Determine colormaps
    for cindex = 2
        temp = gd.Experiment(findex).Image(:,:,cindex);
        gd.Experiment(findex).clim(cindex).limits = [min(temp(:)), max(temp(:))];
        gd.Experiment(findex).clim(cindex).current = [gd.Experiment(findex).clim(cindex).limits(1), prctile(temp(:),99.98)];
        set(gd.(sprintf('Data%d',index)).axes,'CLim',gd.Experiment(findex).clim(cindex).current);
        gd.Experiment(findex).colormap(cindex).current = colormap;
    end
end

% Set current colormaps
gd = AdjustColorMaps(gd, offset+1:offset+numFiles);

% Add Files to Table
% contents = get(gd.Control.selection, 'Data');
% for findex = offset+1:offset+numFiles %{'File','View','Move','Group','X','Y','Path'}
%     [p,f,~] = fileparts(gd.Experiment(findex).filename);
%     contents{findex, 1} = f;
%     contents{findex, 2} = false;
%     contents{findex, 3} = false;
%     contents{findex, 4} = gd.Experiment(findex).Map.YWorldLimits(1) - 0.5;
%     contents{findex, 5} = gd.Experiment(findex).Map.XWorldLimits(1) - 0.5;
%     contents{findex, 6} = p;
% end
% contents{1, 2} = true; % view first file
% set(gd.Control.selection, 'Data', contents, 'Enable', 'on');
% set(gd.Control.save, 'Enable', 'on');

guidata(hObject, gd);
plotDataAxes(gd, index);
plotMergeAxes(gd)

function gd = AdjustColorMaps(gd, indices)
for findex = indices
    gd.Experiment(findex).current = gd.Experiment(findex).Image;
    % Adjust colormap
    for cindex = 2
        temp = gd.Experiment(findex).current(:,:,cindex);
        temp = temp./gd.Experiment(findex).clim(cindex).current(2);
        temp(temp>1) = 1;
        temp(temp<gd.Experiment(findex).clim(cindex).current(1)/gd.Experiment(findex).clim(cindex).current(2)) = gd.Experiment(findex).clim(cindex).current(1)/gd.Experiment(findex).clim(cindex).current(2);
        gd.Experiment(findex).current(:,:,cindex) = temp;
    end
end

function  LoadROIs(hObject,eventdata,gd)

% Select files
if ~iscell(eventdata)
    if isdir(gd.Internal.directory)
        directory = gd.Internal.directory;
    elseif exist('CanalSettings.m', 'file');
        directory = CanalSettings('DataDirectory');
    else
        directory = cd;
    end
    [ROIFiles, p] = uigetfile({'*.mat'},'Choose ROI file(s)', directory, 'MultiSelect', 'on');
    if isnumeric(ROIFiles)
        return
    elseif iscell(ROIFiles)
        for findex = 1:numel(ROIFiles)
            ROIFiles{findex} = fullfile(p, ROIFiles{findex});
        end
    elseif ischar(ROIFiles)
        ROIFiles = {fullfile(p, ROIFiles)};
    end
%     ROIFiles = uipickfiles('Prompt', 'Choose ROI files', 'FilterSpec', [directory, '*.mat']);
%     if isnumeric(ROIFiles)
%         return
%     end
else
    ROIFiles = eventdata;
end
[gd.Internal.directory,~,~] = fileparts(ROIFiles{1});
numFiles = numel(ROIFiles);
index = get(hObject, 'UserData');

% Load ROI Data
if isfield(gd, 'ROIs') && numel(gd.ROIs) >= index  % previous files loaded
    offset = numel(gd.ROIs(index).files); % add files to end
else
    offset = 0;
    gd.ROIs(index).list = {};
    gd.ROIs(index).n = [];
end
for findex = 1:numFiles
    % Save file info
    gd.ROIs(index).files(offset + findex).filename = ROIFiles{findex};
    gd.ROIs(index).files(offset + findex).index = index;
    load(ROIFiles{findex}, 'ROIdata', '-mat');
    gd.ROIs(index).files(offset + findex).ROIdata = ROIdata;
    gd.ROIs(index).files(offset + findex).n = numel(ROIdata.rois);
    
    % Save aggregated info
    gd.ROIs(index).n = cat(1, gd.ROIs(index).n, gd.ROIs(index).files(offset + findex).n);
    gd.ROIs(index).list = [gd.ROIs(index).list, {gd.ROIs(index).files(offset + findex).ROIdata.rois(:).tag}];
end

% Update GUI
set(gd.(sprintf('Data%d',index)).list, 'String', gd.ROIs(index).list, 'Max', sum(gd.ROIs(index).n), 'Min', 0, 'Value', []);
set(gd.Match.save, 'Enable', 'on');

guidata(hObject, gd);
plotDataAxes(gd, index);

function AddROI(hObject,eventdata,gd)
dataIndex = get(hObject, 'UserData');
newIndex = abs(dataIndex-2)+1;

% Determine selected ROIs
roiSelection = get(gd.(sprintf('Data%d',dataIndex)).list, 'Value');
numROIs = numel(roiSelection);
blockStarts = cumsum(gd.ROIs(dataIndex).n);
blockStarts = [0,blockStarts(1:end-1)];
location = bsxfun(@minus, roiSelection', repmat(blockStarts, numROIs, 1));
location(location<=0) = inf;
[ROIindices, FileIndices] = min(location, [], 2);

% % Assign tags to ROIs being added that are not tagged
% for rindex = 1:numROIs
%     if all(ismember(gd.ROIs(dataIndex).files(FileIndices(rindex)).ROIdata.rois(ROIindices(rindex)).tag,'0123456789'));
%         index = get(gd.Match.index, 'String');
%         gd.ROIs(dataIndex).files(FileIndices(rindex)).ROIdata.rois(ROIindices(rindex)).tag = strcat(get(gd.Match.ID, 'String'), get(gd.Match.base, 'String'), index);
%         set(gd.Match.index, 'String', num2str(str2double(index)+1));
%     end
% end

% Extract ROIs to add
Files = [gd.Experiment(:).index];
FileDict = find(Files == dataIndex); 
% Offset = zeros(numROIs, 1);
Maps = cell(numROIs, 1);
rois = gd.ROIs(dataIndex).files(FileIndices(1)).ROIdata.rois(ROIindices(1));
Maps{1} = gd.Experiment(FileDict(FileIndices(1))).Map;
% Offset(1, :) = gd.ROIs(dataIndex).files(FileIndices(1)).Offset;
for rindex = 2:numROIs
    rois(rindex) = gd.ROIs(dataIndex).files(FileIndices(rindex)).ROIdata.rois(ROIindices(rindex));
%     Offset(rindex, :) = gd.ROIs(dataIndex).files(FileIndices(rindex)).Offset;
    Maps{rindex} = gd.Experiment(FileDict(FileIndices(rindex))).Map;
end

% Determine translation from current file to new file
% NewFileOffset = gd.ROIs(newIndex).files(1).Offset; % add to first file
% Translations = bsxfun(@minus, Offset, NewFileOffset);
NewFileDict = find(Files == newIndex, 1); 
NewFileMap = gd.Experiment(NewFileDict).Map;
Translations = zeros(numROIs, 2);
for rindex = 1:numROIs
    Translations(rindex, 1) = Maps{rindex}.XWorldLimits(1) - NewFileMap.XWorldLimits(1);
    Translations(rindex, 2) = Maps{rindex}.YWorldLimits(1) - NewFileMap.YWorldLimits(1);
end

% Translate ROIs
[H,W,~,~] = size(gd.Experiment(NewFileDict).ImageFiles.Average);
fH = figure('Visible', 'off');
imshow(zeros(H,W));
for rindex = 1:numROIs
    rois(rindex).vertices = bsxfun(@plus, rois(rindex).vertices, Translations(rindex, :));
    switch rois(rindex).type
        case 'ellipse'
            rois(rindex).position(1:2) = rois(rindex).position(1:2) + Translations(rindex, :);
            imH = imellipse(gca, rois(rindex).position);
        case 'polygon'
            imH = impoly(gca, rois(rindex).vertices, 'Closed', 1);
    end
    rois(rindex).pixels = createMask(imH); % extract mask
    delete(imH); % delete temporary UI ROI
    
    rois(rindex).mask = [];
    rois(rindex).neuropilmask = [];
    rois(rindex).rawdata = [];
    rois(rindex).rawneuropil = [];
end

% Bug fixes for Old files
fns = fieldnames(gd.ROIs(newIndex).files(1).ROIdata.rois);
if ~any(strcmp(fns, 'centroid'))
    rois = rmfield(rois, 'centroid');
end
if any(strcmp(fns, 'beingEdited'))
    gd.ROIs(newIndex).files(1).ROIdata.rois = rmfield(gd.ROIs(newIndex).files(1).ROIdata.rois, 'beingEdited');
end

% Save ROIs
nOrigROIs = numel(gd.ROIs(newIndex).list);
gd.ROIs(newIndex).files(1).ROIdata.rois(end+1:end+numROIs) = rois;
gd.ROIs(newIndex).list = [gd.ROIs(newIndex).list, {rois(:).tag}];
guidata(hObject, gd);

% Update GUI
set(gd.(sprintf('Data%d',newIndex)).list, 'String', gd.ROIs(newIndex).list, 'Value', nOrigROIs+1:nOrigROIs+numROIs);
plotDataAxes(gd, newIndex);

function gd = plotDataAxes(gd, dataIndex)
if ~isnumeric(dataIndex)
    dataIndex = get(dataIndex, 'UserData');
end

% Select images
viewSelection = find([gd.Experiment(:).index]==dataIndex);

% Determine composite image size
XLim = [inf, -inf];
YLim = [inf, -inf];
for dindex = viewSelection
    XLim(1) = min(XLim(1), gd.Experiment(dindex).Map.XWorldLimits(1));
    XLim(2) = max(XLim(2), gd.Experiment(dindex).Map.XWorldLimits(2));
    YLim(1) = min(YLim(1), gd.Experiment(dindex).Map.YWorldLimits(1));
    YLim(2) = max(YLim(2), gd.Experiment(dindex).Map.YWorldLimits(2));
end
% H = diff(YLim);
% W = diff(XLim);

% Determine offsets
numFiles = numel(viewSelection);
FileOffsets = zeros(numFiles, 2);
for dindex = 1:numFiles
    FileOffsets(dindex, :) = [gd.Experiment(viewSelection(dindex)).Map.XWorldLimits(1) - XLim(1), gd.Experiment(viewSelection(dindex)).Map.YWorldLimits(1) - YLim(1)];
end

% Create full image
img = gd.Experiment(viewSelection(1)).current(:,:,2);
map = gd.Experiment(viewSelection(1)).Map;
for index = 2:numel(viewSelection)
    [img, map] = imfuse(...
        img,...
        map,...
        gd.Experiment(viewSelection(index)).current(:,:,2),...
        gd.Experiment(viewSelection(index)).Map,...
        'blend',...
        'Scaling', 'Independent');
end

% Select axes
axes(gd.(sprintf('Data%d',dataIndex)).axes);

% Display Image
gd.Experiment(dataIndex).imghandle = imagesc(img);
set(gd.Experiment(dataIndex).imghandle, 'ButtonDownFcn', @(hObject,eventdata)ROIClickCallback(hObject,eventdata,guidata(hObject)));
set(gca,'xtick',[],'ytick',[])

% Display ROIs
if isfield(gd, 'ROIs') && (numel(gd.ROIs) < dataIndex || ~isempty(gd.ROIs(dataIndex).n))
    roiSelection = get(gd.(sprintf('Data%d',dataIndex)).list, 'Value');
    blockStarts = cumsum(gd.ROIs(dataIndex).n);
    blockStarts = [0,blockStarts(1:end-1)];
    location = bsxfun(@minus, roiSelection', repmat(blockStarts, numel(roiSelection), 1));
    location(location<=0) = inf;
    [ROIindices, FileIndices] = min(location, [], 2);
    if ~isempty(roiSelection)
        hold all;
        for rindex = 1:numel(ROIindices)
            linewidth = gd.Internal.Settings.ROILineWidth;
            temp = patch(...
                gd.ROIs(dataIndex).files(FileIndices(rindex)).ROIdata.rois(ROIindices(rindex)).vertices(:,1) + FileOffsets(FileIndices(rindex), 1),...
                gd.ROIs(dataIndex).files(FileIndices(rindex)).ROIdata.rois(ROIindices(rindex)).vertices(:,2) + FileOffsets(FileIndices(rindex), 2),...
                'red',...
                'Parent',               gd.(sprintf('Data%d',dataIndex)).axes,...
                'FaceAlpha',            0,...
                'EdgeColor',            'red',...
                'LineWidth',            linewidth,...
                'UserData',             ROIindices(rindex),...
                'ButtonDownFcn',        @(hObject,eventdata)ROIClickCallback(hObject,eventdata,guidata(hObject))); % display ROI
            %         if get(gd.ROI.showIDs, 'Value')
            %             text(...
            %                 gd.ROIs.rois(r).vertices(1,1),...
            %                 gd.ROIs.rois(r).vertices(1,2),...
            %                 num2str(r),...
            %                 'Color',                [1,1,1],...
            %                 'FontSize',             gd.Internal.Settings.ROITextSize); % display ROI's index on screen
            %         end
        end
        hold off;
    end
    plotMatchAxes(gd)
end

% Update GUI
if ~isempty(get(gd.Data1.list, 'Value')) && ~isempty(get(gd.Data2.list, 'Value'))
    set(gd.Match.match, 'Enable', 'on');
else
    set(gd.Match.match, 'Enable', 'off');
end

guidata(gd.fig,gd); % update guidata

function plotMergeAxes(gd)

% Create images
Images = cell(2,1);
Map = cell(2,1);
for mindex = 1:numel(unique([gd.Experiment(:).index]))
    
    % Select images
    fileIndices = find([gd.Experiment(:).index] == mindex);
    numFiles = numel(fileIndices);
    
    if numFiles
        Images{mindex} = gd.Experiment(fileIndices(1)).current(:,:,2);
        Map{mindex} = gd.Experiment(fileIndices(1)).Map;
        for index = 2:numFiles
            [Images{mindex}, Map{mindex}] = imfuse(...
                Images{mindex},...
                Map{mindex},...
                gd.Experiment(fileIndices(index)).current(:,:,2),...
                gd.Experiment(fileIndices(index)).Map,...
                'blend',...
                'Scaling', 'Independent');
        end
    end
    
end

% Determine display type
col = [1,1,1];
% DisplayType = get(gd.Control.overlay, 'Value');
% DisplayType = 1;
% if DisplayType == 1
    DisplayType = 'falsecolor';
    col = [2,1,2];
% elseif DisplayType == 2
%     DisplayType = 'falsecolor';
%     col = [1,2,2];
% elseif DisplayType == 3
%     DisplayType = 'blend';
% elseif DisplayType == 4
%     DisplayType = 'diff';
% end

% Create merged image
if all(~cellfun(@isempty, Images))
    switch DisplayType
        case 'falsecolor'
            img = imfuse(Images{1},Map{1},Images{2},Map{2},DisplayType,'Scaling','Independent','ColorChannels',col);
        otherwise
            img = imfuse(Images{1},Map{1},Images{2},Map{2},DisplayType,'Scaling','Independent');
    end
elseif isempty(Images{1})
    img = Images{1};
else
    img = Images{2};
end

axes(gd.Merge.axes);
imagesc(img);

function plotMatchAxes(gd)

% Initialize composite image
% XLim = [inf, -inf];
% YLim = [inf, -inf];
% for dindex = 1:numel(gd.Experiment)
%     XLim(1) = min(XLim(1), gd.Experiment(dindex).Map.XWorldLimits(1));
%     XLim(2) = max(XLim(2), gd.Experiment(dindex).Map.XWorldLimits(2));
%     YLim(1) = min(YLim(1), gd.Experiment(dindex).Map.YWorldLimits(1));
%     YLim(2) = max(YLim(2), gd.Experiment(dindex).Map.YWorldLimits(2));
% end
% H = diff(YLim);
% W = diff(XLim);
% Image = zeros(H, W, 3);

% Create full image
Image = gd.Experiment(1).current(:,:,2);
Map = gd.Experiment(1).Map;
for index = 2:numel(gd.Experiment)
    [Image, Map] = imfuse(...
        Image,...
        Map,...
        gd.Experiment(index).current(:,:,2),...
        gd.Experiment(index).Map,...
        'blend',...
        'Scaling', 'Independent');
end

% Determine ROIs
rois = [];
for index = 1:2
    roiSelection = get(gd.(sprintf('Data%d',index)).list, 'Value');
    if ~isempty(roiSelection)
        blockStarts = cumsum(gd.ROIs(index).n);
        blockStarts = [0,blockStarts(1:end-1)];
        location = bsxfun(@minus, roiSelection', repmat(blockStarts, numel(roiSelection), 1));
        location(location<=0) = inf;
        [ROIindices, FileIndices] = min(location, [], 2);
        rois = cat(1, rois, cat(2, ROIindices, FileIndices, repmat(index, numel(roiSelection), 1), roiSelection'));
    end
end
numrois = size(rois, 1);

% Display ROIs
[H,W] = size(Image);
Image = zeros(H, W, 3);
Files = [gd.Experiment(:).index];
ColorIndex = [1,3];
for rindex = 1:numrois
    FileIndices = find(Files == rois(rindex, 3));
    Image(:,:,ColorIndex(rois(rindex, 3))) = imfuse(...
        Image(:,:,ColorIndex(rois(rindex, 3))),...
        Map,...
        gd.ROIs(rois(rindex, 3)).files(rois(rindex, 2)).ROIdata.rois(rois(rindex, 1)).pixels,...
        gd.Experiment(FileIndices(rois(rindex, 2))).Map,...
        'blend',...
        'Scaling', 'Independent');
end

axes(gd.Match.axes);
imagesc(Image);

function MatchROIs(hObject,eventdata,gd)

rois = [];
TopSelection = ones(1,2);
for index = 1:2
    roiSelection = get(gd.(sprintf('Data%d',index)).list, 'Value');
    TopSelection(index) = roiSelection(1);
    if ~isempty(roiSelection)
        blockStarts = cumsum(gd.ROIs(index).n);
        blockStarts = [0,blockStarts(1:end-1)];
        location = bsxfun(@minus, roiSelection', repmat(blockStarts, numel(roiSelection), 1));
        location(location<=0) = inf;
        [ROIindices, FileIndices] = min(location, [], 2);
        rois = cat(1, rois, cat(2, ROIindices, FileIndices, repmat(index, numel(roiSelection), 1), roiSelection'));
    end
end
numrois = size(rois, 1);

if numrois
    contents = get(gd.Match.type, 'String');
    switch contents{get(gd.Match.type, 'Value')}
        case 'all'
            index = get(gd.Match.index, 'String');
            Tag = strcat(get(gd.Match.ID, 'String'), get(gd.Match.base, 'String'), index);
            set(gd.Match.index, 'String', num2str(str2double(index)+1));
            Tags = repmat({Tag},numrois,1);
        case 'pair-wise'
            % match first selected to first selected, etc.
            LeftIndices = find(rois(:,3)==1);
            RightIndices = find(rois(:,3)==2);
            numPairs = numel(LeftIndices);
            Tags = cell(numrois,1);
            for pindex = 1:numPairs
                index = get(gd.Match.index, 'String');
                Tag = strcat(get(gd.Match.ID, 'String'), get(gd.Match.base, 'String'), index);
                set(gd.Match.index, 'String', num2str(str2double(index)+1));
                Tags([LeftIndices(pindex);RightIndices(pindex)]) = {Tag};
            end
    end
%     % Determine if any matched ROIs have already been assigned a tag
%     hasTag = false(numrois, 1);
%     for rindex = 1:numrois
%         if ~all(ismember(gd.ROIs(rois(rindex, 3)).files(rois(rindex, 2)).ROIdata.rois(rois(rindex, 1)).tag,'0123456789'));
%             hasTag(rindex) = true;
%         end
%     end
%     % Determine or assign tag
%     if sum(hasTag) > 1
%         warning('Overwriting previously assigned tag');
%     end
%     if any(hasTag) % previously assigned tag exists
%         rindex = find(hasTag, 1); % use first tag
%         Tag = gd.ROIs(rois(rindex, 3)).files(rois(rindex, 2)).ROIdata.rois(rois(rindex, 1)).tag;
%     else % assign new tag
%         index = get(gd.Match.index, 'String');
%         Tag = strcat(get(gd.Match.ID, 'String'), get(gd.Match.base, 'String'), index);
%         set(gd.Match.index, 'String', num2str(str2double(index)+1));
%     end
    
    % Record to structure
    for rindex = 1:numrois
        gd.ROIs(rois(rindex, 3)).files(rois(rindex, 2)).ROIdata.rois(rois(rindex, 1)).tag = Tags{rindex};
        gd.ROIs(rois(rindex, 3)).list{rois(rindex, 4)} = Tags{rindex};
    end
    guidata(hObject, gd);
    
    % Record to lists
    for index = 1:2
        set(gd.(sprintf('Data%d',index)).list, 'String', gd.ROIs(index).list, 'ListboxTop', TopSelection(index));
        if TopSelection(index) < get(gd.(sprintf('Data%d',index)).list, 'Max')
            set(gd.(sprintf('Data%d',index)).list, 'Value', TopSelection(index)+1);
        end
        gd = plotDataAxes(gd, index);
    end   
end

function SaveROIs(hObject,eventdata,gd)
set(gcf, 'pointer', 'watch');
drawnow
fprintf('Saving %d files:\n', numel(gd.ROIs));
for mindex = 1:numel(gd.ROIs)
    for findex = 1:numel(gd.ROIs(mindex).files)
        ROIdata = gd.ROIs(mindex).files(findex).ROIdata;
        save(gd.ROIs(mindex).files(findex).filename, 'ROIdata', '-append');
        fprintf('\tSaved ROIdata to: %s\n', gd.ROIs(mindex).files(findex).filename);
    end
end
set(gcf, 'pointer', 'arrow');