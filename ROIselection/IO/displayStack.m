function displayStack(Images, CLim, CMap)

FrameIndex = 1:500; % loading only

%% Parse input arguments

% Select file
if ~exist('Images', 'var')
    [f,p] = uigetfile({'*.tif;*.sbx'}, 'Select image file');
    if isnumeric(f)
        return
    end
    Images = fullfile(p,f);
end

% Load images
if ischar(Images)
    ImageFile = Images;
    [gd.Images, gd.loadObj] = load2P(ImageFile, 'Type', 'Direct', 'Frames', FrameIndex, 'Double');
else
    gd.Images = Images;
end
gd.class = class(gd.Images);

% Format images
while ndims(gd.Images) < 5
    gd.Images = permute(gd.Images, [1:ndims(gd.Images)-1, ndims(gd.Images)+1, ndims(gd.Images)]);
end
gd.dim = size(gd.Images);

% Determine color limits
if ~exist('CLim', 'var') || isempty(CLim)
    gd.CLim = prctile(Images(linspace(1,numel(Images),512*796*100)), [1,99]);
    gd.CLimits = [min(gd.Images(:)), max(gd.Images(:))];
else
    gd.CLim = CLim;
    gd.CLimits = [min(gd.Images(:)), max(gd.Images(:))];
end

% Determine color map
if ~exist('CMap', 'var')
    gd.CMap = [];
%     gd.CMap = HiLoColormap(CLim(1), CLim(2));
else
    gd.CMap = CMap;
end

%% Code

% Create figure
gd.fig = figure;

% Create axis
if isequal(gd.class, 'logical')
    gd.axes = axes('Parent', gd.fig, 'Units', 'normalized', 'Position', [.1, .1, .8, .7]);
else
    gd.axes = axes('Parent', gd.fig, 'Units', 'normalized', 'Position', [.1, .2, .8, .6]);
end

% Create indexing sliders
sliderIndex = find(gd.dim(3:end)>1);
ndim = numel(sliderIndex);
for sindex = 1:ndim
    maxvalue = gd.dim(sliderIndex(sindex)+2);
    minorstep = 1/(maxvalue-1);
    yloc = .9+(ndim-sindex)*.1/ndim;
    gd.sliders(sindex) = uicontrol(...
        'Style',                'slider',...
        'Parent',               gd.fig,...
        'Units',                'normalized',...
        'Position',             [.1,yloc,.9,.1/ndim],...
        'Min',                  1,...
        'Max',                  maxvalue,...
        'Value',                1,...
        'SliderStep',           [minorstep,max(2*minorstep,.1)],...
        'ToolTipString',        sprintf('Dimension %d', sliderIndex(sindex)),...
        'UserData',             [sindex, sliderIndex(sindex), maxvalue],...
        'Callback',             @(hObject,eventdata)updateindex(hObject, eventdata, guidata(hObject)));
    gd.text(sindex) = uicontrol(...
        'Style',                'text',...
        'Parent',               gd.fig,...
        'Units',                'normalized',...
        'Position',             [0,yloc,.1,.1/ndim],...
        'String',               sprintf('1/%d', maxvalue),...
        'HorizontalAlignment',  'right');
end
gd.Position = ones(1,3);

% Create plot type dropdown
gd.Selection = uicontrol(...
    'Style',            'popupmenu',...
    'Parent',           gd.fig,...
    'String',           {'imagesc','bar3'},...
    'Units',            'normalized',...
    'Position',         [0,.05,.2,.05],...
    'Callback',         @(hObject,eventdata)plotmainaxes(guidata(hObject)));
if strcmp(gd.class,'logical')
    gd.Selection.Enable = 'off';
end

% Create colormap dropdown
gd.Colormap = uicontrol(...
    'Style',            'popupmenu',...
    'Parent',           gd.fig,...
    'String',           {'gray','parula'},...
    'Units',            'normalized',...
    'Position',         [0,0,.2,.05],...
    'Callback',         @(hObject,eventdata)plotmainaxes(guidata(hObject)));
if ~isempty(gd.CMap)
    gd.Colormap.String{3} = 'input';
end

% Create colorlimit sliders
for index = 1:2
    minorstep = 1/(gd.CLimits(2)-gd.CLimits(1));
    yloc = (index-1)*.1/2;
    gd.CLimSliders(index) = uicontrol(...
        'Style',                'slider',...
        'Parent',               gd.fig,...
        'Units',                'normalized',...
        'Position',             [.35,yloc,.45,.1/2],...
        'Min',                  gd.CLimits(1),...
        'Max',                  gd.CLimits(2),...
        'Value',                gd.CLimits(index),...
        'SliderStep',           [minorstep,max(2*minorstep,.1)],...
        'UserData',             index,...
        'Callback',             @(hObject,eventdata)updateCLim(hObject, eventdata, guidata(hObject)));
    gd.CLimText(index) = uicontrol(...
        'Style',                'text',...
        'Parent',               gd.fig,...
        'Units',                'normalized',...
        'Position',             [.2,yloc,.15,.1/2],...
        'String',               sprintf('%d', gd.CLimits(index)),...
        'HorizontalAlignment',  'right');
end
gd.CLim = gd.CLimits;

% Create colorlimit individual toggle
gd.CLimIndiv = uicontrol(...
    'Style',                'radiobutton',...
    'Parent',               gd.fig,...
    'Units',                'normalized',...
    'Position',             [.8,.05,.2,.05],...
    'Value',                0,...
    'String',               'CLim each',...
    'Callback',             @(hObject,eventdata)CLimIndiv(hObject, eventdata, guidata(hObject)));

% Create histeq toggle
gd.histeq = uicontrol(...
    'Style',                'radiobutton',...
    'Parent',               gd.fig,...
    'Units',                'normalized',...
    'Position',             [.8,0,.2,.05],...
    'Value',                0,...
    'String',               'hist eq',...
    'Callback',             @(hObject,eventdata)plotmainaxes(guidata(hObject)));

guidata(gd.fig, gd);

% Plot first image
plotmainaxes(gd);

% If output: wait for figure to be closed
% waitfor(gd.fig);

function updateindex(hObject, ~, gd)
% update index
gd.Position(hObject.UserData(2)) = round(hObject.Value);
guidata(hObject, gd);

% update text
gd.text(hObject.UserData(1)).String = sprintf('%d/%d', gd.Position(hObject.UserData(2)), hObject.UserData(3));

% plot new image
plotmainaxes(gd)


function updateCLim(hObject, ~, gd)
% update CLim
value = hObject.Value;
if hObject.UserData==1
    if value >= gd.CLim(2)
        hObject.Value = hObject.Min;
        return
    end
elseif hObject.UserData==2
    if value <= gd.CLim(1)
        hObject.Value = hObject.Max;
        return
    end
end
gd.CLim(hObject.UserData) = value;
guidata(hObject, gd);

% update text
gd.CLimText(hObject.UserData).String = sprintf('%d', gd.CLim(hObject.UserData));

% plot new image
plotmainaxes(gd);


function CLimIndiv(hObject, ~, gd)
% disable clim sliders
if hObject.Value
    set([gd.CLimSliders,gd.CLimText], 'Enable', 'off');
else
    set([gd.CLimSliders,gd.CLimText], 'Enable', 'on');
end

% plot new image
plotmainaxes(gd);


function plotmainaxes(gd)
axes(gd.axes)
img = gd.Images(:,:, gd.Position(1), gd.Position(2), gd.Position(3));
if gd.Selection.Value==1 % imagesc
    gd.histeq.Enable = 'on';
    if gd.histeq.Value
        img = adapthisteq(img, 'NumTiles', [16 16], 'Distribution', 'Exponential');
    end
    switch gd.class
        case 'logical'
            imshow(img);
        otherwise
            if gd.CLimIndiv.Value
                imagesc(img);
            else
                imagesc(img, gd.CLim);
            end
    end
elseif gd.Selection.Value==2 % bar3
    gd.histeq.Enable = 'off';
    temp = bar3(img);
    for index=1:numel(temp)
        temp(index).CData = temp(index).ZData;
        temp(index).FaceColor = 'interp';
    end
    if gd.CLimIndiv.Value
        zlim([min(img(:)),max(img(:))]);
    else
        gd.axes.CLim = gd.CLim;
        zlim(gd.CLimits);
    end
end

% Set colormap
if gd.Colormap.Value==1
    colormap('gray')
elseif gd.Colormap.Value==2
    colormap('parula')
elseif gd.Colormap.Value==3
    colormap(gd.CMap)
end
% axis off
% xlabel(sprintf('Frame %d', Index));

