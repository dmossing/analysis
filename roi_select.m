function selected = roi_select(img,msk,selected)
% wait for the space bar to be pressed
% on space bar press, which which ROI is being selected, and toggle the
% value of the boolean array "selected" for that ROI
% then replot the ROI masks
% visboundaries(sum(msk,3))
S.fh = figure('units','pixels','keypressfcn',@fh_kpfcn); % 'buttondownfcn',@fh_bdfcn);
S.msk = msk;
S.nroi = size(msk,3);
if nargin < 3
    S.selected = false(S.nroi,1);
else
    S.selected = selected;
end
S.patches = [];
S.img = img;
S.CLim0 = [min(img(:)) max(img(:))];
S.CLim = S.CLim0;
S.scale_step = -log(1-0.1);
S.red_mult = 1;
S.img_handle = [];
S = display_img(S);
S.axes = get(S.fh,'CurrentAxes');
S = set_clim(S);
S = display_msk(S);
guidata(S.fh,S)
disp('ok')

% while S.keep_going
% end
% selected = S.selected;

function [] = fh_kpfcn(H,E)
S = guidata(H);
switch E.Key
    case 'q'
        assignin('base','selected',S.selected)
    case 'space'
        disp('registered space bar')
        [x,y] = ginput(1);
        S = toggle_roi(S,round([x y]));
%         S = display_msk(S);
    case 'r'
        S.red_mult = S.red_mult*exp(-S.scale_step);
        S = set_clim(S);
%         S = display_msk(S);
    case 't'
        S.red_mult = S.red_mult*exp(S.scale_step);
        S = set_clim(S);
%         S = display_msk(S);
end
guidata(S.fh,S);

function S = display_img(S)
if ~isempty(S.img_handle)
    delete(S.img_handle)
end
S.img_handle = imagesc(S.img);
% S.img_handle = image(S.red_mult*S.img);
guidata(S.fh,S);

function S = set_clim(S)
S.CLim = S.CLim0;
S.CLim(2) = S.CLim(1) + (S.CLim(2)-S.CLim(1))/S.red_mult;
S.axes.CLim = S.CLim;
guidata(S.fh,S);

function S = display_msk(S)
if ~isempty(S.patches)
    for iroi=1:S.nroi
        delete(S.patches(iroi));
    end
end
for iroi=1:S.nroi
    bw = bwboundaries(S.msk(:,:,iroi)>0);
    bw = bw{1};
    if S.selected(iroi)
        this_color = 'r';
    else
        this_color = 'c';
    end
    S.patches = [S.patches; patch(bw(:,2),bw(:,1),this_color,'FaceColor','none','EdgeColor',this_color)]; %,'FaceColor','None','EdgeColor',this_color)];
end
guidata(S.fh,S)

function S = toggle_roi(S,pt)
in_this_roi = squeeze(S.msk(pt(2),pt(1),:)>0);
S.selected(in_this_roi) = ~S.selected(in_this_roi);
for iroi=find(in_this_roi)
    S = display_single_patch(S,iroi);
end
guidata(S.fh,S)

function S = display_single_patch(S,patch_ind)
if ~isempty(S.patches)
    delete(S.patches(patch_ind));
end
for iroi=patch_ind
    bw = bwboundaries(S.msk(:,:,iroi)>0);
    bw = bw{1};
    if S.selected(iroi)
        this_color = 'r';
    else
        this_color = 'c';
    end
    S.patches(iroi) = patch(bw(:,2),bw(:,1),this_color,'FaceColor','none','EdgeColor',this_color); %,'FaceColor','None','EdgeColor',this_color)];
end
guidata(S.fh,S)