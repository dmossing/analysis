function roi_select_multichannel(imr,img,msk,selected,deselected)
% wait for the space bar to be pressed
% on space bar press, which which ROI is being selected, and toggle the
% value of the boolean array "selected" for that ROI
% then replot the ROI masks
% visboundaries(sum(msk,3))
S.fh = figure('units','pixels','keypressfcn',@fh_kpfcn); % 'buttondownfcn',@fh_bdfcn);
S.msk = msk>0;
S.nroi = size(msk,3);
if nargin < 4 | isempty(selected)
    S.selected = false(S.nroi,1);
else
    S.selected = selected;
end
if nargin < 5 | isempty(deselected)
    S.deselected = false(S.nroi,1);
else
    S.deselected = deselected;
end
S.patches = [];
S.msk_visibility = true;
S.red_mult = 1;
S.green_mult = 1;
S.imr0 = imr;
S.img0 = img;
S.CLimr0 = [min(imr(:)) max(imr(:))];
S.CLimg0 = [min(img(:)) max(img(:))];
S.CLimr = S.CLimr0;
S.CLimg = S.CLimg0;
S = set_clim(S);
S.im2ch = gen_2channel(S.imr,S.img);
S.scale_step = -log(1-0.1);
S.img_handle = [];
S = display_img(S);
S.axes = get(S.fh,'CurrentAxes');
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
    case 'n'
        disp('registered n')
        [x,y] = ginput(1);
        S = toggle_roi_deselect(S,round([x y]));
    case 'r'
        S.red_mult = S.red_mult*exp(-S.scale_step);
        S = set_clim(S);
        S.img_handle.CData = S.im2ch;
        %         S = display_img(S);
        %         S = display_msk(S);
    case 't'
        S.red_mult = S.red_mult*exp(S.scale_step);
        S = set_clim(S);
        S.img_handle.CData = S.im2ch;
        %         S = display_img(S);
        %         S = display_msk(S);
    case 'g'
        S.green_mult = S.green_mult*exp(-S.scale_step);
        S = set_clim(S);
        S.img_handle.CData = S.im2ch;
        %         S = display_img(S);
        %         S = display_msk(S);
    case 'h'
        S.green_mult = S.green_mult*exp(S.scale_step);
        S = set_clim(S);
        S.img_handle.CData = S.im2ch;
        %         S = display_img(S);
        %         S = display_msk(S);
    case 'v'
        S = toggle_msk_visibility(S);
end
guidata(S.fh,S);

function S = display_img(S)
if ~isempty(S.img_handle)
    delete(S.img_handle)
end
S.img_handle = image(S.im2ch);
% S.img_handle = image(S.red_mult*S.img);
guidata(S.fh,S);

function S = set_clim(S)
S.CLimr = S.CLimr0;
S.CLimg = S.CLimg0;
S.CLimr(2) = S.CLimr(1) + (S.CLimr(2)-S.CLimr(1))/S.red_mult;
S.CLimg(2) = S.CLimg(1) + (S.CLimg(2)-S.CLimg(1))/S.green_mult;
S.imr = scale_im(S.imr0,S.CLimr);
S.img = scale_im(S.img0,S.CLimg);
S.im2ch = gen_2channel(S.imr,S.img);
guidata(S.fh,S);

function im_scaled = scale_im(im,CLim)
im_scaled = (im - CLim(1))/(CLim(2)-CLim(1));
im_scaled = min(im_scaled,1);

function im2ch = gen_2channel(imr,img)
im2ch = zeros(size(imr,1),size(imr,2),3);
im2ch(:,:,1) = imr;
im2ch(:,:,2) = img;

function S = display_msk(S)
% if ~isempty(S.patches)
%     for iroi=1:S.nroi
%         delete(S.patches(iroi));
%     end
% end
for iroi=1:S.nroi
    S = display_single_patch(S,iroi);
%     bw = bwconvhull(S.msk(:,:,iroi)>0);
%     bw = bwboundaries(bw);
%     bw = bw{1};
%     if S.selected(iroi)
%         this_color = 'r';
%     elseif S.deselected(iroi)
%         this_color = 'w';
%     else
%         this_color = 'c';
%     end
%     S.patches = [S.patches; patch(bw(:,2),bw(:,1),this_color,'FaceColor','none','EdgeColor',this_color)]; %,'FaceColor','None','EdgeColor',this_color)];
end
guidata(S.fh,S)

function S = toggle_roi(S,pt)
in_this_roi = squeeze(S.msk(pt(2),pt(1),:)>0);
S.selected(in_this_roi) = ~S.selected(in_this_roi);
for iroi=1:S.nroi
    if in_this_roi(iroi)
        S = set_patch_color(S,iroi);
    end
end
guidata(S.fh,S)

function S = toggle_roi_deselect(S,pt)
in_this_roi = squeeze(S.msk(pt(2),pt(1),:)>0);
S.deselected(in_this_roi) = ~S.deselected(in_this_roi);
for iroi=1:S.nroi
    if in_this_roi(iroi)
        S = set_patch_color(S,iroi);
    end
end
guidata(S.fh,S)

function S = toggle_msk_visibility(S)
S.msk_visibility = ~S.msk_visibility;
if S.msk_visibility
    vis_val = 'on';
else
    vis_val = 'off';
end
for iroi=1:S.nroi
    set(S.patches(iroi),'visible',vis_val);
end

% function S = display_single_patch(S,patch_ind)
% if ~isempty(S.patches)
%     delete(S.patches(patch_ind));
% end
% for iroi=patch_ind
%     bw = bwboundaries(S.msk(:,:,iroi)>0);
%     bw = bw{1};
%     if S.selected(iroi)
%         this_color = 'r';
%     elseif S.deselected(iroi)
%         this_color = 'w';
%     else
%         this_color = 'c';
%     end
%     S.patches(iroi) = patch(bw(:,2),bw(:,1),this_color,'FaceColor','none','EdgeColor',this_color); %,'FaceColor','None','EdgeColor',this_color)];
% end
% guidata(S.fh,S)

function S = display_single_patch(S,iroi)
if numel(S.patches)>=iroi
    delete(S.patches(iroi));
    first_time = false;
else
    first_time = true;
end
% bw = bwconvhull(S.msk(:,:,iroi)>0);
% bw = bwboundaries(bw);
this_msk = S.msk(:,:,iroi);
bw = bwboundaries(this_msk);
if first_time
    se = strel('disk',1);
    nsteps = 0;
    while numel(bw)>1
        this_msk = imdilate(this_msk,se);
        bw = bwboundaries(this_msk);
        if nsteps>1
            nsteps
        end
        nsteps=nsteps+1;
    end
    S.msk(:,:,iroi) = this_msk;
end
bw = bw{1};
this_patch = patch(bw(:,2),bw(:,1),'w','visible',false);
if first_time
    S.patches = [S.patches; this_patch];
else
    S.patches(iroi) = this_patch;
end
S = set_patch_color(S,iroi);
guidata(S.fh,S);

function S = set_patch_color(S,iroi)
if S.selected(iroi)
    this_color = 'r';
elseif S.deselected(iroi)
    this_color = 'w';
else
    this_color = 'c';
end
set(S.patches(iroi),'FaceColor','None');
set(S.patches(iroi),'EdgeColor',this_color);
guidata(S.fh,S);