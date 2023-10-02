function S = gui_optimize_image_to_stack_flowfield(im3d,im2d,options)
% imr a 3d stack, at least as large as the single plane image
% img a single plane image
if ~isfield(options,'S0') || isempty(options.S0)
    if nargin < 3
        options = struct;
    end
    [options,initial_transform] = get_or(options,'initial_transform',[0 0 201 1]);
    [options,scale_step] = get_or(options,'scale_step',-log(1-0.01));
    % [options,interpolant] = get_or(options,'interpolant',...
    %     @(x,y,v) scatteredInterpolant(x,y,v,'natural'));
    [options,interpolant] = get_or(options,'interpolant',...
        @(x,y,v) softmax_interpolant(x,y,v));
    S.scale_step = scale_step;
    S.offset = initial_transform(1:2);
    S.z = initial_transform(3);
    S.scale = initial_transform(4);
    S.red_mult = 1;
    S.green_mult = 1;
    S.interpolant = interpolant;
    S.interp_fn = @(x,y) 0;
    S.flowfield = zeros([size(im3d,1) size(im3d,2) 2]);
    S.mapped_pts1 = [];
    S.mapped_pts2 = [];
else
    S = options.S0;
end
[options,red_green] = get_or(options,'red_green',true);
S.imr = im3d;
S.img = im2d;
S.imr = double(S.imr)/double(max(S.imr(:)));
S.img = double(S.img/max(S.img(:)));
if size(S.img,1)<size(S.imr,1)
    S.img = center_image(S.img,[size(S.imr,1),size(S.imr,2)]);
end
S.imrr = S.imr(:,:,S.z);
S.imgg = gen_rescaled(S.img,S.scale);
S.fh = figure('units','pixels','keypressfcn',@fh_kpfcn);%,'buttondownfcn',@fh_bdfcn);
title('Use arrow keys to move the green image. Press any other key when finished.')
S.keep_going = true;
guidata(S.fh,S)
while S.keep_going
    S = guidata(S.fh);
    S.image = combine_rg(S.imrr,apply_flowfield(S.imgg,S.flowfield),red_green,1,[S.red_mult S.green_mult]);
    imshow(S.image,'InitialMagnification',300)
    title(S.z)
end
output = [S.offset S.z S.scale];
[ops.yblock,ops.xblock] = gen_blocks(size(S.imrr),128,0.5);
stack_image = S.imrr;
S.fh = [];

% [~,flowed_image] = register_and_squish(S.imrr,S.imgg,ops);


function [] = fh_kpfcn(H,E)
S = guidata(H);
switch E.Key
    case 'rightarrow'
        S.offset = S.offset+[0,1];
    case 'leftarrow'
        S.offset = S.offset+[0,-1];
    case 'uparrow'
        S.offset = S.offset+[-1,0];
    case 'downarrow'
        S.offset = S.offset+[1,0];
    case 'w'
        S.z = min(S.z+1,size(S.imr,3));
    case 's'
        S.z = max(S.z-1,1);
    case 'a'
        S.scale = S.scale*exp(-S.scale_step);
    case 'd'
        S.scale = min(S.scale*exp(S.scale_step),1);
    case 'r'
        S.red_mult = S.red_mult*exp(-S.scale_step);
    case 't'
        S.red_mult = S.red_mult*exp(S.scale_step);
    case 'g'
        S.green_mult = S.green_mult*exp(-S.scale_step);
    case 'h'
        S.green_mult = S.green_mult*exp(S.scale_step);
    case 'p'
        did_one = 0;
        while ~did_one || (size(S.mapped_pts1,1)<1)
            title('pick a blue point')
            [xb,yb] = ginput(1);
            title('pick the corresponding red point')
            [xr,yr] = ginput(1);
            [xy0] = apply_flowfield_to_pts(S.flowfield,[xb yb]);
            S.mapped_pts1 = [S.mapped_pts1; xy0];
            S.mapped_pts2 = [S.mapped_pts2; [xr yr]];
            did_one = 1;
        end
        x = S.mapped_pts2(:,1);
        y = S.mapped_pts2(:,2);
        %         p1 = apply_flowfield_to_pts(S.flowfield,S.mapped_pts1);
        v = S.mapped_pts1-S.mapped_pts2;
        [xinterp,yinterp] = meshgrid(1:size(S.imgg,2),1:size(S.imgg,1));
        for iaxis=1:2
            S.interp_fn = S.interpolant(x,y,v(:,iaxis));
            %             S.interp_fn = scatteredInterpolant(x,y,v(:,iaxis),'natural');
            S.flowfield(:,:,3-iaxis) = S.interp_fn(xinterp,yinterp);
        end
    otherwise
        S.keep_going = false;
end
S.imrr = S.imr(:,:,S.z);
S.imgg = gen_rescaled(S.img,S.scale);
S.imgg = circshift(S.imgg,S.offset);
guidata(S.fh,S)

function [] = fh_bdfcn(H,E)
disp('here')

function imqq = gen_rescaled(imq,scale)
imq_small = imresize(imq,scale);
imqq = center_image(imq_small,size(imq));

function imqq = center_image(imq_small,sz)
imqq = zeros(sz);
size_diff = sz-size(imq_small);
offset0 = floor(size_diff/2);
rgs = cell(2,1);
for iaxis=1:2
    rgs{iaxis} = (1+offset0(iaxis)):(size(imq_small,iaxis)+offset0(iaxis));
end
imqq(rgs{1},rgs{2}) = imq_small;

function [options,value] = get_or(options,key,value)
if isfield(options,key)
    value = options.(key);
else
    options.(key) = value;
end

function pts1 = apply_flowfield_to_pts(ff,pts0)
for iaxis=1:2
    idx = sub2ind(size(ff),round(pts0(:,2)),round(pts0(:,1)),(3-iaxis)*ones(size(pts0,1),1));
    ffvals = ff(idx);
    pts1(:,iaxis) = pts0(:,iaxis) + ffvals;
end