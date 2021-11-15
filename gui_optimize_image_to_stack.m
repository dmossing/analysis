function output = gui_optimize_image_to_stack(im3d,im2d,options)
% imr a 3d stack, at least as large as the single plane image
% img a single plane image
if nargin < 3
    options = struct;
end
[options,initial_transform] = get_or(options,'initial_transform',[0 0 201 1 1 1]);
[options,scale_step] = get_or(options,'scale_step',-log(1-0.01));
[options,red_green] = get_or(options,'red_green',true);
[options,msk] = get_or(options,'msk',[]);
S.imr = im3d;
S.img = im2d;
S.mskg = msk;
S.scale_step = scale_step;
S.offset = initial_transform(1:2);
S.z = initial_transform(3);
S.scale = initial_transform(4);
S.red_mult = initial_transform(5);
S.green_mult = initial_transform(6);
S.imr = double(S.imr)/double(max(S.imr(:)));
S.img = double(S.img/max(S.img(:)));
if size(S.img,1)<size(S.imr,1)
    S.img = center_image(S.img,[size(S.imr,1),size(S.imr,2)]);
    if ~isempty(S.mskg)
        S.mskg = center_image(S.mskg,[size(S.imr,1),size(S.imr,2)]);
    end
end
S.imrr = S.imr(:,:,S.z);
S.imgg = gen_rescaled(S.img,S.scale);
if ~isempty(S.mskg)
    S.mskgg = gen_rescaled(S.mskg,S.scale);
end
S.fh = figure('units','pixels','keypressfcn',@fh_kpfcn,'buttondownfcn',@fh_bdfcn);
S.axis = gca;
title('Use arrow keys to move the green image. Press any other key when finished.')
S.keep_going = true;
guidata(S.fh,S)
while S.keep_going
    S = guidata(S.fh);
    S.image = combine_rg(S.imrr,S.imgg,red_green,1,[S.red_mult S.green_mult]);
    if ~isempty(S.mskg)
        se = strel('disk',2);
        bdy = imdilate(S.mskgg,se) & ~S.mskgg;
        [ishow,jshow] = find(bdy);
        for i=1:numel(ishow)
            S.image(ishow(i),jshow(i),:) = 1;
        end
        
%         visboundaries(S.mskgg,'LineWidth',0.1,'Color','m')
    end
    imshow(S.image,'InitialMagnification',300) %,'Parent',S.axis)
    title(S.z)
end
output = [S.offset S.z S.scale S.red_mult S.green_mult];
% [ops.yblock,ops.xblock] = gen_blocks(size(S.imrr),128,0.5);
% stack_image = S.imrr;
% flowed_image = S.imgg;
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
    otherwise
        S.keep_going = false;
end
S.imrr = S.imr(:,:,S.z);
S.imgg = gen_rescaled(S.img,S.scale);
S.imgg = circshift(S.imgg,S.offset);
if ~isempty(S.mskg)
    S.mskgg = gen_rescaled(S.mskg,S.scale);
    S.mskgg = circshift(S.mskgg,S.offset);
end
guidata(S.fh,S)

% function [] = fh_bdfcn(H,E)
% disp('here')

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