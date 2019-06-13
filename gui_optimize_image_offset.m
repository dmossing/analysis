function output = gui_optimize_image_offset(imr,img,initial_offset)
if nargin < 3
    initial_offset = [0,0];
end
global offset keep_going
offset = initial_offset;
imr = imr/max(imr(:));
img = img/max(img(:));
S.fh = figure('units','pixels','keypressfcn',@fh_kpfcn);
title('Use arrow keys to move the green image. Press any other key when finished.')
guidata(S.fh,S)
keep_going = true;
while keep_going
    S.image = combine_rg(imr,circshift(img,offset),1,1);
    imshow(S.image)
end
output = offset;

function [] = fh_kpfcn(H,E)      
global offset keep_going
S = guidata(H);
switch E.Key
    case 'rightarrow'
        offset = offset+[0,1];
    case 'leftarrow'
        offset = offset+[0,-1];
    case 'uparrow'
        offset = offset+[-1,0];
    case 'downarrow'
        offset = offset+[1,0];
    otherwise
        keep_going = false;
end