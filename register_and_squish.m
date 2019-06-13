function [offset_image,flowed_image] = register_and_squish(im1,im2,ops)
% ops should have fields xblock, yblock, as in the convention used by
% suite2p
% computes a flow field offset_image, s.t.
% flowed_image = apply_flowfield(offset_image,im2) maximally resembles im1. 
offset = gui_optimize_image_offset(im1,im2);
%%
redundancy = zeros(size(im2));
for i=1:size(ops.yblock,1)
    yrg = ops.yblock(i,1)+1:ops.yblock(i,2);
    xrg = ops.xblock(i,1)+1:ops.xblock(i,2);
    redundancy(yrg,xrg) = redundancy(yrg,xrg)+1;
end
%%
offsets = zeros(size(ops.yblock,1),2);
A = im1;
B = circshift(im2,offset);
offset_image = zeros([size(im2) 2]);
for i=1:size(ops.yblock,1)
    yrg = ops.yblock(i,1)+1:ops.yblock(i,2);
    xrg = ops.xblock(i,1)+1:ops.xblock(i,2);
    [u,v] = fftalign_dpm(A(yrg,xrg),B(yrg,xrg));
    offsets(i,:) = [u v];
    for idim=1:2
        offset_image(yrg,xrg,idim) = offset_image(yrg,xrg,idim) - offset(idim) + offsets(i,idim);
    end
end
offset_image = offset_image./repmat(redundancy,[1 1 2]);
%%
flowed_image = apply_flowfield(im2,offset_image);