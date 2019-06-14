gma
function rgb = combine_rg(imr,img,rg,normalized)
if nargin < 3
    rg = 0;
end
if nargin < 4
    normalized = 0;
end
if ~normalized
    im1 = imr/max(imr(:));
    im2 = img/max(img(:));
else
    im1 = imr;
    im2 = img;
end
if rg
    r = im1;
    g = im2;
    b = zeros(size(imr));
else
%     y = im1;
%     c = im2;
    r = im1;
    g = im2;
    b = im2;
end
rgb = cat(3,r,g,b);