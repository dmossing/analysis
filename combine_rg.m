function rgb = combine_rg(imr,img,rg,normalized,multipliers)
if nargin < 3
    rg = 0;
end
if nargin < 4
    normalized = 0;
end
if nargin < 5
    multipliers = [1 1];
end
if ~normalized
    im1 = imr/max(imr(:));
    im2 = img/max(img(:));
else
    im1 = imr;
    im2 = img;
end
im1 = im1*multipliers(1);
im2 = im2*multipliers(2);
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