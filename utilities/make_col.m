function [x,flip] = make_col(x)
flip = false;
if size(x,1)==1
    flip = true;
    x = x(:);
end