function tf = in_rg(V,a,b)
% returns logical 
    ct = V;
    ct(:) = 1:numel(V);
    tf = (ct >= a) & (ct < b);
end