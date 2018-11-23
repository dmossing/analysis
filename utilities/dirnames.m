function fnames = dirnames(pattern,withprefix) 
if nargin < 2
    withprefix = '';
end
d = dir(pattern);
fnames = {d(:).name};
fnames=sort(fnames(~ismember(fnames,{'.','..'})));
if withprefix
    for i=1:numel(fnames)
        fnames{i} = [withprefix fnames{i}];
    end
end
        