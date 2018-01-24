function N = sbxchan0(M,c)
% M is a single frame to read, c is which channel you care about. 1 for
% green, 2 for red. Default to green.
if ~exist('c','var') || isempty(c)
    c = 1;
end
if ndims(M)==3
    N = squeeze(M(c,:,:));
else
    N = M;
end