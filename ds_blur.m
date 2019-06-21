function arr_ds = ds_blur(arr,n,dim)
if nargin < 2
    n = 2;
end
if nargin < 3
    dim = 1;
end

% find shape of input array
sz = size(arr);

% generate boxcar filter
sz_filt = ones(1,numel(sz));
sz_filt(dim) = n;
block_filt = ones(sz_filt)/n;

% filter input
arr_blur = convn(arr,block_filt,'same');

% create subscript object
S.type = '()';
S.subs = cell(1,numel(sz));
for i=1:numel(sz)
    S.subs{i} = ':';
end
S.subs{dim} = 1:n:sz(dim);

% downsample filtered input
arr_ds = subsref(arr_blur,S);