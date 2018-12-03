function im = motion_correct(im,T)
sz = size(im);
nsz = numel(sz);
indices = cell(nsz,1);
for i=1:nsz,
    indices{i} = ':';
end
for t=1:sz(end)
    indices{end} = t;
    im(indices{:}) = circshift(im(indices{:}),[T(t,:) zeros(1,nsz-3)]);
end
