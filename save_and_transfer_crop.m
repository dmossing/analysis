function save_and_transfer_crop(foldname)
if foldname(end)~='/'
    foldname = [foldname '/'];
end
fnames = dirnames([foldname '*.mat'],foldname);
for i=1:numel(fnames)
    fnames{i} = fnames{i}(1:end-4);
end
save_crop_for_alignment(fnames(1));
for i=2:numel(fnames)
    transfer_crop(fnames(1),fnames(i))
end