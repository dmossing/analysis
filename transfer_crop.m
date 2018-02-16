function transfer_crop(fns1,fns2)
fold1 = '';
fold2 = '';
if isdir(fns1)
    d = dir([fns1 '/*.sbx']);
    fold1 = fns1;
    fns1 = {d(:).name};
end
if isdir(fns2)
    d = dir([fns2 '/*.sbx']);
    fold2 = fns2;
    fns2 = {d(:).name};
end
assert(numel(fns1)==numel(fns2))
for i=1:numel(fns1)
    i
    filename1 = strrep([fold1 fns1{i}],'.sbx','.mat');
    filename2 = strrep([fold2 fns2{i}],'.sbx','.mat');
    load(filename1,'info')
    rect = info.rect;
    load(filename2,'info')
    if ~isfield(info,'rect')
        info.rect = rect;
        save(filename2,'info')
    else
        disp('crop already saved')
    end
end