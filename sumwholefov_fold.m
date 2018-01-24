function sumwholefov_fold(foldname)
files = dir([foldname '/*.sbx']);
for i=1:numel(files)
    ftotal = sumwholefov([foldname '/' files(i).name]);
end