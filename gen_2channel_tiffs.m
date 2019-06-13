function gen_2channel_tiffs(foldname,filenames)
data_foldbase = '/media/greg/modulation/mossing/2P/';
result_foldbase = '/media/greg/modulation/mossing/visual_stim/';

targetfold = '/media/data/dan/suite2P/raw/';

opts.chunksize = 1000;
opts.green_only = 0;

thisfoldname = foldname;
d = dir([data_foldbase thisfoldname '/M*.mat']);
fnames = {d(:).name};
for i=1:numel(d)
    s = strsplit(fnames{i}(1:end-4),'_');
    exptno = str2num(s{end});
    if ismember(exptno,filenames)
        fileparts = strsplit(thisfoldname,'/');
        animalid = fileparts{2};
        dstr = fileparts{1};
        subfold = num2str(exptno);
        opts.targetfold = [targetfold animalid '/' dstr '/' subfold '/'];
        sbx_to_cropped_tiffs([data_foldbase thisfoldname '/' fnames{i}(1:end-4)],opts);
    end
end