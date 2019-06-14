function gen_2channel_tiffs(foldname,filenames,green_only)
if nargin < 3
    green_only = 0;
end
data_foldbase = '/home/mossing/modulation/2P/';
result_foldbase = '/home/mossing/modulation/visual_stim/';

targetfold = '/home/mossing/data_ssd/suite2P/raw/';

opts.chunksize = 1000;
opts.green_only = green_only;

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