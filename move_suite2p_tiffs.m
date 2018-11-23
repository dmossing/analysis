function move_suite2p_tiffs(exptbase)
suite2pbase = '/home/mossing/scratch/2Pdata/suite2P/raw/';
parts = strsplit(exptbase,'/');
namebase = parts{end};
nameparts = strsplit(namebase,'_');
animalid = nameparts{1};
exptno = num2str(str2num(nameparts{end}));
dstr = parts{end-2};
fnames = dirnames([exptbase '*.tif'],[strjoin(parts(1:end-1),'/') '/']);
targetfold = [suite2pbase animalid '/' dstr '/' exptno '/'];
if ~exist(targetfold,'dir')
    mkdir(targetfold)
end
for i=1:numel(fnames)
    movefile(fnames{i},targetfold);
end