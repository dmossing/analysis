function move_suite2p_tiffs(exptbase,suite2pbase,datafoldname)
if nargin < 2
    % destination for raw tif files
    suite2pbase = '/home/mossing/scratch/2Pdata/suite2P/raw/';
end
if nargin < 3
    % folder within the source location that is one above the date
    % string-named folders
    datafoldname = '2Pdata';
end
disp(['exptbase: ' exptbase])
no_ot = strsplit(exptbase,'_ot_');
no_ot = no_ot{1}; % look at everything in filename before the "_ot_" suffix
disp(['no_ot: ' no_ot])
parts = strsplit(no_ot,'/');
namebase = parts{end};
disp(['namebase: ' namebase])
nameparts = strsplit(namebase,'_');
animalid = nameparts{1};
disp(['animalid: ' animalid])
exptno = num2str(str2num(nameparts{end}));
disp(['exptno: ' exptno])
ind2p = find(strcmp(parts,datafoldname));
dstr = parts{ind2p+1};
fnames = dirnames([exptbase '*.tif'],[strjoin(parts(1:end-1),'/') '/']);
targetfold = [suite2pbase animalid '/' dstr '/' exptno '/'];
disp(['targetfold: ' targetfold])
if ~exist([suite2pbase animalid],'dir')
    mkdir([suite2pbase animalid])
end
if ~exist([suite2pbase animalid '/' dstr],'dir')
    mkdir([suite2pbase animalid '/' dstr])
end
if ~exist(targetfold,'dir')
    mkdir(targetfold)
end
for i=1:numel(fnames)
    movefile(fnames{i},targetfold);
end
