function convert_npy_to_rois(foldname,targetbase)
if nargin < 2 || isempty(targetbase)
    targetbase = '/media/mossing/backup_1/data/2P';
end
if targetbase(end)=='/'
    targetbase = targetbase(1:end-1);
end
procfold = convert_to_procfile(foldname);
cd(procfold);
pathparts = strsplit(pwd,'/');
sourcefold = '.';
animalid = pathparts{end-2};
dstr = pathparts{end-1};
exptnos = pathparts{end};
expt_endings = strsplit(exptnos,'_');
targetfold = [targetbase '/' dstr '/' animalid '/ot/'];
if ~exist(targetfold,'dir')
    mkdir(targetfold)
    foldabove = [targetfold '../'];
    for i=1:numel(expt_endings)
        thismat = dirnames(sprintf([foldabove 'M*%s.mat'],expt_endings{i}),foldabove);
        copyfile(thismat,targetfold)
    end
end
gen_dot_roi(sourcefold,targetfold)
cd(targetfold)
run_preprocessing_fold('.','/home/mossing/modulation/running/');