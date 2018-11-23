function run_preprocessing_fold(foldname,runbase)
% assumes foldname is the ot/ folder, containing .rois files
%%
if nargin < 2
    runbase = '/home/mossing/scratch/running/';
end
w = what(foldname);
p = w.path;
parts = strsplit(p,'/');
date = parts{end-2};
animalid = parts{end-1};
runfold = [runbase '/' date '/' animalid '/'];

run_preprocessing(foldname,runfold,'floorframes',false)
