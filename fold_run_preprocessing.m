function fold_run_preprocessing(foldnames)
datafoldbase = '/home/mossing/scratch/2Pdata/';
runfoldbase = '/home/mossing/modulation/running/';
for i=1:numel(foldnames)
    thisdata = [datafoldbase foldnames{i} '/ot/'];
    thisrunning = [runfoldbase foldnames{i}];
    run_preprocessing(thisdata,thisrunning,'floorframes',false);
end