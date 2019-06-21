%%
cd('/media/mossing/backup_1/data/2P')
[~,otfolds] = unix(['find . -name ot']);
otfolds = strsplit(otfolds,'./');
otfolds = otfolds(2:end);
orig_fold = pwd;
%%
for iexpt=15
    cd([otfolds{iexpt}(1:end-1)])
    d = dirnames('*.rois');
    if numel(d)
        run_preprocessing_fold('.','/home/mossing/modulation/running/')
    end
    cd(orig_fold)
end