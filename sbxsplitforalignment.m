function sbxsplitforalignment(fn,foldname)
if nargin < 2
    foldname = 'ot/';
end
if exist(fn,'dir')
    firstfold = fn;
    d = dir([firstfold '/*.sbx']);
    fnames = {d(:).name};
    for i = 1:numel(fnames)
        fn = [firstfold '/' fnames{i}(1:end-4)] % just the name
        sbxsplit(fn) 
        d = dir([fn '_ot_*.sbx']);
        depthno = numel(d);
        load(fn,'info')
        try
            info = rmfield(info,'otparam');
        end
        if ~exist(foldname,'dir')
            mkdir([firstfold '/' foldname])
        end
        for i=1:depthno
            sbxfile = d(i).name;
            matfile = [firstfold '/' foldname '/' strrep(d(i).name,'.sbx','.mat')];
            save(matfile,'info')
            movefile([firstfold '/' sbxfile], [firstfold '/' foldname '/' sbxfile])
        end
    end
else
    %%% THIS WON'T WORK UNLESS YOU'RE IN THE RIGHT FOLDER ALREADY
    %%% SOLN SHOULD USE THIS CMD: 
    sbxsplit(fn)
    d = dir([fn '_ot_*.sbx']);
    depthno = numel(d);
    load(fn,'info')
    try
        info = rmfield(info,'otparam');
    end
    if ~exist(foldname,'dir')
        mkdir(foldname)
    end
    for i=1:depthno
        sbxfile = d(i).name;
        matfile = [foldname '/' strrep(d(i).name,'.sbx','.mat')];
        save(matfile,'info')
        movefile(sbxfile, [foldname '/' sbxfile])
    end
end
