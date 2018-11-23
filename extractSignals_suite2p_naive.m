function extractSignals_suite2p_naive(suffix,sbxbase)
 
%%
if nargin < 2
    sbxbase = '/home/mossing/scratch/2Pdata/';
end
foldbase = [sbxbase 'suite2P/'];

aux = strsplit(suffix,'/');
animalid = aux{1};
dstr = aux{2};
targetfold = [sbxbase dstr '/' animalid '/ot/'];
if ~exist(targetfold,'dir')
    mkdir(targetfold)
    matfiles = dirnames([sbxbase dstr '/' animalid '/*.mat']);
    for i=1:numel(matfiles)
        copyfile(matfiles{i},targetfold);
    end
end

resultfold = [foldbase 'results/'];
registeredfold = [foldbase 'registered/'];
thisresultfold = [resultfold suffix '/'];
thisregisteredfold = [registeredfold suffix '/'];

subfolds = dirnames(thisresultfold);
for i=1:numel(subfolds)
    fnames = dirnames([thisresultfold subfolds{i} '/F_*_proc.mat']);
    registeredfolds = dirnames([thisregisteredfold subfolds{i}]);
    splitsubfolds = strsplit(subfolds{i},'_');
    ROIdata = cell(numel(fnames),numel(splitsubfolds));
%     ROIfile = cell(size(fnames));
    for j=1:numel(fnames)
            fullname = [thisresultfold subfolds{i} '/' fnames{j}];
            nonproc = strsplit(fullname,'_proc');
            nonproc = [nonproc{1} '.mat'];
            ROIdata(j,:) = suite2P2ROIdata(nonproc,'ROIindex',true); % this outputs ROIdata as a cell array
            
        for k=1:numel(splitsubfolds)
            orig = dirnames([foldbase 'raw/' suffix '/' splitsubfolds{k} '/*.tif']);
            depth = strsplit(orig{1},'_');
            depth = depth{2};
            ROIfile = [targetfold animalid '_' depth sprintf('_%03d_ot_%03d.rois',str2num(num2str(splitsubfolds{k})),j-1)];
            createMasks(ROIdata{j,k}, 'Save', 'SaveFile', ROIfile);
            rf = load(ROIfile,'-mat');
            ROIdata{j,k} = rf.ROIdata;
            fullreg = [thisregisteredfold splitsubfolds{k} '/Plane' num2str(j) '/'];
            Images = dirnames([fullreg '*.tif'],fullreg);
            [ROIdata{j,k}, Data, Neuropil, ROIindex] = extractSignals(Images, ...
                ROIdata{j,k},[],'MotionCorrect',false, 'Save', 'SaveFile', ROIfile);
        end
    end
end

%%

