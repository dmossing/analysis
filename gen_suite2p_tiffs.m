function gen_suite2p_tiffs(foldname,filenos,green_only,varargin)
% foldname: date/animalid
% filenos: which of the sorted .mat files to turn into suite2p tiffs
% green_only: whether to use only the green channel when both channels are
% available

p = inputParser;

p.addParameter('data_foldbase','/media/mossing/backup_0/data/2P/');
p.addParameter('result_foldbase','/home/mossing/modulation/visual_stim/');
p.addParameter('targetfold','/media/mossing/backup_0/data/suite2P/raw/');

p.parse(varargin{:});

data_foldbase = p.Results.data_foldbase;
result_foldbase = p.Results.result_foldbase;
targetfold = p.Results.targetfold;

d = dir([data_foldbase foldname '/M*.mat']);
fnames = {d(:).name}; 
for i=filenos
    sbx_to_cropped_tiffs([data_foldbase foldname '/' fnames{i}(1:end-4)],1000,'',green_only); % 1 for green only; otherwise 0
    move_suite2p_tiffs([data_foldbase foldname '/' fnames{i}(1:end-4)],targetfold,'2P'); 
end