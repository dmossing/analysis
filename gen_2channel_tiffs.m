function gen_2channel_tiffs(foldname,filenames,options)
if nargin < 3
    options = [];
end

green_only = getOr(options,'green_only',0);
result_foldbase = getOr(options,'result_foldbase',...
    '/home/mossing/modulation/visual_stim/');
data_foldbase = getOr(options,'data_foldbase',...
    '/home/mossing/data_ssd/2P/');
targetfold = getOr(options,'targetfold',...
    '/home/mossing/data_ssd/suite2P/raw/');

% result_foldbase = '/home/mossing/modulation/visual_stim/';
% targetfold = '/home/mossing/data_ssd/suite2P/raw/';

opts.chunksize = 1000;
opts.green_only = green_only;

if ~strcmp(data_foldbase(end),'/')
    data_foldbase = [data_foldbase '/'];
end

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
        opts.opto_correct = false; % temporary for 19/12/5 analysis! true;
        opts.opto_settings.type = 'exp';
        opts.opto_settings.sbxbase = sprintf('%s/%s/',data_foldbase,thisfoldname);
        opts.opto_settings.filebase = fnames{i}(1:end-4);
        opts.opto_settings.resultbase = sprintf('%s/%s/',result_foldbase,thisfoldname);
        sbx_to_cropped_tiffs([data_foldbase thisfoldname '/' fnames{i}(1:end-4)],opts);
    end
end

function val = getOr(options,fieldname,default)
if isempty(options) || ~isfield(options,fieldname) || isempty(getfield(options,fieldname))
    val = default;
else
    val = getfield(options,fieldname);
end
