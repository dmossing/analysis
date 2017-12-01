% function sbxsegment_sameFOV(fns,saveto,progress)
% 
% if nargin < 2 || isempty(saveto)
%     saveto = 'fake';
% end
% if nargin < 3 || isempty(progress)
%     if ~exist(saveto)
%         progress='new';
%     else
%         progress='continue';
%     end
% end
% assert(ismember(progress,{'new','continue'}));
% if progress=='new'
%     startat = 1;
% else
%     load(saveto,'startat');
% end
addpath(genpath('~/Documents/code/adesnik_code'))
addpath(genpath('~/Documents/code/adesnaq'))

%%
% fns_all = {'M7307_185_000_ot_000','M7307_185_001_ot_000';
%     'M7307_185_000_ot_001','M7307_185_001_ot_001';
%     'M7307_185_000_ot_002','M7307_185_001_ot_002';
%     'M7307_185_000_ot_003','M7307_185_001_ot_003'};

% fns_all = {'M7306_145_000','M7306_000_000','M7306_145_001','M7306_145_002','M7306_145_003'};

% fns_all = {'M7306_000_000','M7306_125_000','M7306_125_001','M7306_125_002','M7306_125_003'};

% fns_all = {'M7478_132_000'};

% fns_all = {'M7478_100_001_ot_000','M7478_100_002_ot_000','M7478_101_003_ot_000';
%     'M7478_100_001_ot_001','M7478_100_002_ot_001','M7478_101_003_ot_001';
%     'M7478_100_001_ot_002','M7478_100_002_ot_002','M7478_101_003_ot_002';
%     'M7478_100_001_ot_003','M7478_100_002_ot_003','M7478_101_003_ot_003'};

% fns_all = {'M7707_216_001_ot_000','M7707_216_002_ot_000';
%     'M7707_216_001_ot_001','M7707_216_002_ot_001';
%     'M7707_216_001_ot_002','M7707_216_002_ot_002';
%     'M7707_216_001_ot_003','M7707_216_002_ot_003'};

% fns_all = {'M7707_215_000','M7707_000_000','M7707_000_001'};

fns_all = {'M7447_000_003','M7447_200_000'};


%%

j = 1;

fns = fns_all(j,:);

N = numel(fns);

i = 2;

%%
% for i=startat:N
load(fns{i},'info')
dim = info.sz;
if i>1
    matchFoVs({[fns{1},'.align'],[fns{i},'.align']});
end
% moveon = 'n';
% while moveon ~= 'y'
%     moveon = input('All done aligning?\n','s');
% end
%%
if i>1
    transferROIs([fns{i-1},'.segment'], [fns{i-1},'.align'], ...
        [fns{i},'.align'], 'Save', 'SaveFile', [fns{i},'.segment']);
end
sbxsegmentflood(fns{i})
save([fns{i},'.segment'],'-mat','dim','-append')
%     if i<N
%         cont = input('Continue?\n','s');
%         if cont == 'n'
%             startat = i+1;
%             save(saveto,'startat');
%             break;
%         end
%     cont = input('Continue?\n','s');
%     if cont == 'n'
%         startat = i+1;
%         save(saveto,'startat');
%         break;
%     end
% end
%%
ifinal = i;
for j=1:ifinal-1
    transferROIs([fns{ifinal},'.segment'], [fns{ifinal},'.align'], ...
        [fns{j},'.align'], 'Save', 'SaveFile', [fns{j},'.segment']);
end
%%
for j=1:size(fns_all,1)
    % extract dFoF signals
    fns = fns_all(j,:);
    for i=1:N
        ROIFile = [fns{i}, '.rois'];
        ROIdata = sbxDistribute(fns{i}, 'Save', 'SaveFile', ROIFile); % intialize struct
        createMasks(ROIdata, 'Save', 'SaveFile', ROIFile); % create ROI masks
%         config = load2PConfig([fns{i}, '.sbx']);
%         load([fns{i} '.align'],'-mat','T')
        if ~isempty(strfind(fns{i},'_depth'))
            Depth = str2num(fns{i}(end));
            sbxfname = strsplit(fns{i},'_depth');
            sbxfname = [sbxfname{1} '.sbx'];
        else
            sbxfname = [fns{i} '.sbx'];
            Depth = 1;
        end
        Frames = idDepth(sbxfname,[],'Depth',Depth)';
%         frameno = size(T,1);
%         [~, Data, Neuropil, ~] = extractSignals([fns{i},'.sbx'], ROIFile, 'all', 'Save', 'SaveFile', ROIFile, 'MotionCorrect', [fns{i},'.align'], 'Frames', Frames);
        [~, Data, Neuropil, ~] = extractSignals(sbxfname, ROIFile, 'all', 'Save', 'SaveFile', ROIFile, 'MotionCorrect', [fns{i},'.align'], 'Frames', Frames);
        save(ROIFile, 'Data', 'Neuropil', '-mat', '-append');
    end
end
% end