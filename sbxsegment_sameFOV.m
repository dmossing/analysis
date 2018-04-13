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

% fns_all = {'M7447_000_003','M7447_200_000'};

% fns_all = {'M7717_000_008','M7717_172_000'};

% fns_all = {'M7307_000_001','M7307_165_000','M7307_165_001','M7307_165_002'};

% fns_all = {'M7199/M7199_000_000', 'M7199/M7199_220_000', 'M7199/M7199_220_001', 'M7199/M7199_220_002', 'M7199/M7199_220_003';
%     'M7307/M7307_000_001', 'M7307/M7307_165_000', 'M7307/M7307_165_001', 'M7307/M7307_165_002', 'M7307/M7307_165_003'};

% fns_all = ... %{'M7199_000_000', 'M7199_220_000', 'M7199_220_001', 'M7199_220_002', 'M7199_220_003';
%     {'../M7307/M7307_000_001', '../M7307/M7307_165_000', '../M7307/M7307_165_001', '../M7307/M7307_165_002', '../M7307/M7307_165_003'};

% fns_all = {'M7199_000_001','M7199_175_000','M7199_175_001'};

% fns_all = {'M7307_000_000','M7307_110_000','M7307_001_001','M7307_110_001'};

% fns_all = {'M8003_000_001','M8003_182_000','M8003_001_002','M8003_182_001'};

% fns_all = {'M8027_000_005','M8027_195_000'};

% fns_all = {'M8027_000_003','M8027_157_000'};

% fns_all = {'M8027_111_002','M8027_190_003','M8027_000_006'};

% fns_all = {'M7199_185_000','M7199_185_001','M7199_000_003'};

% fns_all = {'M8027_000_002','M8027_190_000','M8027_190_001'};

% fns_all = {'M8538_000_001','M8538_155_000'};

% fns_all = {'M8538_155_001_ot_000','M8538_155_001_ot_001','M8538_155_001_ot_002','M8538_155_001_ot_003'};

% fns_all = {'M8538_000_001','M8538_888_000'};

% fns_all = {'M8538_000_003','M8538_100_000'};

% fns_all = {'180125/M8538/M8538_000_003','180125/M8538/M8538_100_000'; ...
%     '180126/M8538/M8538_176_000','180126/M8538/M8538_175_000'; ...
%     '180128/M8538/M8538_000_210','180128/M8538/M8538_210_001'};

% fns_all = {'ot/M7875_999_000_ot_000','ot_ret/M7875_000_002_ot_000'; ...
%     'ot/M7875_999_000_ot_001','ot_ret/M7875_000_002_ot_001'; ...
%     'ot/M7875_999_000_ot_002','ot_ret/M7875_000_002_ot_002';
%     'ot/M7875_999_000_ot_003','ot_ret/M7875_000_002_ot_003'};

% fns_all = {'M7307/ot/M7307_150_001_ot_000';%,'M7307/ot/M7307_150_000_ot_000'; ...
%     'M7307/ot/M7307_150_001_ot_001';%,'M7307/ot/M7307_150_000_ot_001'; ...
%     'M7307/ot/M7307_150_001_ot_002';%,'M7307/ot/M7307_150_000_ot_002';
%     'M7307/ot/M7307_150_001_ot_003'};%,'M7307/ot/M7307_150_000_ot_003'};
% %     'M7254/ot/M7254_125_005_ot_000','M7254/ot/M7254_125_004_ot_000'; ...
% %     'M7254/ot/M7254_125_005_ot_001','M7254/ot/M7254_125_004_ot_001'; ...
% %     'M7254/ot/M7254_125_005_ot_002','M7254/ot/M7254_125_004_ot_002';
% %     'M7254/ot/M7254_125_005_ot_003','M7254/ot/M7254_125_004_ot_003'};

% fns_all = {'M7307/ot/M7307_120_002_ot_000','M7307/ot/M7307_120_003_ot_000'; ...
%     'M7307/ot/M7307_120_002_ot_001','M7307/ot/M7307_120_003_ot_001'; ...
%     'M7307/ot/M7307_120_002_ot_002','M7307/ot/M7307_120_003_ot_002'; ...
%     'M7307/ot/M7307_120_002_ot_003','M7307/ot/M7307_120_003_ot_003'};

% fns_all = {'M7874_140_002_ot_000','M7874_140_003_ot_000', ...
% 'M7874_140_004_ot_000'};

% fns_all = {'M7194_999_000_ot_000';'M7194_999_000_ot_001'; ...
% 'M7194_999_000_ot_002';'M7194_999_000_ot_003'};
% fns_all = {'M7955_000_000_ot_000','M7955_000_001_ot_000', ...
% 'M7955_000_002_ot_000'};
% fns_all = {'M7955_000_000_ot_000','M7955_000_001_ot_000'}; %, ...
% fns_all = {'M7955_000_002_ot_000','M7955_000_003_ot_000'};%
% fns_all = {'M7955_000_002_ot_001','M7955_000_003_ot_001'};%
% fns_all = {'M7955_000_002_ot_002','M7955_000_003_ot_002'; ...
%     'M7955_000_002_ot_003','M7955_000_003_ot_003'};
% fns_all = {'M8570_150_002_ot_001','M8570_151_001_ot_001','M8570_151_002_ot_001'; ...
%     'M8570_150_002_ot_002','M8570_151_001_ot_002','M8570_151_002_ot_002'; ...
%     'M8570_150_002_ot_003','M8570_151_001_ot_003','M8570_151_002_ot_003'};

% fns_all = {'M7955_000_001_ot_000', 'M7955_000_001_ot_001',...
%     'M7955_000_001_ot_002', 'M7955_000_001_ot_003'};

% fns_all = {...%{'M7955_150_000_ot_000','M7955_150_001_ot_000','M7955_150_002_ot_000'};
%      'M7955_150_000_ot_001','M7955_150_001_ot_001','M7955_150_002_ot_001';
%      'M7955_150_000_ot_002','M7955_150_001_ot_002','M7955_150_002_ot_002';
%      'M7955_150_000_ot_003','M7955_150_001_ot_003','M7955_150_002_ot_003'};

fns_all = {'M8570_000_001_ot_001';'M8570_000_001_ot_002';'M8570_000_001_ot_003'};


% fns_all = dir('*.segment');
% fns_all = {fns_all(:).name};
% for i=1:numel(fns_all)
%     fns_all{i} = strrep(fns_all{i},'.segment','');
% end
% fns_all = fns_all(:);

%%

j = 3;

fns = fns_all(j,:);

N = numel(fns);

i = 3;

%%
% for i=startat:N
load([fns{i} '.mat'],'info')
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
addpath(genpath('/home/mossing/Documents/code/adesnal/'))
for j=1:size(fns_all,1)
    % extract dFoF signals
    fns = fns_all(j,:);
    N = numel(fns);
    for i=1:N
        ROIFile = [fns{i}, '.rois'];
        ROIdata = sbxDistribute(fns{i}, 'Save', 'SaveFile', ROIFile); % intialize struct
        createMasks(ROIdata, 'Save', 'SaveFile', ROIFile); % create ROI masks
%         config = load2PConfig([fns{i}, '.sbx']);
%         load([fns{i} '.align'],'-mat','T')
        if ~isempty(strfind(fns{i},'_depth'))
            Depth = str2num(fns{i}(end));
            sbxfname = strsplit(fns{i},'_depth');d
            sbxfname = [sbxfname{1} '.sbx'];
        else
            sbxfname = [fns{i} '.sbx'];
            Depth = 1;
        end
        Frames = idDepth(sbxfname,[],'Depth',Depth)';
%         frameno = size(T,1);
%         [~, Data, Neuropil, ~] = extractSignals([fns{i},'.sbx'], ROIFile, 'all', 'Save', 'SaveFile', ROIFile, 'MotionCorrect', [fns{i},'.align'], 'Frames', Frames);
        [~, Data, Neuropil, ~] = extractSignals(sbxfname, ROIFile, 'all', 'Save', 'SaveFile', ROIFile, 'MotionCorrect', [fns{i},'.align']); %, 'Frames', Frames);
        save(ROIFile, 'Data', 'Neuropil', '-mat', '-append');
    end
end
% end