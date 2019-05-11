% %%
addpath(genpath('/home/mossing/Documents/code/s2p_current'));
addpath(genpath('/home/mossing/Documents/code/adesnal'));
addpath(genpath('/home/mossing/Documents/code/sbatch_scripts'));
addpath(genpath('/home/mossing/Documents/code/downloads/saveastiff')); 
addpath(genpath('~/Documents/code/OASIS_matlab'));addpath(genpath('~/downloads/sort_nat'));
addpath(genpath('/home/mossing/Documents/code/downloads/EvansCode/'))
addpath(genpath('/home/mossing/Documents/code/downloads/npy-matlab-master'))

%%

% FIRST RUN save_and_transfer_crop(foldname)

% foldname = '/home/mossing/data/2P/181030/M9826/';
% foldname = '/home/mossing/data/2P/190102/M10130/';
% foldname = '/home/mossing/modulation/2P/181213/M10345/';
% foldname = '/home/mossing/modulation/2P/181214/M10130/';
% foldname = '/home/mossing/data/2P/181117/M10039/';
% foldname = '/home/mossing/data/2P/181120/M010039/';
% foldname = '/home/mossing/data/2P/181205/M10130/';

% foldname = '/media/mossing/data_ssd/data/2P/green_only/190221/M9835/';
% foldname = '/media/mossing/data_ssd/data/2P/190225/M10344/';
% foldname = '/media/mossing/data_ssd/data/2P/190301/M9835/';
% foldname = '/media/mossing/data_ssd/data/2P/190304/M10077/';
% foldname = '/media/mossing/data_ssd/data/2P/190307/M9835/';
% foldname = '/media/mossing/data_ssd/data/2P/190314/M10388/';
% foldname = '/media/mossing/data_ssd/data/2P/190318/M10338/';
% foldname = '/media/mossing/data_ssd/data/2P/190320/M10365/';
% foldname = '/media/mossing/data_ssd/data/2P/190321/M0090/';
% foldname = '/media/mossing/data_ssd/data/2P/190326/M0090/';
% foldname = '/media/mossing/data_ssd/data/2P/190107/M10036/';
% foldname = '/media/mossing/data_ssd/data/2P/190407/M10368/';
% foldname = '190411/M0002/';
% foldname = '190318/M10338/';
% foldname = '190320/M10365/';
% foldname = '190410/M10368/';
% foldname = '190408/M0002/';
% foldname = '190407/M10368/';
% foldname = '190501/M0094/';
foldname = '190503/M0002/';


data_foldbase = '/media/mossing/backup_0/data/2P/';
result_foldbase = '/home/mossing/modulation/visual_stim/';

targetfold = '/media/mossing/backup_0/data/suite2P/raw/';

d = dir([data_foldbase foldname '/M*.mat']); 
fnames = {d(:).name}; 
for i=1:4
    sbx_to_cropped_tiffs([data_foldbase foldname '/' fnames{i}(1:end-4)],1000,'',1); % 1 for green only; otherwise 0
    move_suite2p_tiffs([data_foldbase foldname '/' fnames{i}(1:end-4)],targetfold,'2P'); 
end
% for i=2:3
% %     sbx_to_cropped_tiffs([foldname '/' fnames{i}(1:end-4)],1000,'',1); % 1 for green only; otherwise 0
%     load([data_foldbase foldname '/' fnames{i}],'info')
%     load([result_foldbase foldname '/' fnames{i}],'result')
%     frame_offset = 1; % first this many trigger frames are incorrect
%     opto_settings = gen_opto_settings(info,result,frame_offset);
%     sbx_to_cropped_tiffs([data_foldbase foldname '/' fnames{i}(1:end-4)],1000,'',1,[],opto_settings); % 1 for green only; otherwise 0
%     move_suite2p_tiffs([data_foldbase foldname '/' fnames{i}(1:end-4)],targetfold,'2P');
% end
% for i=3
%     sbx_to_cropped_tiffs([data_foldbase foldname '/' fnames{i}(1:end-4)],1000,'',1); % 1 for green only; otherwise 0
%     move_suite2p_tiffs([data_foldbase foldname '/' fnames{i}(1:end-4)],targetfold,'2P'); 
% end
% foldname = '/media/mossing/data_ssd/data/2P/190221/M9836/';
% 
% targetfold = '/media/mossing/data_ssd/data/suite2P/raw/';
% 
% d = dir([foldname '/M*.mat']); 
% fnames = {d(:).name}; 
% for i=1:numel(fnames), 
%     sbx_to_cropped_tiffs([foldname '/' fnames{i}(1:end-4)],1000); 
%     move_suite2p_tiffs([foldname '/' fnames{i}(1:end-4)],targetfold,'2P'); 
% end


% master_file_local
