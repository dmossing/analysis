% %%
addpath(genpath('/home/mossing/chome/s2p_current'));
addpath(genpath('/home/mossing/chome/adesnal'));
addpath(genpath('/home/mossing/chome/sbatch_scripts'));
addpath(genpath('/home/mossing/chome/downloads/saveastiff')); 
addpath(genpath('~/chome/OASIS_matlab'));addpath(genpath('~/downloads/sort_nat'));
addpath(genpath('/home/mossing/Documents/code/downloads/EvansCode/'))

%%

% %%
% 
% % foldname = '/home/mossing/data/2P/181030/M9826/';
% % foldname = '/home/mossing/data/2P/190102/M10130/';
% % foldname = '/home/mossing/modulation/2P/181213/M10345/';
% % foldname = '/home/mossing/modulation/2P/181214/M10130/';
% % foldname = '/home/mossing/data/2P/181117/M10039/';
% % foldname = '/home/mossing/data/2P/181120/M10039/';
% foldname = '/home/mossing/data/2P/181205/M10130/';
% % foldname = '/home/mossing/data/2P/190107/M10369/';
% % foldname = '/home/mossing/data/2P/180220/M7254/';
% % foldname = '/home/mossing/data/2P/190107/M10036/';
% % foldname = '/media/mossing/backup_1/data/2P/190102/M10130/';
% 

foldname = '/media/mossing/backup_1/data/2P/190202/M10075/';

d = dir([foldname '/M*.mat']); 
fnames = {d(:).name}; 
for i=1:numel(fnames), 

    sbx_to_cropped_tiffs([foldname '/' fnames{i}(1:end-4)],1000); 
%     move_suite2p_tiffs([foldname '/' fnames{i}(1:end-4)],'/home/mossing/data/suite2P/raw/','2P');
    move_suite2p_tiffs([foldname '/' fnames{i}(1:end-4)],'/media/mossing/backup_1/data/suite2P/raw/','2P'); 
end

%%

master_file_local