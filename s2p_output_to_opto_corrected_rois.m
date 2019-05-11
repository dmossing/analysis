function s2p_output_to_opto_corrected_rois(foldname,varargin)

p = inputParser;

p.addParameter('datafold','/media/mossing/backup_0/data/2P/');

p.parse(varargin{:});

datafold = p.Results.datafold;

s = what(foldname);
foldname = s.path;

%% copy .mat files to the ot/ folder

path_parts = strsplit(foldname,'/');
file_endings = strsplit(path_parts{end},'_');
datestr = path_parts{end-1};
animalid = path_parts{end-2};
targetfold = sprintf('%s/%s/%s/',datafold,datestr,animalid);
for i=1:numel(file_endings)
    copyfile([targetfold '*' file_endings{i} '.mat'],[targetfold 'ot/'])
end
foldname_short = sprintf('%s/%s/',datestr,animalid);

%% create .rois files in the correct location
convert_npy_to_rois(foldname,datafold)

%% perform optogenetic artifact correction, if necessary
function_run_1p_opto_correction(foldname_short,'datafold',p.Results.datafold)
