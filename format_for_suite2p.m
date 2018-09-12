function format_for_suite2p(foldname,varargin)

p = what(foldname);
foldname = p.path;

defaults = {{'endings',''},...
    {'chunk_size',1000},...
    {'target_fold','/global/scratch/mossing/2Pdata/suite2P/raw/'}};
[endings,chunk_size,target_fold] = parse_args(varargin,defaults);

% if there are no endings specified do everything
if isempty(endings)
    d = dir([foldname '/*.sbx']);
    fnames = {d(:).name};
    endings = cell(size(fnames));
    for i=1:numel(endings)
        a = strsplit(fnames{i},'.sbx');
        b = strsplit(a{1},'_');
        endings{i} = b{end};
    end
else
    % otherwise, only the files specified
    if ~iscell(endings)
        endings = {endings};
    end
    fnames = cell(size(endings));
    for i=1:numel(endings)
        d = dir([foldname '/*_' endings{i} '.sbx']);
        fnames{i} = d.name;
    end
end

% identify the animal id and date
parts = strsplit(foldname,filesep);
animalid = parts{end};
expt_date = parts{end-1};

% create a folder with the correct animalid and date
mkdir([target_fold '/' animalid])
mkdir([target_fold '/' animalid '/' expt_date])


for i=1:numel(fnames)
    % split each file into cropped tiffs, (chunk_size) images long
    thisname = [foldname '/' fnames{i}];
    sbx_to_cropped_tiffs(thisname,chunk_size)
    
    % generate target subfolder
    stripped = num2str(str2num(endings{i}));
    target_subfold = [target_fold '/' animalid '/' expt_date '/' stripped];
    mkdir(target_subfold)
    
    % copy the result into the target folder
    d = dir([foldname '/*_t*.tiff']);
    chunk_names = {d(:).name};
    for j=1:numel(chunk_names)
        movefile([foldname '/' chunk_names{j}],[target_subfold '/' chunk_names{j}(1:end-1)])
    end
end