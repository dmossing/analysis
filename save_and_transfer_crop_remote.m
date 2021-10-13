function save_and_transfer_crop_remote(foldname,sbxbase,copymat,rect)
% if rect is populated, then it is taken to be the pre-specified crop.
% Otherwise, the user is asked to define a crop using the GUI.
% imcrop = im(rect(1):rect(2),rect(3):rect(4))
if nargin < 3
    copymat = true;
end
if foldname(end)~='/'
    foldname = [foldname '/'];
end
if sbxbase(end)~='/'
    sbxbase = [sbxbase '/'];
end
old_fold = pwd;
cd(foldname)
this_fold = pwd;
fileparts = strsplit(this_fold,'/');
date = fileparts{end-1};
animalid = fileparts{end};
sbxfold = strjoin({sbxbase,date,animalid},'/');
if copymat
    fnames = dirnames([sbxfold '/M*.mat'],[sbxfold '/']);
    for i=1:numel(fnames)
        copyfile(fnames{i},foldname)
    end
end
fnames = dirnames([foldname '*.mat'],foldname);
for i=1:numel(fnames)
    fnames{i} = fnames{i}(1:end-4);
end
bidi = zeros(size(fnames))>0;
for i=1:numel(fnames)
    matfile = load([fnames{i} '.mat'],'info');
    % look only at bidirectional files
    bidi(i) = isfield(matfile,'info') && ~matfile.info.scanmode;
end
fnames = fnames(bidi);
if nargin < 4
    save_crop_for_alignment(fnames(1),sbxfold);
else
    save_crop_for_alignment(fnames(1),sbxfold,rect);
end
for i=2:numel(fnames)
    transfer_crop(fnames(1),fnames(i))
end
cd(old_fold)