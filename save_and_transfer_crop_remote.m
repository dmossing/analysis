function save_and_transfer_crop_remote(foldname,sbxbase)
old_fold = pwd;
cd(foldname)
this_fold = pwd;
fileparts = strsplit(this_fold,'/');
date = fileparts{end-1};
animalid = fileparts{end};
if foldname(end)~='/'
    foldname = [foldname '/'];
end
if sbxbase(end)~='/'
    sbxbase = [sbxbase '/'];
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
sbxfold = strjoin({sbxbase,date,animalid},'/');
save_crop_for_alignment(fnames(1),sbxfold);
for i=2:numel(fnames)
    transfer_crop(fnames(1),fnames(i))
end
cd(old_fold)