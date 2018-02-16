function sbxsplitforalignment(fn,foldname)
if nargin < 2
    foldname = 'ot/';
end
sbxsplit(fn)
d = dir([fn '_ot_*.sbx']);
depthno = numel(d);
load(fn,'info')
try
    info = rmfield(info,'otparam');
end
if ~exist(foldname,'dir')
    mkdir(foldname)
end
for i=1:depthno
    sbxfile = d(i).name;
    matfile = [foldname '/' strrep(d(i).name,'.sbx','.mat')];
    save(matfile,'info')
    movefile(sbxfile, [foldname '/' sbxfile])
end
