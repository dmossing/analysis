function sbxsplitforalignment(fn)
sbxsplit(fn)
d = dir([fn '_ot_*.sbx']);
depthno = numel(d);
load(fn,'info')
info = rmfield(info,'otparam');
for i=1:depthno
    save(strrep(d(i).name,'.sbx','.mat'),'info')
end