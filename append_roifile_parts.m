function [output,iplane] = append_roifile_parts(roifile,fieldname,transpose)
if nargin < 3
    transpose = false;
end
output = cell(size(roifile));
iplane = cell(size(roifile));
for i=1:numel(roifile)
    output{i} = getfield(roifile{i},fieldname);
    if transpose
        output{i} = output{i}';
    end
end
for i=1:numel(roifile)
    iplane{i} = i*ones(size(output{i},1),1);
end
output = cell2mat(output);
iplane = cell2mat(iplane);
if transpose
    output = output';
end