function write_meadowlark_lut(lut,filename)
% lut an array of 256 12-bit integer values
fid = fopen(filename,'wt');
for i=1:numel(lut)
    if i~=numel(lut)
        fprintf(fid,'%u\t%u\n',i-1,lut(i));
    else
        fprintf(fid,'%u\t%u',i-1,lut(i));
    end
end
fclose(fid)