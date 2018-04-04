function header = parseSI5Header(input)

sIndex=strfind(input,'scanimage');

for i=1:size(sIndex,2)-1
    
    [field, remainder ] = strtok(input(sIndex(i):sIndex(i+1)-1));
    [tk, val] = strtok(remainder);
    field= strrep(field,'.','_'); %remove . from names if present
    val = val(2:end-1);
    try
        eval(['header.scanimage.(field(15:end))=' val ';']);
    catch %catch weird variables and translate to strings <>
        header.scanimage.(field(15:end))=val;
    end
end
    [field, remainder ] = strtok(input(sIndex(i+1):end));
    [tk, val] = strtok(remainder);
    field= strrep(field,'.','_'); %remove . from names if present
    val2 = strtok(val);
     try
        eval(['header.scanimage.(field(15:end))=' val2 ';']);
     catch
         header.scanimage.(field(15:end))=val;
     end
     
    