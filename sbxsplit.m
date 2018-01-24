function sbxsplit(fn)

global info

% splits a volumetric optotune dataset into N different ones

load(fn);

if info.volscan == 0
    fprintf(2,'File is not a volumetric scan');
    return;
end

try
    nslices = info.otparam(3);  % get the number of slices
catch
    nslices = 2;
end

d = dir(sprintf('%s.sbx',fn));   % get the number of bytes in file
nb = d(1).bytes;
bytesperslice = nb/nslices;

fid = fopen(d(1).name,'r');
fids = zeros(1,nslices);
sname = cell(1,nslices);

for i = 1:nslices
    sname{i} = sprintf('%s_ot_%03d.sbx',fn,i-1);
    [~,~] = system(sprintf('fsutil file createnew %s %d',sname{i},bytesperslice));   %allocate space
    fids(i) = fopen(sname{i},'w');
end

z = sbxread(fn,0,1);

% split file

for n=0:info.max_idx   
    s = mod(n,nslices); % which slice?
    buf = fread(fid,info.nsamples/2,'uint16=>uint16');
    fwrite(fids(s+1),buf,'uint16');
%     subplot(2,2,s+1)
%     imshow(reshape(buf,512,796))
end

drawnow('update');

% close files
fclose(fid);

for s = 1:nslices
    fclose(fids(s));
end

% create matlab files
for i = 1:nslices
    matname{i} = sprintf('%s_ot_%03d.mat',fn,i-1);
    [~,~] = system(sprintf('copy %s %s',[fn '.mat'],matname{i}));  
end


