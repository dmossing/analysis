function tifffile = sbx_to_cropped_tiffs(sbxfile,chunksize,alignfile)
% splits up a .sbx file into one or multiple .tifs
% sbxfile a string
if nargin < 2 || isempty(chunksize)
    chunksize = 10000; % number of frames per split up .tif
end
if nargin < 3 || isempty(alignfile)
    alignfile = '';
end
if isempty(strfind(sbxfile,'.sbx'))
    sbxfile = [sbxfile '.sbx'];
end
filebase = sbxfile(1:end-4);
global info
load(filebase,'info')
if isfield(info,'rect')
    rect = info.rect; % information necessary if image needs to be cropped
else
    rect = [1 info.sz(1) 1 info.sz(2)];
end
if info.channels == 1
    twochan = true;
else
    twochan = false;
end
%temp = sbxreadpacked(filebase,1,1);
temp = squeeze(sbxread(filebase,1,1));
% [~, ~, rect] = crop(sbxreadpacked(filebase,1,1), true);
% %         rect = round([rect(3),rect(3)+rect(4),rect(1),rect(1)+rect(2)]);
% rect = [rect(2),rect(2)+rect(4),rect(1),rect(1)+rect(3)];
% rect(1) = max(rect(1),1);
% rect(3) = max(rect(3),1);
% rect(2) = min(rect(2),info.sz(1));
% rect(4) = min(rect(4),info.sz(2));
% if ~mod(rect(2)-rect(1),2)
%     rect(2) = rect(2)-1;
% end
% if ~mod(rect(4)-rect(3),2)
%     rect(4) = rect(4)-1;
% end
options.big = false;
options.append = false;

% how many frames to load into memory at once
ctr = 0;
if alignfile
    alignfile = strrep(sbxfile,'.sbx','.align');
    load(alignfile,'-mat','T');
end
for i=(1+twochan):chunksize:info.max_idx
    tifffile = strrep(sbxfile,'.sbx',['_t' ddigit(ctr,2) '.tif']);
    if exist(tifffile,'file')
        delete(tifffile)
        disp('deleted old vsn')
    end
    i
    startat = i;
    tstartat = startat/(1+twochan);
    if startat+chunksize>=info.max_idx
        % stop yourself from going overboard, and keep the same # in each plane
        try
            newchunksize = info.max_idx-startat-rem(info.max_idx-1,info.otparam(3));
        catch
            newchunksize = info.max_idx-startat;
        end
    else
        newchunksize = chunksize;
    end
    z = squeeze(sbxread(filebase,startat,newchunksize));
    %z = sbxreadpacked(filebase,startat,newchunksize);
    if twochan
        rejig = permute(z(:,rect(1):rect(2),rect(3):rect(4),:),[2,3,1,4]);
        rejig = motion_correct(rejig,T(tstartat+1:tstartat+newchunksize,:));
        rejig = reshape(rejig,size(rejig,1),size(rejig,2),[]);
        %         mysaveastiff(rejig,tifffile,i==1);
        saveastiff(rejig,tifffile,options); 
        
    else
        rejig = z(rect(1):rect(2),rect(3):rect(4),:);
        rejig = motion_correct(rejig,T(tstartat+1:tstartat+newchunksize,:));
        %         mysaveastiff(z(rect(1):rect(2),rect(3):rect(4),:),tifffile,i==1);
        saveastiff(rejig,tifffile,options); 
        
    end
    ctr = ctr+1;
end

function mysaveastiff(data,tifffile,isfirst)
depth = size(data,3);
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.ImageLength = size(data,1);
tagstruct.ImageWidth = size(data,2);
tagstruct.SamplesPerPixel = 1;
tagstruct.RowsPerStrip = tagstruct.ImageLength;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tagstruct.Compression = Tiff.Compression.None;
tagstruct.BitsPerSample = 16;
% if isfirst
t = Tiff(tifffile,'w8');
% else
%     t = Tiff(tifffile,'a');
% end
for d=1:depth,
    t.setTag(tagstruct);
    t.write(data(:, :, d));
    if d~=depth
        t.writeDirectory();
    end
end
t.close();
