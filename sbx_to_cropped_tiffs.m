function tifffile = sbx_to_cropped_tiffs(sbxfile,opts) %chunksize,alignfile,green_only,empty_red,opto_settings)
% splits up a .sbx file into one or multiple .tifs
% sbxfile a string
if nargin < 2
    opts = [];
end
chunksize = getOr(opts,'chunksize',10000);
alignfile = getOr(opts,'alignfile','');
green_only = getOr(opts,'green_only',false);
empty_red = getOr(opts,'empty_red',false);
opto_correct = getOr(opts,'opto_correct',false);
targetfold = getOr(opts,'targetfold','');
if opto_correct
    opto_settings = opts.opto_settings;
end

if isempty(strfind(sbxfile,'.sbx'))
    sbxfile = [sbxfile '.sbx'];
end
filebase = sbxfile(1:end-4);
% only relevant for running code on big-boi PC
prts = strsplit(filebase,'2P/');
lastpart = prts{end};
matfile_fold = '/home/mossing/modulation/matfiles/'; 
%filebase2 = strrep(filebase,'/home/mossing/modulation/2P/','/home/mossing/modulation1/matfiles/');
filebase2 = [matfile_fold lastpart];
filebase2_no_ot = filebase2;
if strfind(filebase2,matfile_fold)
    strparts = strsplit(filebase2,'/');
    filebase2 = strjoin({strparts{1:end-1} 'ot/' strparts{end}},'/');
    filebase2_no_ot = strjoin({strparts{1:end-1} strparts{end}},'/');
    just_filename = [strparts{end} '.sbx'];
else
    just_filename = sbxfile;
end
global info
load(filebase2_no_ot,'info')
assert(isfield(info,'rect'))
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

temp = squeeze(sbxread(filebase,1,1));

if opto_correct
    if strcmp(opto_settings.type,'square')
        lights_on = opto_settings.lights_on;
        frame = opto_settings.frame;
        while find(diff(frame)<0,1)
            seam = find(diff(frame)<0,1);
            frame(seam+1:end) = frame(seam+1:end)+65536;
        end
        line = opto_settings.line;
        fr_affected = [];
        ln_affected = [];
        for j=1:numel(lights_on)
            if lights_on(j)
                fr = [frame(4*(j-1)+1); frame(4*(j-1)+4)];
                ln = [line(4*(j-1)+1); line(4*(j-1)+4)];
                fr_affected = [fr_affected fr];
                ln_affected = [ln_affected ln];
            end
        end
    end
    if strcmp(opto_settings.type,'exp')
        opto_sbxbase = opto_settings.sbxbase;
        opto_filebase = opto_settings.filebase;
        opto_resultbase = opto_settings.resultbase;
        artifact0 = compute_artifact_decaying_exponential_from_raw(opto_filebase,opto_sbxbase,opto_resultbase);
        artifact = uint16(zeros(size(artifact0,1),info.max_idx+1)); % fill up artifact to be size max_idx+1
        artifact(:,1:size(artifact0,2)) = artifact0;
        artifact = reshape(artifact,size(artifact,1),1,size(artifact,2));
    end
end

options.big = false;
options.append = false;

% how many frames to load into memory at once
ctr = 0;
if alignfile
    alignfile = strrep(sbxfile,'.sbx','.align');
    load(alignfile,'-mat','T');
end
opto_offsets = [];
for i=1:chunksize:info.max_idx
    % just_filename was sbxfile
    tifffile = [targetfold strrep(just_filename,'.sbx',['_t' ddigit(ctr,2) '.tif'])];
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
    if (opto_correct && strcmp(opto_settings.type,'square')) && numel(fr_affected)>0
        opto_offset = estimate_opto_offset(z,fr_affected-startat+1);
        opto_offsets = [opto_offsets; opto_offset];
        if i < 3
            opto_settings.opto_offset = opto_offset;
        else
            opto_offset = opto_settings.opto_offset;
        end
        while numel(fr_affected) > 0 && fr_affected(2,1) < tstartat+newchunksize
            begframe = fr_affected(1,1)-tstartat+1;
            endframe = fr_affected(2,1)-tstartat+1;
            if begframe < 1 % && endframe > 1
                z(:,:,1:endframe-1) = z(:,:,1:endframe-1) - opto_offset;
            else
                z(:,:,begframe+1:endframe-1) = z(:,:,begframe+1:endframe-1) - opto_offset;
                z(ln_affected(1,1):end,:,begframe) = z(ln_affected(1,1):end,:,begframe) - opto_offset;
            end
            z(1:ln_affected(2,1),:,endframe) = z(1:ln_affected(2,1),:,endframe) - opto_offset;
            fr_affected = fr_affected(:,2:end);
            ln_affected = ln_affected(:,2:end);
        end
        if numel(fr_affected) > 0
            begframe = fr_affected(1,1)-tstartat+1;
            endframe = fr_affected(2,1)-tstartat+1;
            if begframe < newchunksize
                z(:,:,begframe+1:end) = z(:,:,begframe+1:end) - opto_offset;
                z(ln_affected(1,1):end,:,begframe) = z(ln_affected(1,1):end,:,begframe) - opto_offset;
            end
        end
    elseif (opto_correct && strcmp(opto_settings.type,'exp')) && ~isempty(artifact)
        z = z - repmat(artifact(:,:,startat+1:startat+newchunksize),1,size(z,2),1);
    end
    
    if twochan
        if ~green_only
            rejig = permute(z(:,rect(1):rect(2),rect(3):rect(4),:),[2,3,1,4]);
        else
            rejig = permute(z(1,rect(1):rect(2),rect(3):rect(4),:),[2,3,1,4]);
        end
        if alignfile
            rejig = motion_correct(rejig,T(tstartat+1:tstartat+newchunksize,:));
        end
        rejig = reshape(rejig,size(rejig,1),size(rejig,2),[]);
        if ~isempty(rejig)
            saveastiff(rejig,tifffile,options);
        end
    else
        rejig = z(rect(1):rect(2),rect(3):rect(4),:);
        if alignfile
            rejig = motion_correct(rejig,T(tstartat+1:tstartat+newchunksize,:));
        end
        if empty_red
            rejig = reshape(rejig,[size(rejig,1) size(rejig,2) 1 size(rejig,3)]);
            redch = zeros(size(rejig));
            rejig = cat(3,rejig,redch);
            rejig = reshape(rejig,size(rejig,1),size(rejig,2),[]);
        end
        if ~isempty(rejig)
            saveastiff(rejig,tifffile,options);
        end
        
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

for d=1:depth,
    t.setTag(tagstruct);
    t.write(data(:, :, d));
    if d~=depth
        t.writeDirectory();
    end
end
t.close();

function opto_offset = estimate_opto_offset(z,fr_affected)
% fr_af = [fr_affected(1,:); fr_affected(1,:) + 5];
% fr_unaf = [fr_affected(1,:)-5; fr_affected(1,:)];
if fr_affected(1,1)>4
    fr_af = [fr_affected(1,:); fr_affected(1,:)];
    fr_unaf = [fr_affected(1,:)-4; fr_affected(1,:)-4];
else
    fr_af = [fr_affected(1,2:end); fr_affected(1,2:end)];
    fr_unaf = [fr_affected(1,2:end)-4; fr_affected(1,2:end)-4];
end
% fr_unaf = reshape([0; fr_affected(1:end-1)'],2,[]);
sz = size(z);
light_on = false(sz(end),1);
light_off = false(sz(end),1);
ind = 1;
while ind <= size(fr_af,2) && fr_af(2,ind) < sz(end)
    %     light_on(fr_af(1,ind)+1:fr_af(2,ind)-1) = 1;
    light_on(fr_af(1,ind)+1) = 1;
    ind = ind+1;
end
ind = 1;
while ind <= size(fr_unaf,2) && fr_unaf(2,ind) < sz(end)
    %     light_off(fr_unaf(1,ind)+1:fr_unaf(2,ind)-1) = 1;
    light_off(fr_unaf(1,ind)+1) = 1;
    ind = ind+1;
end
z_on = z(:,101:end,light_on);
z_off = z(:,101:end,light_off);
opto_offset = mean(z_on(:)) - mean(z_off(:));

function val = getOr(options,fieldname,default)
if isempty(options) || ~isfield(options,fieldname) || isempty(getfield(options,fieldname))
    val = default;
else
    val = getfield(options,fieldname);
end
