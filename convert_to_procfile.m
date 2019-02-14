% python to matlab suite2p
% python version saves Fall file, this converts Fall to a format compatible
% with the matlab GUI

function exptsfold = convert_to_procfile(foldname)
%%
if foldname(end)=='/'
    foldname = foldname(1:end-1);
end

rawfold = strrep(foldname,'results','raw');
exptnos = dirnames(rawfold);
exptsfold = [foldname '/' strjoin(exptnos,'_')];
if ~exist(exptsfold,'dir')
    mkdir(exptsfold)
end

s2pfold = [foldname '/suite2p/'];
planefolds = dirnames([s2pfold 'plane*'],s2pfold);
nplanes = length(planefolds);

foldnamesplit = strsplit(foldname,'/');

saveroot = sprintf([exptsfold '/F_%s_%s_plane'],foldnamesplit{end-1},foldnamesplit{end});

%%
for i=1:nplanes
    %%
    dat = load([planefolds{i} '/Fall.mat']);
    iscell = readNPY([planefolds{i} '/iscell.npy']);
    
    %%
    bds = cumsum([0 dat.ops.frames_per_folder]);
    nfolds = numel(dat.ops.frames_per_folder);
    
    %%
    Fcell = cell(1,nfolds);
    FcellNeu = cell(1,nfolds);
    sp = cell(1,nfolds);
    for j=1:nfolds
        Fcell{j} = dat.F(:,bds(j)+1:bds(j+1));
        FcellNeu{j} = dat.Fneu(:,bds(j)+1:bds(j+1));
        sp{j} = dat.spks(:,bds(j)+1:bds(j+1));
    end
    
    ops = dat.ops;
    
    ops.Ly = single(dat.ops.Ly);
    ops.Lx = single(dat.ops.Lx);
    
    ops.mimg1 = ops.meanImg;
    if isfield(ops, 'meanImg_chan2')
        ops.mimgRED = ops.meanImg_chan2;
    end
    clear stat
    flds = fieldnames(dat.stat{1});
    for n = 1:length(dat.stat)
        for j = 1:length(flds)
            stat(n).(flds{j}) = dat.stat{n}.(flds{j});
        end
        stat(n).ipix = int64(ops.Ly)*(stat(n).xpix) + stat(n).ypix + 1;
        stat(n).ipix = stat(n).ipix(:);
        stat(n).mimgProjAbs = 0;
        stat(n).cmpct = stat(n).compact;
        stat(n).aspect_ratio = double(stat(n).aspect_ratio);
        %stat(n) = dat.stat{n};
    end
    ops.yrange = [1:dat.ops.Ly];
    ops.xrange = [1:dat.ops.Lx];
    ops.yrange_crop = dat.ops.yrange(1)+1:dat.ops.yrange(end);
    ops.xrange_crop = dat.ops.xrange(1)+1:dat.ops.xrange(end);
    
    ops.Vcorr = zeros(dat.ops.Ly, dat.ops.Lx);
    ops.Vcorr(ops.yrange_crop, ops.xrange_crop) = dat.ops.Vcorr;
    
    %%
    clear dat
    for n = 1:length(stat)
        stat(n).iscell = iscell(n,1);
        dat.stat(n).iscell = stat(n).iscell;
    end
    
    %%
    
    save([saveroot num2str(i) '.mat'],'-v7.3','Fcell','FcellNeu','ops','sp','stat','dat');
    save([saveroot num2str(i) '_proc.mat'],'-v7.3','Fcell','FcellNeu','ops','sp','stat','dat');
end