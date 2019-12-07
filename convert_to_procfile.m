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
    red_saved = exist([planefolds{i} '/redcell.npy'],'file');
    if red_saved
        redcell = readNPY([planefolds{i} '/redcell.npy']);
        redratio = redcell(:,2);
    end
    %%
    framesper = zeros(size(dat.ops.frames_per_folder));
    whichfold = zeros([size(dat.ops.filelist,1) 1]);
    for j=1:size(dat.ops.filelist,1) % take list of files, and find which subfold each is in
        ss = strsplit(dat.ops.filelist(j,:),'/');
        whichfold(j) = str2num(ss{end-1});
    end
    [~,ia,~] = unique(whichfold); % take the unique subfolds, and where the first of each occurs in the filelist
    [~,~,ic] = unique(ia); % return the order in which the folders appear in filelist
    framesper = dat.ops.frames_per_folder; %(ic); % trying other way round!
     % for some reason, this was req'd on 3/20 data ^
    %%
    bds = cumsum([0 framesper]);
    nfolds = numel(framesper);
    
    %%
    Fcell = cell(1,nfolds);
    FcellNeu = cell(1,nfolds);
    sp = cell(1,nfolds);
    for j=1:nfolds
        Fcell{ic(j)} = dat.F(:,bds(j)+1:bds(j+1)); % trying other way round!
        FcellNeu{ic(j)} = dat.Fneu(:,bds(j)+1:bds(j+1)); % trying other way round!
        sp{ic(j)} = dat.spks(:,bds(j)+1:bds(j+1)); % trying other way round!
    end
    
%     Fcell(ic) = Fcell; % for some reason, this was req'd on 3/20 data
%     FcellNeu(ic) = FcellNeu;
%     sp(ic) = sp;
    
    ops = dat.ops;
    
    vars_of_interest = {'meanImg','meanImg_chan2','meanImg_chan2_corrected','meanImgE'};
    for ivar=1:numel(vars_of_interest)
        this_var = vars_of_interest{ivar};
        var_fname = sprintf([planefolds{i} '/%s.npy'],this_var);
        try
            var_val = readNPY(var_fname);
            ops = setfield(ops,this_var,var_val);
        catch
            disp(this_var)
        end
    end
    
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
    if red_saved
        save([saveroot num2str(i) '.mat'],'-v7.3','Fcell','FcellNeu','ops','sp','stat','dat','redratio');
        save([saveroot num2str(i) '_proc.mat'],'-v7.3','Fcell','FcellNeu','ops','sp','stat','dat','redratio');
    else
        save([saveroot num2str(i) '.mat'],'-v7.3','Fcell','FcellNeu','ops','sp','stat','dat');
        save([saveroot num2str(i) '_proc.mat'],'-v7.3','Fcell','FcellNeu','ops','sp','stat','dat');
    end
end
