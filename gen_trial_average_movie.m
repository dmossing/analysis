function gen_trial_average_movie(fnames,stimp,foldbase)
% stimfname = '/home/mossing/excitation/visual_stim/180216/M7307/M7307_120_003.mat';
% fnames = dir('*_003_ot_*.sbx')
Nplane = numel(fnames);
% fnames = {fnames(:).name};
frame = cell(Nplane,1);
transl = cell(Nplane,1);
% stimp = cell(numel(fnames),1);
% load(stimfname,'result');
% stimp = result.stimParams;
stimduration = 0;
rect = cell(Nplane,1);
% figure out the right frames for each movie
for z=1:Nplane
    sbxfile = fnames{z};
    matfile = strrep(sbxfile,'.sbx','.mat');
    alignfile = strrep(sbxfile,'.sbx','.align');
    load(matfile,'info')
    if isfield(info,'rect')
        rect{z} = info.rect;
    else
        rect{z} = [];
    end
    load(alignfile,'-mat','T')
    try
        frame{z} = reshape(info.frame(info.event_id==1),2,[])';
    catch
        nbefore = 2;
        nafter = 2;
        % if # of frames is odd
        temp = info.frame(info.event_id==1);
        temp = temp(2:end-10);
        frame{z} = reshape(temp,2,[])';
        frame{z}(:,1) = frame{z}(:,1)-nbefore;
        frame{z}(:,2) = frame{z}(:,2)+nafter;
%         doubled = [temp(2:end-1) temp(2:end-1)]';
%         temp = [temp(1); doubled(:); temp(end)];
%         frame{z} = reshape(temp,2,[])';
%         frame{z}(:,1) = frame{z}(:,1)-nbefore;
%         frame{z}(:,2) = frame{z}(:,2)+nafter;
    end
    transl{z} = T;
    stimduration = max(stimduration,1+min(diff(frame{z},[],2)));
end
% get the unique ones, to decide which clip to attribute to which stim
% [conds,~,cind] = unique(stimp(2:end,:)','rows');
[conds,~,cind] = unique(stimp','rows');
Ncond = size(conds,1);
cond_avg = zeros(512,796*Nplane,stimduration);
for i=1:Ncond
    i,
    %     cond_avg(:,:,:,:,i) = zeros(512,796,Nplane,stimduration);
    trials = find(cind==i);
    Ntrial = numel(trials);
    for z=1:Nplane
        toload = [];
        for r=1:Ntrial
            startat = frame{z}(trials(r),1);
            endat = startat+stimduration-1;
            toload = [toload startat:endat];
        end
        newdata = squeeze(double(load2P(fnames{z},'Frames',toload)));
        newdata = motion_correct(newdata,T(toload,:));
        if ~isempty(rect{z}) % set area outside crop rectangle to zero if present
            aux = uint16(zeros(size(newdata)));
            aux(rect{z}(1):rect{z}(2),rect{z}(3):rect{z}(4),:) ...
                = newdata(rect{z}(1):rect{z}(2),rect{z}(3):rect{z}(4),:);
            newdata = aux;
        end
        newdata = reshape(newdata,size(newdata,1),size(newdata,2),stimduration,[]);
        cond_avg(:,796*(z-1)+1:796*z,1:size(newdata,3)) = mean(newdata,4)/double(intmax('uint16'));
    end
    foldname = [foldbase '/cond' ddigit(i-1,3) '/'];
    mkdir(foldname)
    for t=1:stimduration
        imwrite(cond_avg(:,:,t),[foldname ddigit(t-1,3) '.tif'])
    end
end