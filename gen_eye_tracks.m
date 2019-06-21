function gen_eye_tracks(foldname) % [ctrstar,areastar] =
old_fold = pwd;
cd(foldname)
%%
fnames = dirnames('eye_tracking_*.mat');
%%
N = numel(fnames);
for i=1:N
    fnames{i}
    load(fnames{i},'ctr','area2','ctrstar','areastar')
    if ~exist('ctrstar','var') || ~exist('areastar','var')
        %     subplot(2,N,i)
        %     hold on
        %     plot(ctr2)
        %     plot(ctr)
        %     hold off
        %     subplot(2,N,N+i)
        %     hold on
        %     plot(area2)
        %     plot(area)
        %     hold off
        % end
        %%
        norm_by = 1e3;
        tv_penalty = @(track) nansum(abs(diff(track)))/norm_by;
        tsmooth_penalty = @(track) nansum(abs(diff(track)).^2)/norm_by;
        error_penalty = @(track,meas) nansum(abs(track-meas).^2)/norm_by;
        %%
        ltv = 1e0;
        ltsmooth = 1;
        rgs = split_into_chunks(numel(area2),50,10);
        ctrstar = zeros([size(rgs) 2]);
        areastar = zeros(size(rgs));
        for iwindow = 1:size(rgs,1)
            nonzero = rgs(iwindow,:)>0;
            bds = rgs(iwindow,nonzero);
            for idim = 1:2
                track0 = ctr(bds,idim);
                track0(isnan(track0)) = nanmean(track0);
                cost_track = @(track) error_penalty(track,ctr(bds,idim)) + ltv*tv_penalty(track);
                ctrstar(iwindow,nonzero,idim) = fminunc(cost_track,track0); %,'Display','iter');
            end
            area0 = area2(bds);
            area0(isnan(area0)) = nanmean(area0);
            area_track = @(track) error_penalty(track,area2(bds)) + ltsmooth*tsmooth_penalty(track);
            areastar(iwindow,nonzero) = fminunc(area_track,area0);
        end
        %     last_chunk = rgs(end,rgs(end,:)>0);
        %     n_chunk = numel(last_chunk);
        %     bds = last_chunk;
        %     for idim = 1:2
        %         track0 = ctr(bds,idim);
        %         track0(isnan(track0)) = nanmean(track0);
        %         cost_track = @(track) error_penalty(track,ctr(bds,idim)) + ltv*tv_penalty(track);
        %         ctrstar(end,1:n_chunk,idim) = fminunc(cost_track,track0);
        %     end
        %     area0 = area2(bds);
        %     area0(isnan(area0)) = nanmean(area0);
        %     area_track = @(track) error_penalty(track,area2(bds)) + ltsmooth*tsmooth_penalty(track);
        %     areastar(end,1:n_chunk) = fminunc(area_track,area0);
        temp = zeros(size(ctr));
        for idim=1:2
            temp(:,idim) = stitch_together(ctrstar(:,:,idim),rgs);
        end
        ctrstar = temp;
        areastar = stitch_together(areastar,rgs)';
        save(fnames{i},'ctrstar','areastar','-append');
    end
    clear ctrstar areastar
end

% %%
% figure
% hold on
% plot(abs(track0-trackstar).^2)
% plot(abs(diff(track0)))
% plot(abs(diff(trackstar)))
% hold off
% %%
% figure
% hold on
% plot(areastar)
% plot(area0)
% hold off
%
% %%
% tv_penalty(ctr2(:,1))
%%
cd(old_fold)


function rgs = split_into_chunks(len,chunksize,overlap)
starts = 1:chunksize-overlap:len;
ends = starts + chunksize-1;
ends = min(ends,len);
n = numel(starts);
rgs = zeros(n,chunksize);
for i=1:n
    chunk = starts(i):ends(i);
    rgs(i,1:numel(chunk)) = chunk;
end

function stitched = stitch_together(chunks,rgs)
for i=1:max(rgs(:))
    stitched(i) = mean(chunks(rgs==i));
end