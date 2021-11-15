%%
figure
for irun=1:2
    for ilight=1:2
        subplot(2,2,2*(ilight-1)+irun)
        if irun==1
            plot(sizes,squeeze(nanmean(runcondfr(:,ilight,:),1)))
        elseif irun==2
            plot(sizes,squeeze(nanmean(stillcondfr(:,ilight,:),1)))
        end
    end
end
%%
figure
for isize=1:3
    for icontrast=1:5
        subplot(3,5,5*(isize-1)+icontrast)
        ilight1 = 2;
        ilight2 = 1;
        scatter(runcondfr(:,ilight1,isize),stillcondfr(:,ilight2,isize),5)
        hold on
        plot([0 40],[0 40])
        hold off
    end
end
%%
figure
for isize=1:3
    for icontrast=1:5
        subplot(3,5,5*(isize-1)+icontrast)
        ilight1 = 1;
        ilight2 = 1;
        scatter(runcondfr(:,ilight1,isize),stillcondfr(:,ilight2,isize),5)
        hold on
        plot([0 40],[0 40])
        hold off
    end
end
%%
subplot(1,2,1)
imagesc(squeeze(nanmean(runcondfr(pfsv,2,:)-runcondfr(pfsv,1,:))))
subplot(1,2,2)
imagesc(squeeze(nanmean(runcondfr(prsv,2,:)-runcondfr(prsv,1,:))))
%%
rfs = {stillcondfr(pfsv,:,:),runcondfr(pfsv,:,:)};
rrs = {stillcondfr(prsv,:,:),runcondfr(prsv,:,:)};
dfs = cell(2,1);
drs = cell(2,1);
for irun=1:2
    dfs{irun} = squeeze(nanmean(rfs{irun}(:,2,:)-rfs{irun}(:,1,:)));
    drs{irun} = squeeze(nanmean(rrs{irun}(:,2,:)-rrs{irun}(:,1,:)));
end
% hold on
% for isize=1:3
% %     for icontrast=1:5
%     plot(drs(isize,:),dfs(isize,:))
% %     end
% end
% hold off
% save('r_ephys.mat','rrs','rfs')
%%
scatter(rrs(:,1,1,1),rfs(:,1,1,1))
%%
figure;
for irun=1:2
    subplot(1,2,irun)
    hold on; bar(dfs{irun}(:,end)); bar(drs{irun}(:,end)); hold off
    xticks(1:size(rrs{1},3))
    xticklabels(gca,sizes)
    legend({'FS','RS'})
    xlabel('size ( ^o )')
    ylabel('delta spikes (Hz)')
    title('VIP Halo change in spike rate, 100% contrast')
end

% subplot(1,2,2)
% hold on; bar(dfs(:,end)); bar(drs(:,end)); hold off
% xticks(1:3)
% xticklabels(gca,sizes)
% legend({'FS','RS'})
% xlabel('size ( ^o )')
% ylabel('delta spikes (Hz)')
% title('VIP Halo change in spike rate, 100% contrast')
%%
figure;
hold on
for isize=1:3
    scatter(dfs(isize,end),drs(isize,end))
end
hold off
%%
figure
irun = 1;
ilight1 = 1;
ilight2 = 2;
for iplot=1
    subplot(1,2,iplot)
    hold on
    for ilight=1:2
        plot(squeeze(nanmean(rfs{irun}(:,ilight,:,end))))
    end
    hold off
end
for iplot=2
    subplot(1,2,iplot)
    hold on
    for ilight=1:2
        plot(squeeze(nanmean(rrs{irun}(:,ilight,:,end))))
    end
    hold off
end
%%
[~,prefsize] = max(squeeze(nanmean(rrs{irun}(:,:,:,end),2)),[],2);

%%
for irun=1:2
    rrs_pref_aligned{irun} = align_by_prefsize(rrs{irun});
    rfs_pref_aligned{irun} = align_by_prefsize(rfs{irun});
end

%%
for irun=1:2
    for icontrast=1

        figure
%         subplot(1,2,1)
        hold on
%         for ilight=1:2
        diffy = diff(squeeze(nanmean(rrs_pref_aligned{irun}(:,:,:,icontrast))));
        diffyerr = diff(squeeze(nanstd(rrs_pref_aligned{irun}(:,:,:,end))));
        n_non_nan = sum(~isnan(squeeze(rrs_pref_aligned{irun}(:,ilight,:,icontrast))));
        errorbar(1:5,diffy,diffyerr./sqrt(n_non_nan))
%         end
        xlim([0.5 5.5])
%         hold off

%         subplot(1,2,2)
%         y = squeeze(nanmean(rfs_pref_aligned{irun}(:,:,:,icontrast)));
%         yerr = squeeze(nanstd(rfs_pref_aligned{irun}(:,:,:,end)));
        diffy = diff(squeeze(nanmean(rfs_pref_aligned{irun}(:,:,:,icontrast))));
        diffyerr = diff(squeeze(nanstd(rfs_pref_aligned{irun}(:,:,:,end))));
        n_non_nan = sum(~isnan(squeeze(rfs_pref_aligned{irun}(:,ilight,:,icontrast))));
        errorbar(1:5,diffy,diffyerr./sqrt(n_non_nan))
        
        xlim([0.5 5.5])
        hold off
    end
end

%%
for irun=1:2
    for isize=4

        figure
%         subplot(1,2,1)
        hold on
%         for ilight=1:2
        diffy = diff(squeeze(nanmean(rrs_pref_aligned{irun}(:,:,isize,:))));
        diffyerr = diff(squeeze(nanstd(rrs_pref_aligned{irun}(:,:,:,end))));
        n_non_nan = sum(~isnan(squeeze(rrs_pref_aligned{irun}(:,ilight,isize,:))));
        errorbar(1:5,diffy,diffyerr./sqrt(n_non_nan))
%         end
        xlim([0.5 5.5])
%         hold off

%         subplot(1,2,2)
%         y = squeeze(nanmean(rfs_pref_aligned{irun}(:,:,:,icontrast)));
%         yerr = squeeze(nanstd(rfs_pref_aligned{irun}(:,:,:,end)));
        diffy = diff(squeeze(nanmean(rfs_pref_aligned{irun}(:,:,isize,:))));
        diffyerr = diff(squeeze(nanstd(rfs_pref_aligned{irun}(:,:,:,end))));
        n_non_nan = sum(~isnan(squeeze(rfs_pref_aligned{irun}(:,ilight,isize,:))));
        errorbar(1:5,diffy,diffyerr./sqrt(n_non_nan))
        
        xlim([0.5 5.5])
        hold off
    end
end

%%
figure
for irun=1:2
    for ilight=1:2
        subplot(2,2,2*(ilight-1)+irun)
        if irun==1
            plot(sizes,squeeze(nanmean(runcondfr(:,ilight,:,:),1)))
        elseif irun==2
            plot(sizes,squeeze(nanmean(stillcondfr(:,ilight,:,:),1)))
        end
    end
end
%%
figure
for isize=1:3
    for icontrast=1:5
        subplot(3,5,5*(isize-1)+icontrast)
        ilight1 = 2;
        ilight2 = 1;
        scatter(runcondfr(:,ilight1,isize,icontrast),stillcondfr(:,ilight2,isize,icontrast),5)
        hold on
        plot([0 40],[0 40])
        hold off
    end
end
%%
figure
for isize=1:3
    for icontrast=1:5
        subplot(3,5,5*(isize-1)+icontrast)
        ilight1 = 1;
        ilight2 = 1;
        scatter(runcondfr(:,ilight1,isize,icontrast),stillcondfr(:,ilight2,isize,icontrast),5)
        hold on
        plot([0 40],[0 40])
        hold off
    end
end
%%
subplot(1,2,1)
imagesc(squeeze(nanmean(runcondfr(pfsv,2,:,:)-runcondfr(pfsv,1,:,:))))
subplot(1,2,2)
imagesc(squeeze(nanmean(runcondfr(prsv,2,:,:)-runcondfr(prsv,1,:,:))))
%%
rfs = {stillcondfr(pfsv,:,:,:),runcondfr(pfsv,:,:,:)};
rrs = {stillcondfr(prsv,:,:,:),runcondfr(prsv,:,:,:)};
dfs = cell(2,1);
drs = cell(2,1);
for irun=1:2
    dfs{irun} = squeeze(nanmean(rfs{irun}(:,2,:,:)-rfs{irun}(:,1,:,:)));
    drs{irun} = squeeze(nanmean(rrs{irun}(:,2,:,:)-rrs{irun}(:,1,:,:)));
end
% hold on
% for isize=1:3
% %     for icontrast=1:5
%     plot(drs(isize,:),dfs(isize,:))
% %     end
% end
% hold off
% save('r_ephys.mat','rrs','rfs')
%%
scatter(rrs(:,1,1,1),rfs(:,1,1,1))
%%
figure;
for irun=1:2
    subplot(1,2,irun)
    hold on; bar(dfs{irun}(:,end)); bar(drs{irun}(:,end)); hold off
    xticks(1:3)
    xticklabels(gca,sizes)
    legend({'FS','RS'})
    xlabel('size ( ^o )')
    ylabel('delta spikes (Hz)')
    title('VIP Halo change in spike rate, 100% contrast')
end

% subplot(1,2,2)
% hold on; bar(dfs(:,end)); bar(drs(:,end)); hold off
% xticks(1:3)
% xticklabels(gca,sizes)
% legend({'FS','RS'})
% xlabel('size ( ^o )')
% ylabel('delta spikes (Hz)')
% title('VIP Halo change in spike rate, 100% contrast')
%%
figure;
hold on
for isize=1:3
    scatter(dfs(isize,end),drs(isize,end))
end
hold off
%%
figure
irun = 1;
ilight1 = 1;
ilight2 = 2;
for iplot=1
    subplot(1,2,iplot)
    hold on
    for ilight=1:2
        plot(squeeze(nanmean(rfs{irun}(:,ilight,:,end))))
    end
    hold off
end
for iplot=2
    subplot(1,2,iplot)
    hold on
    for ilight=1:2
        plot(squeeze(nanmean(rrs{irun}(:,ilight,:,end))))
    end
    hold off
end
%%
[~,prefsize] = max(squeeze(nanmean(rrs{irun}(:,:,:,end),2)),[],2);

%%
for irun=1:2
    rrs_pref_aligned{irun} = align_by_prefsize(rrs{irun});
    rfs_pref_aligned{irun} = align_by_prefsize(rfs{irun});
end

%%
running_lbl = {'non-running','running'};
for irun=1:2
    
    figure
%     suptitle(running_lbl{irun})
    
    subplot(1,2,1)
    title('RS')
    hold on
    for ilight=1:2
        n_non_nan = sum(~isnan(squeeze(rrs_pref_aligned{irun}(:,ilight,:))))';
        errorbar(1:9,squeeze(nanmean(rrs_pref_aligned{irun}(:,ilight,:))),squeeze(nanstd(rrs_pref_aligned{irun}(:,ilight,:)))./sqrt(n_non_nan))
    end
    xlim([0.5 9.5])
    hold off
    
    subplot(1,2,2)
    title('FS')
    hold on
    for ilight=1:2
        n_non_nan = sum(~isnan(squeeze(rfs_pref_aligned{irun}(:,ilight,:))))';
        errorbar(1:9,squeeze(nanmean(rfs_pref_aligned{irun}(:,ilight,:))),squeeze(nanstd(rrs_pref_aligned{irun}(:,ilight,:)))./sqrt(n_non_nan))
    end
    xlim([0.5 9.5])
    hold off
end

%%
running_lbl = {'non-running','running'};
for irun=1:2
    nsize = size(rrs{irun},3);
    figure
%     suptitle(running_lbl{irun})
    
    subplot(1,2,1)
    title('RS')
    hold on
    for ilight=1:2
        n_non_nan = sum(~isnan(squeeze(rrs{irun}(:,ilight,:))))';
        errorbar(1:nsize,squeeze(nanmean(rrs{irun}(:,ilight,:))),squeeze(nanstd(rrs{irun}(:,ilight,:)))./sqrt(n_non_nan))
    end
    xlim([0.5 nsize+0.5])
    hold off
    
    subplot(1,2,2)
    title('FS')
    hold on
    for ilight=1:2
        n_non_nan = sum(~isnan(squeeze(rfs{irun}(:,ilight,:))))';
        errorbar(1:nsize,squeeze(nanmean(rfs{irun}(:,ilight,:))),squeeze(nanstd(rrs{irun}(:,ilight,:)))./sqrt(n_non_nan))
    end
    xlim([0.5 nsize+0.5])
    hold off
end

%%

for irun=1:2
    for icontrast=1

        figure
%         subplot(1,2,1)
        hold on
%         for ilight=1:2
        diffy = diff(squeeze(nanmean(rrs_pref_aligned{irun}(:,:,:,icontrast))));
        normby = max(abs(diffy));
        diffyerr = diff(squeeze(nanstd(rrs_pref_aligned{irun}(:,:,:,end))));
        n_non_nan = sum(~isnan(squeeze(rrs_pref_aligned{irun}(:,ilight,:,icontrast))));
        errorbar(1:9,diffy/normby,diffyerr./sqrt(n_non_nan)/normby)
%         end
        xlim([0.5 9.5])
%         hold off

%         subplot(1,2,2)
%         y = squeeze(nanmean(rfs_pref_aligned{irun}(:,:,:,icontrast)));
%         yerr = squeeze(nanstd(rfs_pref_aligned{irun}(:,:,:,end)));
        diffy = diff(squeeze(nanmean(rfs_pref_aligned{irun}(:,:,:,icontrast))));
        normby = max(abs(diffy));
        diffyerr = diff(squeeze(nanstd(rfs_pref_aligned{irun}(:,:,:,end))));
        n_non_nan = sum(~isnan(squeeze(rfs_pref_aligned{irun}(:,ilight,:,icontrast))));
        errorbar(1:9,diffy/normby,diffyerr./sqrt(n_non_nan)/normby)
        
        xlim([0.5 9.5])
        hold off
    end
end

%%
irun = 2;
data = cell(2,1);
n_non_nan = cell(2,1);
err = cell(2,1);
all_data = {rrs{irun},rfs{irun}};
for idata=1:2
    data{idata} = squeeze(nanmean(all_data{idata}));
    n_non_nan{idata} = squeeze(sum(~isnan(all_data{idata})));
    err{idata} = squeeze(nanstd(rrs{irun}))./sqrt(n_non_nan{idata});
end
figure
for idata=1:2
    subplot(1,2,idata)
    xerr = squeeze(err{idata}(1,:,:))';
    yerr = squeeze(err{idata}(2,:,:))';
    
    hold on
    for isize=1:5
        x = squeeze(data{idata}(1,isize,:));
        y = squeeze(data{idata}(2,isize,:));
        ye = squeeze(yerr(isize,:));
        xe = squeeze(xerr(isize,:));
        errorbar(x,y,ye/2,ye/2,xe/2,xe/2,'.')
    end
    mx = max(data{idata}(:));
    plot([0 1.1*mx],[0 1.1*mx])
    hold off
end
%%
irun = 2;
data = cell(2,1);
n_non_nan = cell(2,1);
err = cell(2,1);
all_data = {rrs{irun},rfs{irun}};
for idata=1:2
    data{idata} = squeeze(nanmean(all_data{idata}));
    n_non_nan{idata} = squeeze(sum(~isnan(all_data{idata})));
    err{idata} = squeeze(nanstd(rrs{irun}))./sqrt(n_non_nan{idata});
end
figure
formats = {'k.','b.'};
for idata=1:2
%     subplot(1,2,idata)
    xerr = squeeze(err{idata}(1,:,:))';
    yerr = squeeze(err{idata}(2,:,:))';
    
    hold on
    for isize=1:5
        x = squeeze(data{idata}(1,isize,:));
        y = squeeze(data{idata}(2,isize,:));
        ye = squeeze(yerr(isize,:));
        xe = squeeze(xerr(isize,:));
        errorbar(x,y,ye/2,ye/2,xe/2,xe/2,formats{idata})
    end
    mx = max(data{idata}(:));
    plot([0 1.1*mx],[0 1.1*mx])
    hold off
end