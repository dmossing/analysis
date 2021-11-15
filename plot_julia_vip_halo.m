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
rmi = (stillcondfr(:,1,:,:)-runcondfr(:,1,:,:))./(stillcondfr(:,1,:,:)+runcondfr(:,1,:,:));
rmi_light = (stillcondfr(:,2,:,:)-runcondfr(:,2,:,:))./(stillcondfr(:,2,:,:)+runcondfr(:,2,:,:));
omi = (runcondfr(:,2,:,:)-runcondfr(:,1,:,:))./(runcondfr(:,2,:,:)+runcondfr(:,1,:,:));
fr = runcondfr(:,1,:,:);
lkat = (~isnan(rmi) & ~isnan(omi) & ~isnan(rmi_light));
rmi = rmi(lkat);
rmi_light = rmi_light(lkat);
omi = omi(lkat);
fr = fr(lkat);
%%
figure
scatter(-rmi,omi)
%%
figure
hold on
scatter(-rmi,-rmi_light)
plot([-1 1],[-1 1],'k')
xlabel('running modulation index (light off)')
ylabel('running modulation index (light on)')
hold off
%%
[r,p] = corrcoef(rmi,omi);
xlabel('running modulation index')
ylabel('optogenetic modulation index (during running)')
% corrcoef(omi,fr)
%%
tt = zeros(3,5);
figure
for isize=1:3
    for icontrast=1:5
        subplot(3,5,5*(isize-1)+icontrast)
        ilight1 = 2;
        ilight2 = 1;
        xdata = runcondfr(:,ilight1,isize,icontrast);
        ydata = stillcondfr(:,ilight2,isize,icontrast);
        xe = nanstd(xdata)/sqrt(sum(~isnan(xdata)));
        ye = nanstd(ydata)/sqrt(sum(~isnan(ydata)));
        hold on
        scatter(xdata,ydata,5)
        errorbar(nanmean(xdata),nanmean(ydata),ye/2,ye/2,xe/2,xe/2)
        plot([0 40],[0 40])
        hold off
        xlim([0 20])
        ylim([0 20])
        tt(isize,icontrast) = ttest(runcondfr(:,ilight1,isize,icontrast),stillcondfr(:,ilight2,isize,icontrast));
    end
end
%%
figure
for isize=1:3
    for icontrast=1:5
        subplot(3,5,5*(isize-1)+icontrast)
        ilight1 = 2;
        ilight2 = 2;
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
figure
irun = 2;
isize = 1;
hold on
data = squeeze(rrs{irun}(:,1,isize,:));
plot_sem_errorbar(clevels,data./repmat(nanmax(data,[],2),1,5))
data = squeeze(rfs{irun}(:,1,isize,:));
plot_sem_errorbar(clevels,data./repmat(nanmax(data,[],2),1,5))
hold off
%%
irun = 2;
isize = 1;
hold on
data = squeeze(rrs{irun}(:,1,isize,:));
plot_sem_errorbar(clevels,data)%./repmat(nanmax(data,[],2),1,5))
data = squeeze(rfs{irun}(:,1,isize,:));
plot_sem_errorbar(clevels,data)%./repmat(nanmax(data,[],2),1,5))
hold off
%%
irun = 2;
figure
x = squeeze(nanmean(rrs{irun}(:,1,:,:)));
y = drs{irun};
z = dfs{irun};
scatter(x(:),y(:))
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
    scatter(dfs{irun}(isize,end),drs{irun}(isize,end))
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
    for icontrast=4

        figure
        subplot(1,2,1)
        hold on
        for ilight=1:2
            n_non_nan = sum(~isnan(squeeze(rrs_pref_aligned{irun}(:,ilight,:,icontrast))))';
            errorbar(1:5,squeeze(nanmean(rrs_pref_aligned{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rrs_pref_aligned{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        title(['RS, ' running_lbl{irun}])
        xlim([0.5 5.5])
        hold off

        subplot(1,2,2)
        hold on
        for ilight=1:2
            n_non_nan = sum(~isnan(squeeze(rfs_pref_aligned{irun}(:,ilight,:,icontrast))))';
            errorbar(1:5,squeeze(nanmean(rfs_pref_aligned{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rrs_pref_aligned{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        title(['FS, ' running_lbl{irun}])
        xlim([0.5 5.5])
        hold off
    end
end

%%
running_lbl = {'non-running','running'};
for irun=1:2
    figure
    for icontrast=1:5

        subplot(5,2,2*(icontrast-1)+1)
        hold on
        for ilight=1:2
            n_non_nan = sum(~isnan(squeeze(rrs_pref_aligned{irun}(:,ilight,:,icontrast))))';
            errorbar(1:5,squeeze(nanmean(rrs_pref_aligned{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rrs_pref_aligned{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        title(['RS, ' running_lbl{irun}])
        xlim([0.5 5.5])
        hold off

        subplot(5,2,2*(icontrast-1)+2)
        hold on
        for ilight=1:2
            n_non_nan = sum(~isnan(squeeze(rfs_pref_aligned{irun}(:,ilight,:,icontrast))))';
            errorbar(1:5,squeeze(nanmean(rfs_pref_aligned{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rrs_pref_aligned{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        title(['FS, ' running_lbl{irun}])
        xlim([0.5 5.5])
        hold off
    end
end

%%
running_lbl = {'non-running','running'};
H = cell(2,1);
for irun=1:2
    H{irun} = zeros(5,2);
    figure
    for icontrast=1:5

        subplot(5,2,2*(icontrast-1)+1)
        hold on
        data = nanmean(rrs_pref_aligned{irun}(:,:,1:2,icontrast),3);
        scatter(data(:,1),data(:,2),'k')
        plot([0 max(data(:))],[0 max(data(:))])
%         data = nanmean(rrs_pref_aligned{irun}(:,:,4:5,icontrast),3);
%         scatter(data(:,1),data(:,2),'b')
        title(['RS, ' running_lbl{irun}])
%         xlim([0.5 5.5])
        h(icontrast,1) = ttest(data(:,1),data(:,2));
        hold off

        subplot(5,2,2*(icontrast-1)+2)
        hold on
        data = nanmean(rfs_pref_aligned{irun}(:,:,1:2,icontrast),3);
        scatter(data(:,1),data(:,2),'k')
        plot([0 max(data(:))],[0 max(data(:))])
%         data = nanmean(rrs_pref_aligned{irun}(:,:,4:5,icontrast),3);
%         scatter(data(:,1),data(:,2),'b')
        title(['FS, ' running_lbl{irun}])
        h(icontrast,2) = ttest(data(:,1),data(:,2));
%         xlim([0.5 5.5])
        hold off
    end
    H{irun} = h;
end

%%
running_lbl = {'non-running','running'};
for irun=1:2
    figure
    for icontrast=1:5

        subplot(5,2,2*(icontrast-1)+1)
        hold on
        data = nanmean(rrs_pref_aligned{irun}(:,:,4:5,icontrast),3);
        scatter(data(:,1),data(:,2),'k')
        plot([0 max(data(:))],[0 max(data(:))])
%         data = nanmean(rrs_pref_aligned{irun}(:,:,4:5,icontrast),3);
%         scatter(data(:,1),data(:,2),'b')
        title(['RS, ' running_lbl{irun}])
%         xlim([0.5 5.5])
        hold off

        subplot(5,2,2*(icontrast-1)+2)
        hold on
        data = nanmean(rfs_pref_aligned{irun}(:,:,4:5,icontrast),3);
        scatter(data(:,1),data(:,2),'k')
        plot([0 max(data(:))],[0 max(data(:))])
%         data = nanmean(rrs_pref_aligned{irun}(:,:,4:5,icontrast),3);
%         scatter(data(:,1),data(:,2),'b')
        title(['FS, ' running_lbl{irun}])
%         xlim([0.5 5.5])
        hold off
    end
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
for ilight=1:2
    for icontrast=5
        figure
        subplot(1,2,1)
        hold on
        for irun=1:2
            n_non_nan = sum(~isnan(squeeze(rrs{irun}(:,ilight,:,icontrast))))';
            errorbar(1:3,squeeze(nanmean(rrs{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rrs{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        xlim([0.5 3.5])
        hold off

        subplot(1,2,2)
        hold on
        for irun=1:2
            n_non_nan = sum(~isnan(squeeze(rfs{irun}(:,ilight,:,icontrast))))';
            errorbar(1:3,squeeze(nanmean(rfs{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rfs{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        xlim([0.5 3.5])
        hold off
    end
end
%%
for ilight=1:2
    for isize=2
        figure
        subplot(1,2,1)
        hold on
        for irun=1:2
            n_non_nan = sum(~isnan(squeeze(rrs{irun}(:,ilight,isize,:))))';
            errorbar(1:5,squeeze(nanmean(rrs{irun}(:,ilight,isize,:))),squeeze(nanstd(rrs{irun}(:,ilight,isize,:)))./sqrt(n_non_nan))
        end
        xlim([0.5 5.5])
        hold off

        subplot(1,2,2)
        hold on
        for irun=1:2
            n_non_nan = sum(~isnan(squeeze(rfs{irun}(:,ilight,isize,:))))';
            errorbar(1:5,squeeze(nanmean(rfs{irun}(:,ilight,isize,:))),squeeze(nanstd(rfs{irun}(:,ilight,isize,:)))./sqrt(n_non_nan))
        end
        xlim([0.5 5.5])
        hold off
    end
end
%%
for irun=1:2
    for icontrast=5
        figure
        subplot(1,2,1)
        hold on
        for ilight=1:2
            n_non_nan = sum(~isnan(squeeze(rrs_pref_aligned{irun}(:,ilight,:,icontrast))))';
            errorbar(1:5,squeeze(nanmean(rrs_pref_aligned{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rrs_pref_aligned{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        xlim([0.5 5.5])
        hold off

        subplot(1,2,2)
        hold on
        for ilight=1:2
            n_non_nan = sum(~isnan(squeeze(rfs_pref_aligned{irun}(:,ilight,:,icontrast))))';
            errorbar(1:5,squeeze(nanmean(rfs_pref_aligned{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rfs_pref_aligned{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        xlim([0.5 5.5])
        hold off
    end
end
%%
for irun=1:2
    for icontrast=5
        figure
        subplot(1,2,1)
        hold on
        for ilight=1:2
            n_non_nan = sum(~isnan(squeeze(rrs_pref_aligned{irun}(:,ilight,:,icontrast))))';
            errorbar(1:5,squeeze(nanmean(rrs_pref_aligned{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rrs_pref_aligned{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        xlim([0.5 5.5])
        hold off

        subplot(1,2,2)
        hold on
        for ilight=1:2
            n_non_nan = sum(~isnan(squeeze(rfs_pref_aligned{irun}(:,ilight,:,icontrast))))';
            errorbar(1:5,squeeze(nanmean(rfs_pref_aligned{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rfs_pref_aligned{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        xlim([0.5 5.5])
        hold off
    end
end

%%
for irun=1:2
    for icontrast=[5]
        figure
        hold on
%         subplot(1,2,1)
        ratio = squeeze((rrs{irun}(:,2,:,icontrast)-rrs{irun}(:,1,:,icontrast))./(rrs{irun}(:,1,:,icontrast)+rrs{irun}(:,2,:,icontrast)));
        n_non_nan = sum(~isnan(ratio));
        errorbar(1:3,squeeze(nanmean(ratio)),squeeze(nanstd(ratio))./sqrt(n_non_nan))
        xlim([0.5 3.5])

%         subplot(1,2,2)
        ratio = squeeze((rfs{irun}(:,2,:,icontrast)-rfs{irun}(:,1,:,icontrast))./(rfs{irun}(:,1,:,icontrast)+rfs{irun}(:,2,:,icontrast)));
        n_non_nan = sum(~isnan(ratio));
        errorbar(0.1+[1:3],squeeze(nanmean(ratio)),squeeze(nanstd(ratio))./sqrt(n_non_nan))
        xlim([0.5 3.5])
        hold off
    end
end
%%
for irun=1:2
    for icontrast=[3 4 5]
        figure
        generate_omi_by_size_plot(rrs_pref_aligned{irun},rfs_pref_aligned{irun},icontrast)
        title(sprintf('%d%% contrast',round(100*clevels(icontrast))))
%         saveas(gcf,sprintf('rs_fs_omi_by_rel_size_run%d_contrast%d.jpg',irun,icontrast))
    end
end
%%
for irun=1:2
    for isize=[1 2 3]
        figure
        generate_omi_by_contrast_plot(rrs{irun},rfs{irun},isize)
        title(sprintf('%d^o',round(sizes(isize))))
%         saveas(gcf,sprintf('rs_fs_omi_by_rel_size_run%d_contrast%d.jpg',irun,icontrast))
    end
end
%%
for irun=1:2
    for icontrast=[4 5]
        figure
        generate_omi_by_size_plot(rrs{irun},rfs{irun},icontrast)
        title(sprintf('%d%% contrast',round(100*clevels(icontrast))))
        xlabel('Size ( ^o )')
        xticks(1:3)
        xticklabels(sizes)
        saveas(gcf,sprintf('rs_fs_omi_by_abs_size_run%d_contrast%d.jpg',irun,icontrast))
    end
end
%%
figure
colors = {'k','b'};
for irun=1:2
    subplot(1,2,irun)
    xdata = squeeze(nanmean(rrs{irun}));
    n_non_nan = squeeze(sum(~isnan(rrs{irun})));
    xerr = squeeze(nanstd(rrs{irun}))./sqrt(n_non_nan);
    ydata = squeeze(nanmean(rfs{irun}));
    n_non_nan = squeeze(sum(~isnan(rfs{irun})));
    yerr = squeeze(nanstd(rfs{irun}))./sqrt(n_non_nan);
    hold on
    for ilight=1:2
        x = xdata(ilight,:,:);
        y = ydata(ilight,:,:);
        ye = yerr(ilight,:,:);
        xe = xerr(ilight,:,:);
        [p,S] = polyfit(x,y,2);
        xx = linspace(min(x(:)),max(x(:)),100);
        [y_fit,delta] = polyval(p,xx,S);
        errorbar(x(:),y(:),ye(:)/2,ye(:)/2,xe(:)/2,xe(:)/2,[colors{ilight} '.'])
%         plot(xx,y_fit,[colors{ilight} '-'])
%         plot(xx,y_fit+delta,[colors{ilight} '--'])
%         plot(xx,y_fit-delta,[colors{ilight} '--'])
    end
    hold off
    xlabel('RS FR (Hz)')
    legend({'light off','','','','light on'})
end
subplot(1,2,1)
ylabel('FS FR (Hz)')
% saveas(gcf,'rs_fs_scatter_errorbar_with_fit_lines.jpg')
%%
data = squeeze(nanmean(rrs{irun}(:,:,:,:)))./(squeeze(nanmean(rrs{irun}(:,:,:,:)))+squeeze(nanmean(rfs{irun}(:,:,:,:))));
plot(reshape(data,2,[]),'k')
xticks([1 2])
set(gca,'xticklabel',{'light off','light on'})
ylabel('RS firing rate / (RS firing rate + FS firing rate)')
xlim([0.9 2.1])
saveas(gcf,'rs_fs_ratio_light_off_light_on_conditions.eps')
%%
% for ilight=1:2
%     figure
%     imagesc(squeeze(data(ilight,:,:)))
% end
%%
for irun=1:2
    for icontrast=[2 5]
        figure
        subplot(1,2,1)
        hold on
        for ilight=1:2
            n_non_nan = sum(~isnan(squeeze(rrs{irun}(:,ilight,:,icontrast))))';
            errorbar(1:3,squeeze(nanmean(rrs{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rrs{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        xlim([0.5 3.5])
        hold off

        subplot(1,2,2)
        hold on
        for ilight=1:2
            n_non_nan = sum(~isnan(squeeze(rfs{irun}(:,ilight,:,icontrast))))';
            errorbar(1:3,squeeze(nanmean(rfs{irun}(:,ilight,:,icontrast))),squeeze(nanstd(rfs{irun}(:,ilight,:,end)))./sqrt(n_non_nan))
        end
        xlim([0.5 3.5])
        hold off
    end
end
%%

for irun=1:2
    for icontrast=5

        figure
%         subplot(1,2,1)
        hold on
%         for ilight=1:2
        diffy = diff(squeeze(nanmean(rrs_pref_aligned{irun}(:,:,:,icontrast))));
        normby = max(abs(diffy));
        diffyerr = diff(squeeze(nanstd(rrs_pref_aligned{irun}(:,:,:,end))));
        n_non_nan = sum(~isnan(squeeze(rrs_pref_aligned{irun}(:,ilight,:,icontrast))));
        errorbar(0.1+[1:5],diffy/normby,diffyerr./sqrt(n_non_nan)/normby)
%         end
        xlim([0.5 5.5])
%         hold off

%         subplot(1,2,2)
%         y = squeeze(nanmean(rfs_pref_aligned{irun}(:,:,:,icontrast)));
%         yerr = squeeze(nanstd(rfs_pref_aligned{irun}(:,:,:,end)));
        diffy = diff(squeeze(nanmean(rfs_pref_aligned{irun}(:,:,:,icontrast))));
        normby = max(abs(diffy));
        diffyerr = diff(squeeze(nanstd(rfs_pref_aligned{irun}(:,:,:,end))));
        n_non_nan = sum(~isnan(squeeze(rfs_pref_aligned{irun}(:,ilight,:,icontrast))));
        errorbar(-0.1+[1:5],diffy/normby,diffyerr./sqrt(n_non_nan)/normby)
        
        xlim([0.5 5.5])
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
    xerr = squeeze(err{idata}(1,:,:));
    yerr = squeeze(err{idata}(2,:,:));
    
    hold on
    for isize=1:3
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
figure
hold on
omi = (stillcontrolfr(:,2)-stillcontrolfr(:,1))./(stillcontrolfr(:,1)+stillcontrolfr(:,2));
hist(omi(prsv))
hist(omi(pfsv))
hold off