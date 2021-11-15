function plot_sem_errorbar(x,data,fmt)
n_non_nan = squeeze(sum(~isnan(data)));
if nargin < 3
    errorbar(x,squeeze(nanmean(data)),squeeze(nanstd(data))./sqrt(n_non_nan),'.')
else
    errorbar(x,squeeze(nanmean(data)),squeeze(nanstd(data))./sqrt(n_non_nan),'.',fmt)
end