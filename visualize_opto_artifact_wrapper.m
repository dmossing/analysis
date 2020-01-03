function trigaligned = visualize_opto_artifact_wrapper(foldname,varargin)

p = inputParser;

p.addParameter('datafold','/media/mossing/data1/2P/');
p.addParameter('sbxfold','/home/mossing/modulation/2P/');

p.parse(varargin{:});

datafold = p.Results.datafold;

s = what(foldname);
foldname = s.path;

path_parts = strsplit(foldname,'/');
file_endings = strsplit(path_parts{end},'_');
datestr = path_parts{end-1};
animalid = path_parts{end-2};
targetfold = sprintf('%s/%s/%s/',datafold,datestr,animalid);
foldname_short = sprintf('%s/%s/',datestr,animalid);

[ontrigaligned,offtrigaligned,meanex,p] = visualize_opto_artifact_fold(foldname_short,'datafold',p.Results.datafold,...
     'sbxfold',p.Results.sbxfold);

 %%
for ff=1:numel(ontrigaligned)
    close all
    figure
    hold on
    p = plot(ontrigaligned{ff}','color',[0 0 1 1e-2]); %,'alpha',0.01)
    plot(mean(ontrigaligned{ff}),'color','r')
    hold off
    savefig(sprintf([targetfold 'opto_artifact_on_%03d.fig'],ff))
end
for ff=1:numel(offtrigaligned)
    close all
    figure
    hold on
    p = plot(offtrigaligned{ff}','color',[0 0 1 1e-2]); %,'alpha',0.01)
    plot(mean(offtrigaligned{ff}),'color','r')
    hold off
    savefig(sprintf([targetfold 'opto_artifact_off_%03d.fig'],ff))
end
close all
figure
hold on
scatter(meanex{1}(:),meanex{2}(:))
plot([min(meanex{1}(:)) max(meanex{1}(:))],[min(meanex{1}(:)) max(meanex{1}(:))])
hold off
p = polyfit(meanex{1}(:),meanex{2}(:),1);
title(sprintf('%.3fx+%.3f',p(1),p(2)))
savefig([targetfold 'opto_artifact_scatterplot.fig'])
save([targetfold 'polyfit_results'],'meanex','p')
close all