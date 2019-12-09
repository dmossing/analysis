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

trigaligned = visualize_opto_artifact_fold(foldname_short,'datafold',p.Results.datafold,...
     'sbxfold',p.Results.sbxfold);

 %%
for ff=1:numel(trigaligned)
    figure
    hold on
    p = plot(trigaligned{ff}','color',[0 0 1 1e-2]); %,'alpha',0.01)
    plot(mean(trigaligned{ff}),'color','r')
    hold off
    savefig(sprintf([targetfold 'opto_artifact_%03d.fig'],ff))
end