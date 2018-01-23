function save_npil_corrected(foldname,newoption)
if nargin < 2 || isempty(newoption)
    newoption = false;
end
fnames = dir([foldname '/*.rois']);
fnames = {fnames(:).name};
for i=1:numel(fnames)
    name = fnames{i};
    load(name,'-mat','Data','Neuropil','ROIdata')
    if newoption
        [corrected,baseline,neuropilMultiplier] = neuropilRR(Data,Neuropil);
        save(strrep(name,'.rois','_corrected'),'corrected','baseline','neuropilMultiplier')
    else
        [neuropilMultiplier,ROIdata] = determineNeuropilWeight(ROIdata);
        corrected = Data-repmat(neuropilMultiplier,1,size(Neuropil,2)).*Neuropil;
        save(strrep(name,'.rois','_corrected'),'corrected','neuropilMultiplier')
    end
end