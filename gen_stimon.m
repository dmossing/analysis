function stimon = gen_stimon(stimParams,stimroi)
uori = unique(stimParams(1,:));
nori = numel(uori);
ntrial = round(size(stimParams,2)/nori);
stimon = zeros(nori,ntrial);
for i=1:nori
    stimon(i,:) = stimParams(2,stimParams(1,:)==uori(i))==stimroi;
end