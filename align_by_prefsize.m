function arr_pref_aligned = align_by_prefsize(arr)
%%
nsize = size(arr,3);
[~,prefsize] = max(squeeze(nanmean(arr(:,:,:,end),2)),[],2);
%%
sz = size(arr);
%%
while numel(sz)<4
    sz = [sz 1];
end
%%
nroi = sz(1);
arr_pref_aligned = nan*ones(nroi,sz(2),2*nsize-1,sz(4));
startat = nsize-1-prefsize; % prefsize-nsize+1; % (2*nsize-1)+
endat = prefsize+nsize-1; % (2*nsize-1)+
these_nos = zeros(nroi,nsize);
for iroi=1:nroi
%     nsize-prefsize(iroi):2*nsize-1-prefsize(iroi);
    these_nos(iroi,:) = nsize+1-prefsize(iroi):2*nsize-prefsize(iroi); %startat(iroi):endat(iroi);
    arr_pref_aligned(iroi,:,these_nos(iroi,:),:) = arr(iroi,:,:,:);
end
% %%
% hold on
% for ilight=1:2
%     plot(squeeze(nanmean(arr_pref_aligned(:,ilight,:,end))))
% end
% hold off