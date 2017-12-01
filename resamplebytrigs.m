function signal2 = resamplebytrigs(signal1,sz2,frametrig1,frametrig2)
% synchronize signal1 to signal2 by aligning frametrig1 with frametrig2
% output signal2, of size sz2
assert(numel(frametrig1)==numel(frametrig2))
signal2 = zeros(sz2,1);
for i=1:numel(frametrig2)-1
    ind1 = frametrig1(i)+1:frametrig1(i+1);
    ind2 = frametrig2(i)+1:frametrig2(i+1);
    ptno1 = numel(ind1);
    ptno2 = numel(ind2);
    signal2(ind2) = interp1(linspace(0,1,ptno1),signal1(ind1),linspace(0,1,ptno2));
end
%     signal2 = zeros(size(
%     for i,tr in enumerate(frametrig1[:-1]):
%         ptno1 = frametrig1[i+1]-frametrig1[i]
%         ptno2 = frametrig2[i+1]-frametrig2[i]
%         signal2[frametrig2[i]:frametrig2[i+1]] = np.interp(np.linspace(0,1,ptno2),np.linspace(0,1,ptno1),signal1[frametrig1[i]:frametrig1[i+1]])
%     return signal2[frametrig2[0]:frametrig2[-1]]
%     end