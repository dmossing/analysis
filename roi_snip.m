function msk = roi_snip(img,msk)
imagesc(img)
visboundaries(sum(msk,3))
S.fh = figure('units','pixels','buttondownfcn',@fh_bdfcn); % 'keypressfcn',@fh_kpfcn,
S.msk = msk;

function [] = fh_bdfcn(H,E)
S = guidata(H);
pt = found(get(gca,'CurrentPoint'));
in_this_roi = msk(pt(2),pt(1),:)>0;
candidates = find(in_this_roi);
ncands = numel(candidates);
if false %ncands>1
    for icand=1:ncands
        bw = bwboundaries(msk(:,:,candidates(icand))>0);
        p(icand) = patch(bw(:,2),bw(:,1),'FaceColor','m','EdgeColor','k','FaceAlpha',0.25);
    end
    current_icand = 1;
    finalized = false;
    while ~finalized
        bw = bwboundaries(msk(:,:,candidates(current_icand))>0);
        pp = patch(bw(:,2),bw(:,1),'FaceColor','m','EdgeColor','k');
        current_icand = 1+mod(current_icand+to_add-1,ncands);
    end
    
    % draw multiple transparent ROIs, with one highlighted. Arrow keys
    % swap them, and ENTER indicates current highlighted ROI selected.
else
    bw = bwboundaries(msk(:,:,candidates(1))>0);
    pp = patch(bw(:,2),bw(:,1),'FaceColor','m','EdgeColor','k');
    current_icand = 1;
    % set this ROI to the selected one
end
target_roi = candidates(current_icand);

% gui draw a line. Everything above that line (dot product value with
% orthogonal?) is one ROI, everything below is appended to the end
lh = imline;
pos = getPosition(lh);
vector = diff(pos);
orth = [vector(2); -vector(1)];
line_dot_prod = sum(pos(1,:)*orth);
this_msk = msk(:,:,target_roi)>0;
[y,x] = find(this_msk);
msk_dot_prods = [x y]*orth;
msk1 = this_msk & msk_dot_prods>line_dot_prod;
msk2 = this_msk & msk_dot_prods<line_dot_prod;
msk(:,:,target_roi) = msk1;
msk = cat(3,msk,msk2);
guidata(S.fh,S);