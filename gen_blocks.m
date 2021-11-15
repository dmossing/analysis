function [yblock,xblock] = gen_blocks(sz,block_width,overlap)
starting_pts = cell(2,1);
ending_pts = cell(2,1);
intvl = floor(block_width*(1-overlap));
for iaxis=1:2
    starting_pts{iaxis} = 0:intvl:sz(iaxis)-block_width;
    ending_pts{iaxis} = block_width:intvl:sz(iaxis);
    if ending_pts{iaxis}(end) ~=sz(iaxis)
        starting_pts{iaxis} = [starting_pts{iaxis} sz(iaxis)-block_width];
        ending_pts{iaxis} = [ending_pts{iaxis} sz(iaxis)];
    end
end
[xxs,yys] = meshgrid(starting_pts{2},starting_pts{1});
[xxe,yye] = meshgrid(ending_pts{2},ending_pts{1});
yblock = [yys(:) yye(:)];
xblock = [xxs(:) xxe(:)];