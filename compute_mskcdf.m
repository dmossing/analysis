function [mskcdf,iplane] = compute_mskcdf(roifile,info)
nplanes = numel(roifile);
if nargin > 1
    nlines = info.sz(1);
else
    nlines = 512;
end
mskcdf_cell = cell(size(roifile));
iplane_cell = cell(size(roifile));
for i=1:nplanes
    msk = roifile{i}.msk;
    mskline = squeeze(sum(msk,2));
    mskcdf = cumsum(mskline,1)./repmat(sum(mskline,1),size(msk,1),1);
    mskcdf_cell{i} = mskcdf';
    iplane_cell{i} = i*ones(size(mskcdf_cell{i},1),1);
end
if (nargin > 1) & isfield(info,'rect')
    voffset = info.rect(1);
else
    voffset = 0;
end
mskcdf_plane = cell2mat(mskcdf_cell);
mskcdf = ones(size(mskcdf_plane,1),nlines*nplanes);
iplane = cell2mat(iplane_cell);
for i=1:nplanes
    lkat = iplane==i;
    this_offset = (i-1)*nlines+voffset;
    mskcdf(lkat,1:this_offset) = 0;
    mskcdf(lkat,this_offset+1:this_offset+size(mskcdf_plane,2)) = mskcdf_plane(lkat,:);
end