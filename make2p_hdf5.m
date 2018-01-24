function make2p_hdf5(fname0,zno)
ex = squeeze(sbxreadpacked(fname0,0,1));
sz = size(ex);
fnametgt = [fname0 '.hdf5'];
frameno = calcframeno(fname0);
startat = 1+(1:zno);
tno = floor((frameno-1)/zno);
endat = (frameno-zno+1:frameno)-rem(frameno-1,zno);
tic
% h5create(fnametgt,'/')
% datasetId = h5datacreate(fnametgt,'/raw_frames','size',[tno zno sz],'type','uint16');
h5create(fnametgt,'/raw_frames',[tno zno sz],'Datatype','uint16')
prop = hdf5prop(fnametgt,'/raw_frames','rw');
toc
for z=1:zno
    for t=1:tno
        [z t]
        Images = squeeze(sbxreadpacked(fname0,zno*(t-1)+startat(z),1));
        Images = reshape(Images,[1 1 size(Images)]);
        tic
        prop(t,z,:,:) = Images;
%         h5write(fnametgt,'/raw_frames',Images,[t z 1 1],[1 1 sz])
        toc
    end
end



function frameno = calcframeno(fname0)
d = dir([fname0 '.sbx']);
load(fname0,'info');
switch info.channels
    case 1
        info.nchan = 2;      % both PMT0 & 1
        factor = 1;
    case 2
        info.nchan = 1;      % PMT 0
        factor = 2;
    case 3
        info.nchan = 1;      % PMT 1
        factor = 2;
end
frameno = d.bytes/info.recordsPerBuffer/info.sz(2)*factor/4;