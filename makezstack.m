function zstack = makezstack(fname,zno,framesperz)
Images = squeeze(load2P(fname,'frames',1:zno*framesperz));
sz = size(Images);
Images = reshape(Images,sz(1),sz(2),framesperz,zno);
zstack = permute(Images,[4 1 2 3]);