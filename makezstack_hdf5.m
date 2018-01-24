function makezstack_hdf5(fname,zno,framesperz,mindepth,step)
ex = load2P(fname,'frames',1);
sz = size(ex);
fnamebase = strsplit(fname,'.');
fnamebase = fnamebase{1};
fnametgt = [fnamebase '.hdf5']
h5create(fnametgt,'/raw_frames',[zno sz framesperz],'Datatype','uint16')
parpool
parfor i=1:zno
    i
    Images = squeeze(load2P(fname,'frames',(i-1)*framesperz+(1:framesperz)));
    Images = reshape(Images,[1 size(Images)]);
    h5write(fnametgt,'/raw_frames',Images,[i 1 1 1],[1 sz framesperz])
end
h5writeatt(fnametgt,'/','depths',mindepth:step:(mindepth+(zno-1)*step))