function bds = gen_bds(msk)
bds = msk-imerode(msk,strel('disk',1));