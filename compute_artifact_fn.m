function artifact = compute_artifact_fn(roiline,iplane,neuropil,info,lights_on,loffset1,loffset2)
offset = 1; % the first these many triggers are fake
toffset = 1; % static, related to the timing properties of the triggers and LED

nplanes = 4; %numel(roifile);

affected = false(size(neuropil));
control = false(size(neuropil));

for j=1:numel(lights_on)
    frames = 1+floor((-toffset+info.frame(offset+(j-1)*4+1))/nplanes):1+floor((-toffset+info.frame(offset+j*4))/nplanes);
    lines = [info.line(offset+(j-1)*4+1) info.line(offset+j*4)];
    lines(1) = lines(1) + mod(-toffset+info.frame(offset+(j-1)*4+1),nplanes)*512 + loffset1;
    lines(end) = lines(end) + mod(-toffset+info.frame(offset+j*4),nplanes)*512 + loffset2;
    if lights_on(j)
        affected(:,frames(2:end-1)) = 1;
        affected(:,frames(1)) = ((iplane-1)*512+roiline)>lines(1);
        affected(:,frames(end)) = ((iplane-1)*512+roiline)<lines(end);
    else
        control(:,frames(2:end-1)) = 1;
        control(:,frames(1)) = ((iplane-1)*512+roiline)>lines(1);
        control(:,frames(end)) = ((iplane-1)*512+roiline)<lines(end);
    end
end
af_off = [diff(affected,[],2)>0 zeros(size(affected,1),1)]>0;
af_off = af_off | [zeros(size(affected,1),1) diff(affected,[],2)<0]>0;
af_on = [zeros(size(affected,1),1) diff(affected,[],2)>0]>0;
af_on = af_on | [diff(affected,[],2)<0 zeros(size(affected,1),1)]>0;

caf_off = [diff(control,[],2)>0 zeros(size(control,1),1)]>0;
caf_off = caf_off | [zeros(size(control,1),1) diff(control,[],2)<0]>0;
caf_on = [zeros(size(control,1),1) diff(control,[],2)>0]>0;
caf_on = caf_on | [diff(control,[],2)<0 zeros(size(control,1),1)]>0;

artifact_size = mean(neuropil(af_on)-neuropil(af_off));
control_size = mean(neuropil(caf_on)-neuropil(caf_off));

artifact = (artifact_size-control_size)*affected;