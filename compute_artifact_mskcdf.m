function artifact = compute_artifact_mskcdf(roiline,mskcdf,iplane,neuropil,info,yoff,lights_on,loffset1,loffset2,pad_nans)
%%
offset = 1; % the first these many triggers are fake
toffset = 1; % static, related to the timing properties of the triggers and LED

% nplanes = 4; %numel(roifile);
% nlines = 512;
% nplanes = info.otparam(end);
% nlines = info.sz(1);
sz = size(neuropil);

[affected,control] = compute_affected_mskcdf(mskcdf,sz,info,yoff,iplane,lights_on,offset,toffset,loffset1,loffset2,0);
% [affected_r,control_r] = compute_affected_roiline(roiline,iplane,sz,info,yoff,lights_on,offset,toffset,loffset1,loffset2);

% affected = zeros(size(neuropil));
% control = zeros(size(neuropil));

% for j=1:numel(lights_on)
%     frames = 1+floor((-toffset+info.frame(offset+(j-1)*4+1))/nplanes):1+floor((-toffset+info.frame(offset+j*4))/nplanes);
%     lines = [info.line(offset+(j-1)*4+1) info.line(offset+j*4)];
%     lines(1) = lines(1) + mod(-toffset+info.frame(offset+(j-1)*4+1),nplanes)*nlines + loffset1;
%     lines(end) = lines(end) + mod(-toffset+info.frame(offset+j*4),nplanes)*nlines + loffset2;
%     if lines(1)<1
%         lines(1) = lines(1)+nlines*nplanes;
%         frames = [frames(1)-1 frames];
%     elseif lines(1)>nlines*nplanes
%         lines(1) = lines(1)-nlines*nplanes;
%         frames = frames(2:end);
%     end
%     if lines(end)<1
%         lines(end) = lines(end)+nlines*nplanes;
%         frames = frames(1:end-1);
%     elseif lines(end)>nlines*nplanes
%         lines(end) = lines(end)-nlines*nplanes;
%         frames = [frames frames(end)+1];
%     end
%     if lights_on(j)
%      [affected,control] =    affected(:,frames(2:end-1)) = 1;
%         affected(:,frames(1)) = mskcdf(:,lines(1)); %((iplane-1)*512+roiline)>lines(1);
%         affected(:,frames(end)) = 1-mskcdf(:,lines(end)); %((iplane-1)*512+roiline)<lines(end);
%     else
%         control(:,frames(2:end-1)) = 1;
%         control(:,frames(1)) = mskcdf(:,lines(1)); %((iplane-1)*512+roiline)>lines(1);
%         control(:,frames(end)) = 1-mskcdf(:,lines(end)); %((iplane-1)*512+roiline)<lines(end);
%     end
% end
% af_off = [diff(affected,[],2)==1 zeros(size(affected,1),1)]>0;
% af_off = af_off | [zeros(size(affected,1),1) diff(affected,[],2)==-1]>0;
% af_on = [zeros(size(affected,1),1) diff(affected,[],2)==1]>0;
% af_on = af_on | [diff(affected,[],2)==-1 zeros(size(affected,1),1)]>0;
% 
% caf_off = [diff(control,[],2)==1 zeros(size(control,1),1)]>0;
% caf_off = caf_off | [zeros(size(control,1),1) diff(control,[],2)==-1]>0;
% caf_on = [zeros(size(control,1),1) diff(control,[],2)==1]>0;
% caf_on = caf_on | [diff(control,[],2)==-1 zeros(size(control,1),1)]>0;
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
if pad_nans
    affected(af_on) = nan;
end

artifact = (artifact_size-control_size)*affected;

function [affected,control] = compute_affected_roiline(roiline,iplane,sz,info,lights_on,offset,toffset,loffset1,loffset2)
affected = zeros(sz);
control = zeros(sz);
nplanes = info.otparam(end);
nlines = info.sz(1);
for j=1:numel(lights_on)
    frames = 1+floor((-toffset+info.frame(offset+(j-1)*4+1))/nplanes):1+floor((-toffset+info.frame(offset+j*4))/nplanes);
    lines = [info.line(offset+(j-1)*4+1) info.line(offset+j*4)];
    lines(1) = lines(1) + mod(-toffset+info.frame(offset+(j-1)*4+1),nplanes)*nlines + loffset1;
    lines(end) = lines(end) + mod(-toffset+info.frame(offset+j*4),nplanes)*nlines + loffset2;
    if lines(1)<1
        lines(1) = lines(1)+nlines*nplanes;
        frames = [frames(1)-1 frames];
    elseif lines(1)>nlines*nplanes
        lines(1) = lines(1)-nlines*nplanes;
        frames = frames(2:end);
    end
    if lines(end)<1
        lines(end) = lines(end)+nlines*nplanes;
        frames = frames(1:end-1);
    elseif lines(end)>nlines*nplanes
        lines(end) = lines(end)-nlines*nplanes;
        frames = [frames frames(end)+1];
    end
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

function [affected,control] = compute_affected_mskcdf(mskcdf,sz,info,yoff,iplane,lights_on,offset,toffset,loffset1,loffset2,pad_nans)
affected = zeros(sz);
control = zeros(sz);
try
    nplanes = info.otparam(end);
catch
    nplanes = 4;
end
nlines = info.sz(1);
for j=1:numel(lights_on)
    frames = 1+floor((-toffset+info.frame(offset+(j-1)*4+1))/nplanes):1+floor((-toffset+info.frame(offset+j*4))/nplanes);
    lines = [info.line(offset+(j-1)*4+1) info.line(offset+j*4)];
    lines(1) = lines(1) + mod(-toffset+info.frame(offset+(j-1)*4+1),nplanes)*nlines + loffset1;
    lines(end) = lines(end) + mod(-toffset+info.frame(offset+j*4),nplanes)*nlines + loffset2;
    if lines(1)<1
        lines(1) = lines(1)+nlines*nplanes;
        frames = [frames(1)-1 frames];
    elseif lines(1)>nlines*nplanes
        lines(1) = lines(1)-nlines*nplanes;
        frames = frames(2:end);
    end
    if lines(end)<1
        lines(end) = lines(end)+nlines*nplanes;
        frames = frames(1:end-1);
    elseif lines(end)>nlines*nplanes
        lines(end) = lines(end)-nlines*nplanes;
        frames = [frames frames(end)+1];
    end
    thisplane = 1+fix((lines-1)/nlines);
    lines(1) = lines(1) + round(yoff(thisplane(1),frames(1))); % motion correction correction
    lines(end) = lines(end) + round(yoff(thisplane(end),frames(end))); % motion correction correction
    lines = max(lines,1);
    lines = min(lines,nlines*nplanes);
    if lights_on(j)
        affected(:,frames(2:end-1)) = 1;
        if pad_nans
            affected(:,frames(1)) = nan;
            affected(:,frames(end)) = nan;
        else
            affected(:,frames(1)) = 1-mskcdf(:,lines(1)); %((iplane-1)*512+roiline)>lines(1);
            affected(:,frames(end)) = mskcdf(:,lines(end)); %((iplane-1)*512+roiline)<lines(end);
        end
    else
        control(:,frames(2:end-1)) = 1;
        if pad_nans
            control(:,frames(1)) = nan;
            control(:,frames(end)) = nan;
        else
            control(:,frames(1)) = 1-mskcdf(:,lines(1)); %((iplane-1)*512+roiline)>lines(1);
            control(:,frames(end)) = mskcdf(:,lines(end)); %((iplane-1)*512+roiline)<lines(end);
        end
    end
end