function artifact = compute_opto_artifact_decaying_exponential(sbxbase,filebase,mskcdf,iplane,neuropil,info,yoff,lights_on,loffset1,loffset2)

% stimbase = '/home/mossing/modulation/visual_stim/190807/M0153/';
% foldbase = '/home/mossing/modulation/matfiles/190807/M0153/ot/';
% sbxbase = '/home/mossing/modulation/2P/190807/M0153/';
% filebase = 'M0153_090_004';
filename = sprintf('%s%s',sbxbase,filebase);
% for i=1:4
%     filenames{i} = sprintf('%s%s_ot_%03d.rois',foldbase,filebase,i-1);
% end

%%
offset = 1;
toffset = 1;
ts = 4*(find(lights_on)-1)+1+offset;
naround = 3;
nbefore = 100;
nafter = 200;
nlines = info.sz(1);
sz = size(neuropil);
signal = zeros(numel(ts),naround*nlines);
for i=1:numel(ts)
    im = sbxreadpacked(filename,info.frame(ts(i))-1,naround);
    signal(i,:) = reshape(mean(im(:,101:end,:),2),1,[]);
end
%%
trigaligned = zeros(numel(ts),nbefore+nafter+1);
for i=1:numel(ts)
    trigaligned(i,:) = signal(i,nlines+info.line(ts(i))-nbefore:nlines+info.line(ts(i))+nafter);
end
% figure
% plot(trigaligned')
%%
t = -nbefore:nafter;
t0 = loffset1;
expfun = @(t,t0,a,b,c,d)(t>t0).*(a*exp(-(t-t0)/b)+d)+c;
expfun_ = @(t,t0,x) expfun(t,t0,x(1),x(2),x(3),x(4));
lsqfun = @(model)sum(abs(mean(trigaligned)-model).^1);
costfun = @(x)lsqfun(expfun_(t,t0,x));
c = mean(trigaligned(:,1));
d = mean(trigaligned(:,end))-c;
a = 5e3;
b = 30;
x0 = [a b c d];
xstar = fminunc(costfun,x0);
line_offset_to_artifact = @(t) expfun_(t,0,xstar);
[artifact,~] = compute_affected_mskcdf(mskcdf,sz,info,yoff,iplane,lights_on,offset,toffset,loffset1,loffset2,line_offset_to_artifact);

%%
% nplanes = max(iplane);
% artifact_cell = cell(nplanes,1);
% for i=1:nplanes
%     artifact_cell{i} = artifact(iplane==i,:);
% end
%%
% t0s = 90:110;
% cstar = zeros(size(t0s));
% xstar = zeros(numel(t0s),4);
% for i=1:numel(t0s)
%     expfun = @(t,t0,a,b,c,d)(t>t0).*(a*exp(-(t-t0)/b)+d)+c;
%     expfun_ = @(t,t0,x) expfun(t,t0,x(1),x(2),x(3),x(4));
%     t = [1:size(trigaligned,2)];
%     lsqfun = @(model)sum(abs(mean(trigaligned)-model).^1);
%     t0 = t0s(i);
%     costfun = @(x)lsqfun(expfun_(t,t0,x));
%     c = mean(trigaligned(:,1));
%     d = mean(trigaligned(:,end))-c;
%     a = 5e3;
%     b = 30;
%     x0 = [a b c d];
%     xstar(i,:) = fminunc(costfun,x0);
%     cstar(i) = costfun(xstar(i,:));
% end
% [~,imin] = min(cstar);
%%
% figure
% hold on
% plot(mean(trigaligned))
% plot(expfun_(t,t0,xstar))
% % plot(expfun_(t,t0,x0))
% hold off

function [affected,control] = compute_affected_mskcdf(mskcdf,sz,info,yoff,iplane,lights_on,offset,toffset,loffset1,loffset2,line_offset_to_artifact)
nroi = size(mskcdf,1);
mskpdf = diff(mskcdf,[],2);
mskpdf = sparse(mskpdf.*(mskpdf>1e-3));
affected = zeros(sz);
control = zeros(sz);
nplanes = info.otparam(end);
nlines = info.sz(1);
baseline = line_offset_to_artifact(inf);
t = 0:nplanes*nlines-1;
first_frame = line_offset_to_artifact(t);
% second_frame = line_offset_to_artifact(t+nplanes*nlines);
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
    lines(1) = lines(1) + round(-yoff(thisplane(1),frames(1))); % motion correction correction
    lines(end) = lines(end) + round(-yoff(thisplane(end),frames(end))); % motion correction correction
    lines = max(lines,1);
    lines = min(lines,nlines*nplanes);
    mskpdf_this_stim = circshift(mskpdf,[0 1-lines(1)]);
    this_first_frame = zeros(size(first_frame));
    this_second_frame = zeros(size(first_frame));
    this_first_frame(lines(1):end) = first_frame(1:end-lines(1)+1);
    this_second_frame(1:lines(1)-1) = first_frame(end-lines(1)+2:end);
    this_second_frame(lines(1):end) = baseline;
    if lights_on(j)
        affected(:,frames(1)) = sum(mskpdf.*this_first_frame,2); %((iplane-1)*512+roiline)>lines(1);
        affected(:,frames(2)) = sum(mskpdf.*this_second_frame,2);
        affected(:,frames(3:end-1)) = baseline;
        affected(:,frames(end)) = baseline.*mskcdf(:,lines(end)); %((iplane-1)*512+roiline)<lines(end);
    else
        control(:,frames(1)) = sum(mskpdf.*this_first_frame,2); %((iplane-1)*512+roiline)>lines(1);
        control(:,frames(2)) = sum(mskpdf.*this_second_frame,2);
        control(:,frames(3:end-1)) = baseline;
        control(:,frames(end)) = baseline.*mskcdf(:,lines(end)); %((iplane-1)*512+roiline)<lines(end);
    end
end

% % artifact = compute_artifact_mskcdf(roiline,iplane,neuropil,info,lights_on,loffset1,loffset2);
% artifact_cell = cell(nplanes,1);
% for i=1:nplanes
%     artifact_cell{i} = affected(iplane==i,:);
% end