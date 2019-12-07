function artifact = compute_artifact_decaying_exponential_from_raw(filebase,sbxbase,resultbase)

load(sprintf('%s%s',sbxbase,filebase),'info')

load(sprintf('%s%s',resultbase,filebase),'result')

if isfield(result,'gratingInfo') && isfield(result.gratingInfo,'lightsOn')
    lights_on = result.gratingInfo.lightsOn;
else
    artifact = [];
    return
end

nskip = 1;
nbefore = 100; 
nafter = 200;
nboundary = 100;
winaroundtrig = 20;
naround = 3;
nlines = info.sz(1);
nplanes = 4; %info.otparam(end); % why isn't otparam being saved??
noffset = nlines*(naround-1)/2;
levels = unique(lights_on(lights_on>0));
nlevels = numel(levels); % number of distinct light levels

ts_on = cell(nlevels,1);
signal = cell(nlevels,1);
for ilevel=1:nlevels
    ts_on{ilevel} = 4*(find(lights_on==levels(ilevel))-1)+1+nskip;
    signal{ilevel} = zeros(numel(ts_on{ilevel}),naround*nlines);
end

% deal with overflow in trigger frame variable
while find(diff(info.frame)<0,1)
    seam = find(diff(info.frame)<0,1);
    info.frame(seam+1:end) = info.frame(seam+1:end)+65536;
end

%%

artifact = uint16(zeros(nlines,max(info.frame)));
for ilevel=1:nlevels
    for itrial=1:numel(ts_on{ilevel})
        im1 = sbxreadpacked(sprintf('%s%s',sbxbase,filebase),info.frame(ts_on{ilevel}(itrial))-1,naround);
        im2 = sbxreadpacked(sprintf('%s%s',sbxbase,filebase),info.frame(ts_on{ilevel}(itrial))-1-nplanes,naround);
        signal{ilevel}(itrial,:) = reshape(mean(im1(:,nboundary+1:end-nboundary,:)-im2(:,nboundary+1:end-nboundary,:),2),1,[]);
    end
    
    trigaligned = zeros(numel(ts_on{ilevel}),nbefore+nafter+1);
    for itrial=1:numel(ts_on{ilevel})
        dif = diff(signal{ilevel}(itrial,(noffset+info.line(ts_on{ilevel}(itrial))-winaroundtrig):(noffset+info.line(ts_on{ilevel}(itrial))+winaroundtrig)));
        dif = dif(1:end-1) + dif(2:end);
        [~,maxind] = max(abs(dif));
        center_on(itrial) = noffset+info.line(ts_on{ilevel}(itrial))+maxind-winaroundtrig+1;
        trigaligned(itrial,:) = signal{ilevel}(itrial,center_on(itrial)-nbefore:center_on(itrial)+nafter);
    end
    
    %%
    
    ts_off = nplanes*(find(lights_on==levels(ilevel))-1)+1+nskip+3;
    signal{ilevel} = zeros(numel(ts_off),naround*nlines);
    for itrial=1:numel(ts_off)
        im1 = sbxreadpacked(sprintf('%s%s',sbxbase,filebase),info.frame(ts_off(itrial))-1,naround);
        im2 = sbxreadpacked(sprintf('%s%s',sbxbase,filebase),info.frame(ts_off(itrial))-1-nplanes,naround);
        signal{ilevel}(itrial,:) = reshape(mean(im1(:,nboundary+1:end-nboundary,:)-im2(:,nboundary+1:end-nboundary,:),2),1,[]);
    end
    
    trigaligned_off = zeros(numel(ts_off),nbefore+nafter+1);
    for itrial=1:numel(ts_off)
        dif = diff(signal{ilevel}(itrial,(noffset+info.line(ts_off(itrial))-winaroundtrig):(noffset+info.line(ts_off(itrial))+winaroundtrig)));
        dif = dif(1:end-1) + dif(2:end);
        [~,maxind] = max(abs(dif));
        center_off(itrial) = noffset+info.line(ts_off(itrial))+maxind-winaroundtrig+1;
        trigaligned_off(itrial,:) = signal{ilevel}(itrial,center_off(itrial)-nbefore:center_off(itrial)+nafter);
    end
    
    %%
    
    t0 = nbefore;
    expfun = @(t,t0,a,b,c,d)(t>=t0).*(a*exp(-(t-t0)/b)+d)+c;
    expfun_ = @(t,t0,x) expfun(t,t0,x(1),x(2),x(3),x(4));
    t = [1:size(trigaligned,2)];
    lsqfun = @(model)sum(abs(mean(trigaligned)-model).^1);
    costfun = @(x)lsqfun(expfun_(t,t0,x));
    c = mean(trigaligned(:,1));
    d = mean(trigaligned(:,end))-c;
    a = 5e3;
    b = 30;
    x0 = [a b c d];
    xstar = fminunc(costfun,x0);
    cstar = costfun(xstar);
    
    for itrial=1:numel(ts_on{ilevel})
        frame_on = info.frame(ts_on{ilevel}(itrial));
        frame_off = info.frame(ts_off(itrial));
%         t = 1:nlines*(frame_off-frame_on)+center_off(itrial)-center_on(itrial);
    %     artifact(nlines*(frame_on-1)+center_on(itrial)-noffset:nlines*(frame_off-1)+center_off(itrial)-noffset-1) = expfun_(t,0,xstar);
        lastline = min(nlines*frame_off+center_off(itrial)-noffset-1,numel(artifact));
        t = 1:lastline-nlines*frame_on-center_on(itrial)+noffset+1;
        artifact(nlines*frame_on+center_on(itrial)-noffset:lastline) = expfun_(t,0,xstar);
    end
end

% artifact = artifact';
