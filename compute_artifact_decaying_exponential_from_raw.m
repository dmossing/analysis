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
nplanes = info.otparam(end);
noffset = nlines*(naround-1)/2;

ts_on = 4*(find(lights_on)-1)+1+nskip;
signal = zeros(numel(ts_on),naround*nlines);

% deal with overflow in trigger frame variable
while find(diff(info.frame)<0,1)
    seam = find(diff(info.frame)<0,1);
    info.frame(seam+1:end) = info.frame(seam+1:end)+65536;
end

for i=1:numel(ts_on)
    im1 = sbxreadpacked(sprintf('%s%s',sbxbase,filebase),info.frame(ts_on(i))-1,naround);
    im2 = sbxreadpacked(sprintf('%s%s',sbxbase,filebase),info.frame(ts_on(i))-1-nplanes,naround);
    signal(i,:) = reshape(mean(im1(:,nboundary+1:end-nboundary,:)-im2(:,nboundary+1:end-nboundary,:),2),1,[]);
end

%%
for i=1:numel(ts_on)
    dif = diff(signal(i,(noffset+info.line(ts_on(i))-winaroundtrig):(noffset+info.line(ts_on(i))+winaroundtrig)));
    dif = dif(1:end-1) + dif(2:end);
    [~,maxind] = max(abs(dif));
    center_on(i) = noffset+info.line(ts_on(i))+maxind-winaroundtrig+1;
    trigaligned(i,:) = signal(i,center_on(i)-nbefore:center_on(i)+nafter);
end

ts_off = nplanes*(find(lights_on)-1)+1+nskip+3;
signal = zeros(numel(ts_off),naround*nlines);
for i=1:numel(ts_off)
    im1 = sbxreadpacked(sprintf('%s%s',sbxbase,filebase),info.frame(ts_off(i))-1,naround);
    im2 = sbxreadpacked(sprintf('%s%s',sbxbase,filebase),info.frame(ts_off(i))-1-nplanes,naround);
    signal(i,:) = reshape(mean(im1(:,nboundary+1:end-nboundary,:)-im2(:,nboundary+1:end-nboundary,:),2),1,[]);
end

for i=1:numel(ts_off)
    dif = diff(signal(i,(noffset+info.line(ts_off(i))-winaroundtrig):(noffset+info.line(ts_off(i))+winaroundtrig)));
    dif = dif(1:end-1) + dif(2:end);
    [~,maxind] = max(abs(dif));
    center_off(i) = noffset+info.line(ts_off(i))+maxind-winaroundtrig+1;
end

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

artifact = uint16(zeros(nlines,max(info.frame)));
for i=1:numel(ts_on)
    frame_on = info.frame(ts_on(i));
    frame_off = info.frame(ts_off(i));
    t = 1:nlines*(frame_off-frame_on)+center_off(i)-center_on(i);
    artifact(nlines*(frame_on-1)+center_on(i)-noffset:nlines*(frame_off-1)+center_off(i)-noffset-1) = expfun_(t,0,xstar);
end

% artifact = artifact';