function [t,trig] = maketriggervect(tno,frame)
t = 1:frame(1);
trig = zeros(1,frame(1));
for i=1:numel(frame)-1
    t = [t frame(i) frame(i):frame(i+1)];
    trig = [trig 1 zeros(1,frame(i+1)-frame(i)+1)];
end
t = [t frame(end) frame(end):tno];
trig = [trig 1 zeros(1,tno-frame(end)+1)];