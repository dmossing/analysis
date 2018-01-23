function dstr = yymmdd(dt)
if nargin<1
    dt = date;
end
yr = ddigit(year(dt),4);
dstr = [yr(3:4) ddigit(month(dt),2) ddigit(day(dt),2)];