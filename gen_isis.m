function isi = gen_isis(trialno,tmin,tmax,p)
% input:    trialno - desired number of ISIs to generate
%           tmin - minimum ISI
%           tmax - maximum ISI
%           p    - dimensionless, describes how nearly exponential to 
%           make the distribution. last time bin is 1/e^p as likely as
%           first time bin. p = 0 -> uniform distribution.
if nargin < 4
   p = 2;
end
q = p/(tmax-tmin);
R = rand(trialno,1);
isi = -1/q*log(exp(-q*tmin) - (exp(-q*tmin)-exp(-q*tmax))*R);
% math