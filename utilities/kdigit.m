function z = kdigit(n,mx)
d = ceil(log10(mx+1e-10));
z = ddigit(n,d);
end