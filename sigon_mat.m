function sigon = sigon_mat(endpts)
begs = endpts(1:2:end);
ends = endpts(2:2:end);
N = numel(begs);
assert(numel(ends)==N);
lens = ends-begs+1;
T = min(lens);
sigon = vec_colon(begs,T);