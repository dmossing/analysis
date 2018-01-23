function sigon = sigon_mat_addon(endpts,addon)
if ~exist('addon','var') || isempty(addon)
    addon = 0;
end
begs = endpts(1:2:end);
ends = endpts(2:2:end);
N = numel(begs);
assert(numel(ends)==N);
lens = ends-begs+1;
T = min(lens);
sigon = vec_colon(begs,T+addon);