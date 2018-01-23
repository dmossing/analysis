function [Mfinal,this_perm] = shuf_mat_fmin_anneal(M,T,temps)
% attempts to find an optimal permutation of rows and columns to minimize
% frob. norm ||M-T|| over a symmetric input matrix M, by annealing
if nargin < 3 || isempty(temps)
    temps = zeros(1,10000);
end
assert(all(all(abs(M-M')<1e-10)));
N = size(M,1);
assert(N==size(M,2));
this_perm = 1:N;
this_cost = norm(M-T);
for t=1:numel(temps)
    schwip = randi(N,1,2);
    schwap = fliplr(schwip);
    cand_perm = this_perm;
    cand_perm(schwip) = cand_perm(schwap);
    cand_cost = norm(M(cand_perm,cand_perm)-T);
    dcost = cand_cost-this_cost;
    %     dcost = norm(M(cand_perm(schwip),cand_perm)-T(schwip,:)) ...
    %         - norm(M(this_perm(schwip),this_perm)-T(schwip,:));
    if rand < exp(-dcost/temps(t))
        if dcost > 0 & temps(t) == 0
            disp('watch out')
        end
        this_perm = cand_perm;
        this_cost = cand_cost;
%         subplot(1,2,1);
%         imagesc(M(this_perm,this_perm));
%         xlabel(num2str(temps(t)))
%         subplot(1,2,2);
%         imagesc(T)
%         xlabel(num2str(this_cost))
%         pause(0.001)
    end
end
Mfinal = M(this_perm,this_perm);
end
