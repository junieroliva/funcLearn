function [PCs, inds, ind_used]  = L2_PC(X, varargin)
%L2_PC   Compute denisty projection coefficients
%   Inputs -
%   X: N element cell array of matrices X{i}: ni x d, where X{i} is a 
%       sample of ni points from a distribution over [0,1]^d
%   opts (optional): a struct of options with the following possible fields
%       T: number of basis functions per dimension
%       verbose: print out status
%   Ouputs -
%   PCs: N x T^d matrix of projection coefficients for each distribution

if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end

N = length(X);
d = size(X{1},2);
verbose = get_opt(opts,'verbose', false);
T = get_opt(opts,'T',10);
t_max = get_opt(opts,'t_max',inf);
CV = get_opt(opts,'CV',false);

inds = outerprodinds(T,d);
norms = sum(inds.^2,2);
[sv,si] = sort(norms);
ind_used = si(sv<=t_max^2);
norms = norms(ind_used);

if CV
    t2_vals = unique(norms);
    N_rot = get_opt(opts,'N_rot', min(50,N));
    rprm = randperm(N, N_rot);
    cv_t2 = nan(N_rot,1);
    for ii = 1:N_rot
        % cv number of projection coefficients using min of
        % \int \hat{f}^2 - 2 \int \hat{f}*f
        i = rprm(ii);
        n = size(X{i},1);
        [pc, phix] = get_meanpc(X{i},T);
        pc = pc(ind_used);
        phix = phix(:,ind_used);
        
        sumnorms2 = cumsum(pc.^2);
        sumphi2 = cumsum(sum( phix.^2 ));
        lastnorms = [norms(1:end-1)~=norms(2:end); true];
        
        scores = (1-2*n/(n-1))*sumnorms2(lastnorms) + (2/(n*(n-1)))*sumphi2(lastnorms);
        [~, tm] = min(scores);
        cv_t2(ii) = t2_vals(tm);
    end
    cv_t2 = mean(cv_t2);
    ind_used = ind_used(norms<=cv_t2);
end
inds = inds(ind_used,:);

PCs = nan(N,length(inds));
parfor i=1:N   
    pc = get_meanpc(X{i},T);
    PCs(i,:) = pc(ind_used);
        
    if verbose && mod(i,100)==0
        fprintf('i:%i\n',i);
    end
end

end