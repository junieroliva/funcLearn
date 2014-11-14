function [ beta, inds ] = get_pc_den( X, varargin )
%GET_PC_REG Get projection coefficients for density estimation
%   Detailed explanation goes here
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end
[n, d] = size(X);

Phi = get_opt(opts,'Phi',[]);
if isempty(Phi)
    % get indices for basis functions
    M = get_opt(opts,'M',floor(n^(1/d)));
    if d==1
        inds = (1:M)';
    elseif d==2
        [i1,i2] = ind2sub([M M], (1:M^2)');
        inds = [i1 i2];
        inds = inds(sum(inds.^2,2)<=M^2);
    else
        [i1,i2,i3] = ind2sub([M M M], (1:M^3)');
        inds = [i1 i2 i3];
        inds = inds(sum(inds.^2,2)<=M^2);
    end
    % get basis values at X
    basis = get_opt(opts,'basis','trig');
    Phi = eval_basis(X,inds,basis);
else
    inds = get_opt(opts,'inds',[]);
end
norms = sum(inds.^2,2);
[~,si] = sort(norms);
inds = inds(si,:);
norms = norms(si);
Phi = Phi(:,si);
% cross-validate number of basis funcs?
do_cv = get_opt(opts,'do_cv',false);
beta = mean(Phi)';
if do_cv
    % cv number of projection coefficients using min of
    % \int \hat{f}^2 - 2 \int \hat{f}*f
    t2_vals = unique(norms);
    sumnorms2 = cumsum(beta.^2);
    sumphi2 = cumsum(sum( Phi.^2 ))';
    lastnorms = [norms(1:end-1)~=norms(2:end); true];
    scores = (1-2*n/(n-1))*sumnorms2(lastnorms) + (2/(n*(n-1)))*sumphi2(lastnorms);
    [~, tm] = min(scores);
    ti = find(norms==t2_vals(tm),1,'last');
    inds = inds(1:ti,:);
    beta = beta(1:ti);
end

end

