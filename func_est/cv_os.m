function [beta, Yhat, inds, SM] = cv_os(x, y, varargin)
if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

[n,d] = size(x);

max_norm = get_opt(opts, 'max_norm', min(20,n-1));
list = 0:max_norm;
inds = outerprodinds(list, d, max_norm);
phix = eval_basis(x, inds);

norms = sqrt(sum(inds.^2,2));
last_norms = [norms(1:end-1)~=norms(2:end); true];
lis = find(last_norms);

nlis = length(lis);
ropts.cv = 'loo';
ropts.lambdars = 0;
mses = nan(nlis,1);
for i=1:nlis % TODO: make faster with block inversion
    rreg = ridge_reg(phix(:,1:lis(i)),y,ropts);
    mses(i) = rreg.cv.lam_mse(1);
end

[~,li] = min(mses);
if nargout==4
    [rreg, Yhat, SM] = ridge_reg(phix(:,1:lis(li)),y,ropts); 
else
    [rreg, Yhat] = ridge_reg(phix(:,1:lis(li)),y,ropts); 
end
beta = rreg.beta;
inds = inds(1:lis(li),:);
end
