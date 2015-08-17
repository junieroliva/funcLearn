function [beta, Yhat, inds, SM] = cv_os(x, y, varargin)
if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

[n,d] = size(x);

max_norm = get_opt(opts, 'max_norm', (n-2)^(1/d));
list = 0:max_norm;
inds = outerprodinds(list, d, max_norm);
phix = eval_basis(x, inds);

norms = sqrt(sum(inds.^2,2));
last_norms = [norms(1:end-1)~=norms(2:end); true];
lis = find(last_norms);

tol = 1E-10;
nlis = length(lis);

% tic;
% mses = nan(nlis,1);
ropts.cv = 'loo';
ropts.lambdars = 0;
% for i=1:nlis % TODO: make faster with block inversion
%     rreg = ridge_reg(phix(:,1:lis(i)),y,ropts);
%     mses(i) = rreg.cv.lam_mse(1);
% end
% mses_org = mses;
% toc;

% tic;
mses = nan(nlis,1);
best_mse = inf;
best_phixtphix_invphixt = nan;
best_beta = nan;
best_i = nan;
% CV the number of basis functions quickly using block inversion, loocv
% diff_b = nan(nlis,1);
for i=1:nlis 
    if i==1
        phixtphix_inv = inv(phix(:,1:lis(i))'*phix(:,1:lis(i)));
        phixty = phix(:,1:lis(i))'*y;
    else
        B = phix(:,lis(i-1)+1:lis(i))'*phix(:,1:lis(i-1));
        D = phix(:,lis(i-1)+1:lis(i))'*phix(:,lis(i-1)+1:lis(i));
        UP_inv = inv(D-B*A_inv*B');
        BA_inv = B*A_inv;
        A_invBt = A_inv*B';
        phixtphix_inv = ...
            [A_inv + A_invBt*UP_inv*BA_inv, -A_invBt*UP_inv;...
             -UP_inv*BA_inv,                UP_inv];
        phixty = [phixty; phix(:,lis(i-1)+1:lis(i))'*y];
        
        phixtphix = phix(:,1:lis(i))'*phix(:,1:lis(i));
        invml = phixtphix_inv*phixtphix;
        invmr = phixtphix*phixtphix_inv;
        spI = speye(lis(i));
        
        % check if matrix inverse is becaming inprecise
        if max(max(abs(invml-spI)))>tol || max(max(abs(invmr-spI)))>tol
            break;
        end
    end
    
    phixtphix_invphixt = phixtphix_inv*phix(:,1:lis(i))';
    beta = phixtphix_inv*phixty;
    
%     beta0 = (phix(:,1:lis(i))'*phix(:,1:lis(i)))\(phix(:,1:lis(i))'*y);
%     diff_b(i) = max(abs(beta-beta0));
    
    yhat = phix(:,1:lis(i))*beta;
    SMii = sum(phix(:,1:lis(i)).*phixtphix_invphixt',2);
    mses(i) = mean( sum(bsxfun(@times,y-yhat,1./(1-SMii)).^2,2) );
    
    if mses(i)<best_mse
        best_mse = mses(i);
        best_phixtphix_invphixt = phixtphix_invphixt;
        best_beta = beta;
        best_i = i;
    end
    
    A_inv = phixtphix_inv;
end
% toc;

% [~,li] = min(mses);
% if nargout==4
%     %[rreg, Yhat, SM] = ridge_reg(phix(:,1:lis(li)),y,ropts); 
%     PtP_invPt = (phix(:,1:lis(i))'*phix(:,1:lis(i)))\phix(:,1:lis(i))';
%     beta = PtP_invPt*y;
%     SM = phix(:,1:lis(i))*PtP_invPt;
% else
%     %[rreg, Yhat] = ridge_reg(phix(:,1:lis(li)),y,ropts); 
%     beta = (phix(:,1:lis(i))'*phix(:,1:lis(i)))\(phix(:,1:lis(i))'*y);
% end
% %beta = rreg.beta;

phixtphix_invphixt = best_phixtphix_invphixt;
beta = best_beta;
i = best_i;
if nargout==4
    SM = phix(:,1:lis(i))*phixtphix_invphixt;
end

Yhat = phix(:,1:lis(i))*beta;
inds = inds(1:lis(i),:);
end
