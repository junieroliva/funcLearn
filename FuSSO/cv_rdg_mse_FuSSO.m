function [ Y_pred, norms, sqerr, lambdar, lambdars ] = cv_rdg_mse_FuSSO( Y, PC, p, varargin )
%cv_mse_FuSSO Summary of this function goes here
%   Detailed explanation goes here
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end
N = size(PC,1);
M_n = size(PC,2)/p;
verbose = get_opt(opts,'verbose',false);
% get lambdas
intercept = get_opt(opts,'intercept',true);
opts.intercept = intercept;
lambdars = get_opt(opts,'lambdars',2.^(30:-1:-30));
opts.lambdars = lambdars;

norms = nan(N,p);
Y_pred = nan(N,1);
sqerr = nan(N,1);
lambdar = nan(N,1);
stime = tic;
for i = 1:N
    trn_set = true(N,1);
    trn_set(i) = false;
    lambdar(i) = cv_rdg_lam_FuSSO( Y(trn_set), PC(trn_set,:), opts );

    if intercept
        PC_act = [PC(trn_set,:) ones(N-1,1)];
    else
        PC_act = PC(trn_set,:);
    end
    [U,S] = eig(PC_act*PC_act');
    S = diag(S);
    PCtU = PC_act'*U;
    PCtY = PC_act'*Y(trn_set);
    UtPCPCtY = PCtU'*PCtY;
    beta_act = (1/lambdar(i))*(PCtY-PCtU*(UtPCPCtY./(S+lambdar(i))));
    if intercept
        Y_pred(i) = PC(i,:)*beta_act(1:end-1)+beta_act(end);
        norms(i,:) = sqrt(sum(reshape(beta_act(1:end-1),M_n,[]).^2,1));
    else
        Y_pred(i) = PC(i,:)*beta_act;
        norms(i,:) = sqrt(sum(reshape(beta_act,M_n,1).^2,[]));
    end

    sqerr(i) = (Y(i)-Y_pred(i)).^2;
    
    if verbose
        fprintf('###### [i: %i] sqerr: %g elapsed:%f \n', i, sqerr(i), toc(stime));
    end
end


end

