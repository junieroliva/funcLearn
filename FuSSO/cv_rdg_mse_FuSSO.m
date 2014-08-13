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
lambdars = get_opt(opts,'lambdars',2.^(20:-1:-20));
opts.lambdars = lambdars;
nfolds = get_opt(opts,'nfolds',5);

norms = nan(N,p);
Y_pred = nan(N,1);
sqerr = nan(N,1);
lambdar = nan(N,1);
stime = tic;
for i = 1:N
    topts = opts;
    trn_set = true(N,1);
    trn_set(i) = false;
    cv_lambdar = nan(nfolds,1);
    finds = crossvalind('Kfold', N-1, nfolds);
    for trl=1:nfolds
        if verbose
            fprintf('*** [i: %i] trial: %i elapsed:%f \n', i, trl, toc(stime));
        end
        topts.trn_set = finds==trl;
        cv_lambdar(trl) = cv_rdg_lam_FuSSO( Y(trn_set), PC(trn_set,:), topts );
    end
    lambdar(i) = mean(cv_lambdar);

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
        fprintf('###### [i: %i] sqerr: %g lambdar: %g elapsed: %fs \n', i, sqerr(i), lambdar(i) , toc(stime));
    end
end


end

