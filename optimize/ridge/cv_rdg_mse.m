function [ Y_pred, sqerr, norms, outfolds, lambdar, lambdars ] = cv_rdg_mse( Y, PC, p, varargin )
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
ninfolds = get_opt(opts,'ninfolds',5);
noutfolds = get_opt(opts,'noutfolds',10);

cY_pred = cell(N,1);
csqerr = cell(N,1);
norms = nan(noutfolds,p);
lambdar = nan(noutfolds,1);
outfolds = crossvalind('Kfold', N, noutfolds);
stime = tic;
parfor i = 1:noutfolds
    topts = opts;
    
    trn_set = true(N,1);
    trn_set(outfolds==i) = false;
    cv_lambdar = nan(ninfolds,1);
    cv_MSE = nan(ninfolds,1);
    cv_MSEs = nan(ninfolds,length(lambdars));
    infolds = crossvalind('Kfold', sum(trn_set), ninfolds);
    for trl=1:ninfolds
        if verbose
            fprintf('*** [i: %i] trial: %i elapsed:%f \n', i, trl, toc(stime));
        end
        topts.trn_set = infolds~=trl;
        [cv_lambdar(trl), ~, cv_MSE(trl), cv_MSEs(trl,:)] = cv_rdg_lam( Y(trn_set), PC(trn_set,:), topts );
    end
    lambdar(i) = mean(cv_lambdar);

    if intercept
        PC_act = [PC(trn_set,:) ones(sum(trn_set),1)];
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
        cY_pred{i} = PC(~trn_set,:)*beta_act(1:end-1)+beta_act(end);
        norms(i,:) = sqrt(sum(reshape(beta_act(1:end-1),M_n,[]).^2,1));
    else
        cY_pred{i} = PC(~trn_set,:)*beta_act;
        norms(i,:) = sqrt(sum(reshape(beta_act,M_n,[]).^2,1));
    end

    csqerr{i} = (Y(~trn_set)-cY_pred{i}).^2;
    
    if verbose
        fprintf('###### [i: %i] sqerr: %g lambdar: %g elapsed: %fs \n', i, mean(csqerr{i}), lambdar(i) , toc(stime));
    end
end

Y_pred = nan(N,1);
sqerr = nan(N,1);
for i = 1:noutfolds
    trn_set = true(N,1);
    trn_set(outfolds==i) = false;
    Y_pred(~trn_set) = cY_pred{i};
    sqerr(~trn_set) = csqerr{i};
end

end

