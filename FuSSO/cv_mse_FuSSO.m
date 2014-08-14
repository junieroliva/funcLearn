function [ Y_pred, sqerr, active, betas, outfolds, ...
    lambda, lambdae, lambdar, lambdas, lambdaes, lambdars ] = ...
    cv_mse_FuSSO( Y, PC, p, varargin )
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
lambdas = get_opt(opts,'lambdas',[]);
if isempty(lambdas)
    nlambdas = get_opt(opts,'nlambdas',100);
    min_lambda_ratio = get_opt(opts,'min_lambda_ratio',1E-2);
    if intercept
        Y_0 = Y-mean(Y);
    else
        Y_0 = Y;
    end
    max_lambda = max(sqrt(sum(reshape(PC'*Y_0,M_n,[]).^2,1)));
    b = max_lambda*min_lambda_ratio;
    B = max_lambda;
    lambdas = b*((B/b).^([(nlambdas-1):-1:0]/(nlambdas-1)));
    opts.lambdas = lambdas;
end
lambdars = get_opt(opts,'lambdars',2.^(20:-1:-20));
opts.lambdars = lambdars;
lambdaes = get_opt(opts,'lambdaes',[0 4.^(1:2)]);
opts.lambdaes = lambdaes;
ninfolds = get_opt(opts,'ninfolds',5);
noutfolds = get_opt(opts,'noutfolds',N);

cY_pred = cell(noutfolds,1);
csqerr = cell(noutfolds,1);
active = nan(noutfolds,p);
lambda = nan(noutfolds,1);
lambdae = nan(noutfolds,1);
lambdar = nan(noutfolds,1);
betas = cell(noutfolds,1);
outfolds = crossvalind('Kfold', N, noutfolds);
stime = tic;
parfor i = 1:noutfolds
    topts = opts;
    trn_set = true(N,1);
    trn_set(outfolds==i) = false;
    cv_lambda = nan(ninfolds,1);
    cv_lambdae = nan(ninfolds,1);
    cv_lambdar = nan(ninfolds,1);
    infolds = crossvalind('Kfold', sum(trn_set), ninfolds);
    for trl=1:ninfolds
        if verbose
            fprintf('*** [i: %i] trial: %i elapsed:%f \n', i, trl, toc(stime));
        end
        topts.trn_set = infolds~=trl;
        [ ~, ~, cv_lambda(trl), cv_lambdae(trl), cv_lambdar(trl) ] = cv_supp_FuSSO( Y(trn_set), PC(trn_set,:), p, topts );
    end
    lambdae(i) = mean(cv_lambdae);
    lambdar(i) = mean(cv_lambdar);
    [~,li] = min(abs(lambdas-mean(cv_lambda)));
    lambda(i) = lambdas(li);
    topts = opts;
    topts.lambdas = lambdas(1:li);
    topts.lambdae = lambdae(i);
    [~,norms] = eval_FuSSO( Y(trn_set), PC(trn_set,:), p, topts );
    active(i,:) = norms(end,:)>0;
    supp = repmat(active(i,:),M_n,1);
    supp = supp(:)>0;
    
    nactive = sum(active(i,:));
    if nactive>0
        if intercept
            PC_act = [PC(trn_set,supp) ones(sum(trn_set),1)];
        else
            PC_act = PC(trn_set,supp);
        end
        [U,S] = eig(PC_act*PC_act');
        S = diag(S);
        PCtU = PC_act'*U;
        PCtY = PC_act'*Y(trn_set);
        UtPCPCtY = PCtU'*PCtY;
        beta_act = (1/lambdar(i))*(PCtY-PCtU*(UtPCPCtY./(S+lambdar(i))));
        if intercept
            cY_pred{i} = PC(~trn_set,supp)*beta_act(1:end-1)+beta_act(end);
        else
            cY_pred{i} = PC(~trn_set,supp)*beta_act;
        end
        betas{i} = beta_act;
    else
        if intercept
            cY_pred{i} = mean(Y(trn_set)).*ones(sum(~trn_set),1) ;
        else
            cY_pred{i} = zeros(sum(~trn_set),1);
        end
    end
    csqerr{i} = (Y(~trn_set)-cY_pred{i}).^2;
    
    if verbose
        fprintf('###### [i: %i] active: %i, sqerr: %g elapsed:%f \n', i, nactive, mean(csqerr{i}), toc(stime));
        fprintf('###### [i: %i] lambda: %g, lambdae: %g lambdar:%g \n', i, lambda(i), lambdae(i), lambdar(i));
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

