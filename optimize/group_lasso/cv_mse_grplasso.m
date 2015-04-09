function [ Y_pred, sqerr, active, betas, outfolds, ...
    lambda, lambdae, lambdar, lambdas, lambdaes, lambdars ] = ...
    cv_mse_grplasso( Y, K, g, varargin )
% Cross-validate (with inner and outer loops) the MSE of a group lasso
% problem with response Y and covariate matrix K while chosing lambdas.
%   Inputs - 
%   Outputs -
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end
N = size(K,1);

% set up groups
if length(g)==1
    if mod(size(K,2),g)~=0
        error('#covariates not disvisible by #groups indicated.');
    end
    gsize = size(K,2)/g;
    ginds = [];
    gmult = [];
    p = g;
else
    gsize = nan;
    ginds = g;
    gmult = get_opt( opts, 'gmult', sqrt(ginds-[0; ginds(1:end-1)]) );
    p = length(g);
end

% options
verbose = get_opt(opts,'verbose',false);
intercept = get_opt(opts,'intercept',true);
opts.intercept = intercept;
params.intercept = intercept;
params.gsize = gsize;
params.ginds = ginds;
params.gmult = gmult;

% get randge of lambdas
lambdas = get_opt(opts,'lambdas',[]);
if isempty(lambdas)
    groups.gsize = gsize;
    groups.ginds = ginds;
    groups.gmult = gmult;
    lambdas = get_lambda_range(Y, K, groups, opts);
end
opts.lambdas = lambdas;

% ridge lambdas and elastic-net lambda
lambdars = get_opt(opts,'lambdars',2.^(20:-1:-20));
opts.lambdars = lambdars;
lambdaes = get_opt(opts,'lambdaes',[0 4.^(1:2)]);
opts.lambdaes = lambdaes;

% folds
ninfolds = get_opt(opts,'ninfolds',5);
noutfolds = get_opt(opts,'noutfolds',N);

funcs = make_active_group_lasso_funcs();

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
        [ ~, ~, cv_lambda(trl), cv_lambdae(trl), cv_lambdar(trl) ] = cv_supp_grplasso( Y(trn_set), K(trn_set,:), g, topts );
    end
    lambdae(i) = mean(cv_lambdae);
    lambdar(i) = mean(cv_lambdar);
    [~,li] = min(abs(lambdas-mean(cv_lambda)));
    lambda(i) = lambdas(li);
    topts = opts;
    topts.lambdas = lambdas(1:li);
    topts.lambdae = lambdae(i);
    [~,gnorms] = eval_grplasso( Y(trn_set), K(trn_set,:), g, topts );
    
    active(i,:) = gnorms(end,:)>0;
    [~, supp] = funcs.get_active_inds(active(i,:),params);
    supp = supp(:)>0;
    
    nactive = sum(active(i,:));
    if nactive>0
        if intercept
            K_act = [K(trn_set,supp) ones(sum(trn_set),1)];
        else
            K_act = K(trn_set,supp);
        end
        [U,S] = eig(K_act*K_act');
        S = diag(S);
        KtU = K_act'*U;
        KtY = K_act'*Y(trn_set);
        UtKKtY = KtU'*KtY;
        beta_act = (1/lambdar(i))*(KtY-KtU*(UtKKtY./(S+lambdar(i))));
        if intercept
            cY_pred{i} = K(~trn_set,supp)*beta_act(1:end-1)+beta_act(end);
        else
            cY_pred{i} = K(~trn_set,supp)*beta_act;
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

