function [ Y_pred, err, active, betas, outfolds, ...
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

class_prob = do_classify(Y);

% options
verbose = get_opt(opts,'verbose',false);
intercept = get_opt(opts,'intercept',true);
opts.intercept = intercept;
params.intercept = intercept;
params.gsize = gsize;
params.ginds = ginds;
params.gmult = gmult;
params.do_classify = class_prob;

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
if class_prob
    lambdars = get_opt(opts, 'lambdars', 10.^(6:-1:-3));
else
    lambdars = get_opt(opts,'lambdars',10.^(20:-1:-20));
end
opts.lambdars = lambdars;
lambdaes = get_opt(opts,'lambdaes',[0 4.^(1:2)]);
opts.lambdaes = lambdaes;

% folds
ninfolds = get_opt(opts,'ninfolds',5);
noutfolds = get_opt(opts,'noutfolds',N);

funcs = make_active_group_lasso_funcs();

cY_pred = cell(noutfolds,1);
errs = cell(noutfolds,1);
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
    if ~class_prob
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
        errs{i} = (Y(~trn_set)-cY_pred{i}).^2;
    else
        options_act = optimoptions(@fminunc,'GradObj','on','Algorithm','quasi-newton','Display','off');
        if nactive>0
            K_act = K(trn_set,supp);
            params_act = params;
            params_act.K = K_act;
            params_act.Y = Y(trn_set);
            params_act.lambda1 = 0;
            params_act.lambda2 = 0;
            beta_act = zeros(size(K_act,2)+intercept,1);
            for lr=5:-1:0
                params_act.lambdae = lambdar(i)*10^lr;
                funcs_act = make_group_lasso_funcs(params_act);
                optiLasso = @(x)multi_output(x, @(y)funcs_act.g(y), @(y)funcs_act.grad_g(y));
                beta_act = fminunc(optiLasso,beta_act,options_act);
            end
            if intercept
                cY_pred{i} = K(~trn_set,supp)*beta_act(1:end-1)+beta_act(end)>=0;
            else
                cY_pred{i} = K(~trn_set,supp)*beta_act>=0;
            end
            betas{i} = beta_act;
        else
            if intercept
                cY_pred{i} = mode(Y(trn_set)).*ones(sum(~trn_set),1) ;
            else
                cY_pred{i} = ones(sum(~trn_set),1);
            end
        end
        errs{i} = (Y(~trn_set)~=cY_pred{i});
    end
    
    if verbose
        fprintf('###### [i: %i] active: %i, err: %g elapsed:%f \n', i, nactive, mean(errs{i}), toc(stime));
        fprintf('###### [i: %i] lambda: %g, lambdae: %g lambdar:%g \n', i, lambda(i), lambdae(i), lambdar(i));
    end
end

Y_pred = nan(N,1);
err = nan(N,1);
for i = 1:noutfolds
    trn_set = true(N,1);
    trn_set(outfolds==i) = false;
    Y_pred(~trn_set) = cY_pred{i};
    err(~trn_set) = errs{i};
end

end

