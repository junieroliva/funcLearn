function [ active, supp, lambda, lambdae, lambdar, lambdas, lambdaes, lambdars ] = ...
    cv_supp_grplasso( Y, K, g, varargin )
% Cross-validate the support of a group lasso problem with response Y and 
% covariate matrix K while chosing lambdas.
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

% get randge of lambdas
lambdas = get_opt(opts,'lambdas',[]);
if isempty(lambdas)
    groups.gsize = gsize;
    groups.ginds = ginds;
    groups.gmult = gmult;
    lambdas = get_lambda_range(Y, K, groups, opts);
end
nlambdas = length(lambdas);

% options
verbose = get_opt(opts,'verbose',false);
intercept = get_opt(opts,'intercept',true);
maxactive = get_opt(opts,'maxactive',inf);

% ridge lambdas and elastic-net lambda
lambdars = get_opt(opts,'lambdars',10.^(15:-1:-15));
nlambdars = length(lambdars);
lambdaes = get_opt(opts,'lambdaes',[0 4.^(1:2)]);
nlambdaes = length(lambdaes);

% get training/hold-out sets
trn_set = get_opt(opts,'trn_set',[]);
if isempty(trn_set)
    trn_perc = get_opt(opts,'trn_perc',.9);
    trn_set = false(N,1);
    trn_set(randperm(N,ceil(N*trn_perc))) = true;
end
N_trn = sum(trn_set);
N_hol = sum(~trn_set);
K_hol = K(~trn_set,:);
Y_hol = Y(~trn_set);
K = K(trn_set,:);
Y = Y(trn_set);

% set opti params
cv_opts = struct;
cv_opts.maxIter = get_opt(opts,'maxIter',50000);
cv_opts.epsilon = get_opt(opts,'epsilon',1E-10);
cv_opts.accel = true;
cv_opts.verbose = false;

funcs = make_active_group_lasso_funcs();
screen = inf(p,1);
strong_lambdas = inf(p,1);
params.K = K;
params.Y = Y;
params.gsize = gsize;
params.ginds = ginds;
params.gmult = gmult;
params.lambda1 = 0;
params.intercept = intercept;

best_hol_MSE = inf;
best_active = [];
best_supp = [];
best_lambda = nan;
best_lambdar = nan;
best_lambdae = nan;
stime = tic;
for le = 1:nlambdaes
    params.lambdae = lambdaes(le);
    beta = zeros(size(K,2)+intercept,1);
    for l = 1:nlambdas
        params.lambda2 = lambdas(l);

        [beta,screen,strong_lambdas] = fista_active(beta, funcs, lambdas(max(l-1,1)), lambdas(l), strong_lambdas, screen, params, cv_opts);
        if intercept
            g_norms = group_norms(beta(1:end-1),gsize,ginds);
        else
            g_norms = group_norms(beta,gsize,ginds);
        end
        gactive = g_norms>0;
        nactive = sum(gactive);
        [~, active] = funcs.get_active_inds(gactive,params);
        active = active(:);

        % get ridge estimates using found support -- fast for fat matrices
        best_hol_MSE_r = inf;
        best_lambdar_r = nan;
        if nactive>0
            if intercept
                K_act = [K(:,active) ones(N_trn,1)];
                K_hol_act = [K_hol(:,active) ones(N_hol,1)];
            else
                K_act = K(:,active);
                K_hol_act = K_hol(:,active);
            end
            [U,S] = eig(K_act*K_act');
            S = diag(S);
            KtU = K_act'*U;
            KtY = K_act'*Y;
            UtKKtY = KtU'*KtY;
            hol_MSEs = nan(nlambdars,1);
            for lr=1:nlambdars
                lambdar = lambdars(lr);
                %beta_act = (1/lambdar)*(Ig-PC_act'*U*diag(1./(S+lambdar))*U'*PC_act)*(PC_act'*Y);
                beta_act = (1/lambdar)*(KtY-KtU*(UtKKtY./(S+lambdar)));
                hol_MSE = mean( (Y_hol-K_hol_act*beta_act).^2 );
                hol_MSEs(lr) = hol_MSE;
                if hol_MSE<best_hol_MSE_r
                    best_hol_MSE_r = hol_MSE;
                    best_lambdar_r = lambdars(lr);
                end
                if hol_MSE<best_hol_MSE
                    best_hol_MSE = hol_MSE;
                    best_active = gactive;
                    best_supp = active;
                    best_lambda = lambdas(l);
                    best_lambdar = lambdars(lr);
                    best_lambdae = lambdaes(le);
                end
            end
        else
            if intercept
                best_hol_MSE_r = mean((Y_hol-beta(end)).^2);
            else
                best_hol_MSE_r = mean(Y_hol.^2);
            end
            if best_hol_MSE_r<best_hol_MSE
                best_hol_MSE = best_hol_MSE_r;
                best_active = gactive;
                best_supp = active;
                best_lambda = lambdas(l);
                best_lambdar = max(lambdars);
                best_lambdae = lambdaes(le);
            end
        end
        
        if verbose
            fprintf('[l:%g, lr:%g, le:%g] active: %i, hol_mse: %g elapsed:%f \n', lambdas(l), best_lambdar_r, lambdaes(le), nactive, best_hol_MSE_r, toc(stime));
        end

        if nactive>maxactive
            break;
        end
    end
end
active = best_active;
supp = best_supp;
lambda = best_lambda;
lambdar = best_lambdar;
lambdae = best_lambdae;

end

