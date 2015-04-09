function [ objs, gnorms, lambdas, supps, betas ] = eval_grplasso( Y, K, g, varargin )
% Optimize a group lasso problem over a range of lambda values. 
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end

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
maxactive = get_opt(opts,'maxactive',inf);
intercept = get_opt(opts,'intercept',true);

% get range of lambdas
lambdas = get_opt(opts,'lambdas',[]);
if isempty(lambdas)
    groups.gsize = gsize;
    groups.ginds = ginds;
    groups.gmult = gmult;
    lambdas = get_lambda_range(Y, K, groups, opts);
end
nlambdas = length(lambdas);

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
params.lambda1 = get_opt(opts,'lambda1',0);
params.lambdae = get_opt(opts,'lambdae',0);
params.intercept = intercept;

supps = false(nlambdas,p);
if nargout>=5
    betas = cell(1,nlambdas);
end
gnorms = nan(nlambdas,p);
objs = nan(nlambdas,1);
beta = zeros(size(K,2)+intercept,1);
stime = tic;
for l = 1:nlambdas
    params.lambda2 = lambdas(l);

    [beta,screen,strong_lambdas,o] = fista_active(beta, funcs, lambdas(max(l-1,1)), lambdas(l), strong_lambdas, screen, params, cv_opts);
    if intercept
        g_norms = group_norms(beta(1:end-1),gsize,ginds);
    else
        g_norms = group_norms(beta,gsize,ginds);
    end
    supps(l,:) = g_norms>0;
    nactive = sum(supps(l,:));
    
    gnorms(l,:) = g_norms;
	objs(l) = o(end);
    
    if nargout>=5
        betas{l} = sparse(beta);
    end
    
    if verbose
        fprintf('lambda: %f, obj: %f, active: %i, elapsed:%f \n', lambdas(l), objs(l), nactive, toc(stime));
    end
    
    if nactive>maxactive
        break;
    end
end
gnorms = gnorms(1:l,:);
objs = objs(1:l);

end

