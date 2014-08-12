function [ objs, norms, lambdas ] = eval_FuSSO( Y, PC, p, varargin )
%EVAL_FUSSO Summary of this function goes here
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
else
    nlambdas = length(lambdas);
end
maxactive = get_opt(opts,'maxactive',inf);
% set opti params
cv_opts = struct;
cv_opts.maxIter = 50000;
cv_opts.epsilon = 1E-10;
cv_opts.accel = true;
cv_opts.verbose = false;

funcs = make_active_group_lasso_funcs();
screen = inf(p,1);
strong_lambdas = inf(p,1);
params.K = PC;
params.Y = Y;
params.gsize = M_n;
params.lambda1 = 0;
params.lambdae = get_opt(opts,'lambdae',0);
params.intercept = intercept;

norms = nan(nlambdas,p);
objs = nan(nlambdas,1);
tt_a = zeros(size(PC,2)+intercept,1);
for l = 1:nlambdas
	stime = tic;
    params.lambda2 = lambdas(l);

    [tt_a,screen,strong_lambdas,o] = fista_active(tt_a, funcs, lambdas(max(l-1,1)), lambdas(l), strong_lambdas, screen, params, cv_opts);
    if intercept
        tt_norms = sqrt(sum(reshape(tt_a(1:end-1),M_n,[]).^2,1));
    else
        tt_norms = sqrt(sum(reshape(tt_a,M_n,[]).^2,1));
    end
    nactive = sum(tt_norms>0);
    
    norms(l,:) = tt_norms;
	objs(l) = o(end);
    
    if verbose
        fprintf('lambda: %f, obj: %f, active: %i, elapsed:%f \n', lambdas(l), objs(l), nactive, toc(stime));
    end
    
    if nactive>maxactive
        break;
    end
end
norms = norms(1:l,:);
objs = objs(1:l);
end

