function [ o_opts ] = make_glasso_opts( opts )
%MAKE_GLASSO_OPTS Summary of this function goes here
%   Detailed explanation goes here
o_opts.beta = get_opt(opts,'beta',.8);
o_opts.maxIter = get_opt(opts,'maxIter',1000);
o_opts.epsilon = get_opt(opts,'epsilon',1E-8);
o_opts.t_k = get_opt(opts,'t_0',1);
o_opts.verbose = get_opt(opts,'verbose',false);
o_opts.intercept = get_opt(opts,'intercept',false);


o_opts.lambda1 = get_opt(opts,'lambda1',0);
o_opts.lambda2 = get_opt(opts,'lambda2',0);
o_opts.lambdae = get_opt(opts,'lambdae',0);
o_opts.g_size = get_opt(opts,'g_size',1);

end

