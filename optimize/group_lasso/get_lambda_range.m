function lambdas = get_lambda_range(Y, K, groups, opts)

gsize = groups.gsize;
ginds = groups.ginds;
gmult = groups.gmult;

nlambdas = get_opt(opts,'nlambdas',100);
min_lambda_ratio = get_opt(opts,'min_lambda_ratio',.033);
intercept = get_opt(opts,'intercept',true);
if intercept
    Y_0 = Y-mean(Y);
else
    Y_0 = Y;
end

resid_norms = group_norms(K'*Y_0, gsize, ginds);
if isempty(ginds)
    max_lambda = max(resid_norms);
else
    max_lambda = max(resid_norms./gmult);
end
b = max_lambda*min_lambda_ratio;
B = max_lambda;
lambdas = b*((B/b).^( ((nlambdas-1):-1:0)/(nlambdas-1)) );

end