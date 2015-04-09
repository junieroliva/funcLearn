N = 10;
p = 40;
msz = 5;

% test group lasso with different sized groups, no intercept, no other
% lambdas

gsizes = randi(msz,p,1);
ginds = cumsum(gsizes);

X = randn(10,ginds(end));
Y = X*randn(ginds(end),1);
opts = struct;
opts.verbose = false;
opts.intercept = false;
[ objs, ~, lambdas, ~, betas ] = eval_grplasso( Y, X, ginds, opts);

rli = randi(length(lambdas));
lambda = lambdas(rli);
padind = false(msize,p);
for i=1:p
    padind(1:gsizes(i),i) = true;
end
Xpad = zeros(N,msize*p);
Xpad(:,padind) = X;
npadind = ~padind;

cvx_begin quiet
    variable betal(msize,p)

    minimize( .5*sum(sum_square(Y-Xpad*betal(:))) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) )

    subject to
        betal(npadind) == 0
cvx_end

cvx_obj1 = .5*sum(sum_square(Y-Xpad*betal(:))) + lambda*dot(sqrt(gsizes),norms(betal,2,1));
opt_obj1 = objs(rli);
abs(opt_obj1-cvx_obj1)/cvx_obj1


% test with intercept
opts = struct;
opts.verbose = false;
opts.intercept = true;
[ objs, ~, lambdas, ~, betas ] = eval_grplasso( Y, [X, ones(N,1)], ginds, opts);

rli = randi(length(lambdas));
lambda = lambdas(rli);
cvx_begin quiet
    variables betal(msize,p) b(1,1)

    minimize( .5*sum(sum_square(Y-Xpad*betal(:)-b)) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) )

    subject to
        betal(npadind) == 0
cvx_end

cvx_obj2 = .5*sum(sum_square(Y-Xpad*betal(:)-b)) + lambda*dot(sqrt(gsizes),norms(betal,2,1));
opt_obj2 = objs(rli);
abs(opt_obj2-cvx_obj2)/cvx_obj2


% test with l1 sparsity and elastic net lambdas
opts = struct;
opts.verbose = false;
opts.intercept = false;
opts.lambda1 = rand*lambda;
opts.lambdae = rand*lambda;
[ objs, ~, lambdas, ~, betas ] = eval_grplasso( Y, X, ginds, opts);

rli = randi(length(lambdas));
lambda = lambdas(rli);
lambda1 = opts.lambda1;
lambdae = opts.lambdae;
cvx_begin quiet
    variables betal(msize,p)

    minimize( .5*sum(sum_square(Y-Xpad*betal(:))) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) ...
                + lambda1*sum(abs(betal(:))) + (lambdae/2)*sum_square(betal(:)) )

    subject to
        betal(npadind) == 0
cvx_end

cvx_obj3 = .5*sum(sum_square(Y-Xpad*betal(:))) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) ...
                + lambda1*sum(abs(betal(:))) + (lambdae/2)*sum_square(betal(:));
opt_obj3 = objs(rli);
abs(opt_obj3-cvx_obj3)/cvx_obj3

% test with l1 sparsity and elastic net lambdas, and intercept
opts.intercept = true;
[ objs, ~, lambdas, ~, betas ] = eval_grplasso( Y, [X, ones(N,1)], ginds, opts);

rli = randi(length(lambdas));
lambda = lambdas(rli);
lambda1 = opts.lambda1;
lambdae = opts.lambdae;
cvx_begin quiet
    variables betal(msize,p) b(1,1)

    minimize( .5*sum(sum_square(Y-Xpad*betal(:)-b)) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) ...
                + lambda1*sum(abs(betal(:))) + (lambdae/2)*sum_square(betal(:)) )

    subject to
        betal(npadind) == 0
cvx_end

cvx_obj4 = .5*sum(sum_square(Y-Xpad*betal(:)-b)) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) ...
                + lambda1*sum(abs(betal(:))) + (lambdae/2)*sum_square(betal(:));
opt_obj4 = objs(rli);
abs(opt_obj4-cvx_obj4)/cvx_obj4