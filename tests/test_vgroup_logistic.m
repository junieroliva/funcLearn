N = 10;
p = 40;
msize = 5;

% test group lasso with different sized groups, no intercept, no other
% lambdas

gsizes = randi(msize,p,1);
ginds = cumsum(gsizes);

X = randn(10,ginds(end));
Y = 1./(1+exp(-X*randn(ginds(end),1)))>.5;
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

    minimize( -Y'*Xpad*betal(:) +sum(log_sum_exp([zeros(1,N); betal(:)'*Xpad'])) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) )

    subject to
        betal(npadind) == 0
cvx_end

cvx_obj1 = -Y'*Xpad*betal(:) +sum(log_sum_exp([zeros(1,N); betal(:)'*Xpad'])) + lambda*dot(sqrt(gsizes),norms(betal,2,1));
opt_obj1 = objs(rli);
ratio1 = abs(opt_obj1-cvx_obj1)/cvx_obj1
if ratio1>1E-6
    error('Unacceptable objective ratio');
end


% test with intercept
opts = struct;
opts.verbose = false;
opts.intercept = true;
[ objs, ~, lambdas, ~, betas ] = eval_grplasso( Y, X, ginds, opts);

rli = randi(length(lambdas));
lambda = lambdas(rli);
cvx_begin quiet
    variables betal(msize,p) b(1,1)

    minimize( -Y'*(Xpad*betal(:)+b) +sum(log_sum_exp([zeros(1,N); betal(:)'*Xpad'+b])) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) )

    subject to
        betal(npadind) == 0
cvx_end

cvx_obj2 = -Y'*(Xpad*betal(:)+b) +sum(log_sum_exp([zeros(1,N); betal(:)'*Xpad'+b])) + lambda*dot(sqrt(gsizes),norms(betal,2,1));
opt_obj2 = objs(rli);
ratio2 = abs(opt_obj2-cvx_obj2)/cvx_obj2
if ratio2>1E-6
    error('Unacceptable objective ratio');
end

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

    minimize( -Y'*Xpad*betal(:) +sum(log_sum_exp([zeros(1,N); betal(:)'*Xpad'])) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) ...
                + lambda1*sum(abs(betal(:))) + (lambdae/2)*sum_square(betal(:)) )

    subject to
        betal(npadind) == 0
cvx_end

cvx_obj3 = -Y'*Xpad*betal(:) +sum(log_sum_exp([zeros(1,N); betal(:)'*Xpad'])) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) ...
                + lambda1*sum(abs(betal(:))) + (lambdae/2)*sum_square(betal(:));
opt_obj3 = objs(rli);
ratio3 = abs(opt_obj3-cvx_obj3)/cvx_obj3
if ratio3>1E-6
    error('Unacceptable objective ratio');
end

% test with l1 sparsity and elastic net lambdas, and intercept
opts.intercept = true;
[ objs, ~, lambdas, ~, betas ] = eval_grplasso( Y, X, ginds, opts);

rli = randi(length(lambdas));
lambda = lambdas(rli);
lambda1 = opts.lambda1;
lambdae = opts.lambdae;
cvx_begin quiet
    variables betal(msize,p) b(1,1)

    minimize( -Y'*(Xpad*betal(:)+b) +sum(log_sum_exp([zeros(1,N); betal(:)'*Xpad'+b])) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) ...
                + lambda1*sum(abs(betal(:))) + (lambdae/2)*sum_square(betal(:)) )

    subject to
        betal(npadind) == 0
cvx_end

cvx_obj4 = -Y'*(Xpad*betal(:)+b) +sum(log_sum_exp([zeros(1,N); betal(:)'*Xpad'+b])) + lambda*dot(sqrt(gsizes),norms(betal,2,1)) ...
                + lambda1*sum(abs(betal(:))) + (lambdae/2)*sum_square(betal(:));
opt_obj4 = objs(rli);
ratio4 = abs(opt_obj4-cvx_obj4)/cvx_obj4
if ratio4>1E-6
    error('Unacceptable objective ratio');
end
