%%%%%%%%%%%%%
% Parameters
%%%%%%%%%%%%%
% number of basis, support, dims
s = 15;
p = 10000;
N = 500;
std_y = .1;
% multiplier for lambda_e (0 for no elastic)
if ~exist('l_mult','var')
    l_mult = 0;
end
% multiplier for lambda_e (0 for no elastic)
if ~exist('intercept','var')
    intercept = true;
end
X = rand(N,p);
beta = [randn(s,1); zeros(p-s,1)];
if intercept
    beta_hat = randn;
else
    beta_hat = 0;
end
Y = X*beta+beta_hat+std_y*randn(N,1);

%%%%%%%%%%%%%%%%%
% Get objectives 
%%%%%%%%%%%%%%%%%
lambdae = l_mult*rand*max(sqrt(sum(reshape(X'*(Y-mean(Y)),1,[]).^2,1)));
fopts.verbose = true;
fopts.lambdae = lambdae;
fopts.intercept = intercept;
[objs,onorms,lambdas] = eval_FuSSO(Y,X,p,fopts);
% plot group norms
figure;
h1 = plot(lambdas,onorms(:,1:s),'r');
hold on;
h2 = plot(lambdas,onorms(:,s+1:end),'b');
legend([h1(1) h2(1)],{'Support','Non-support'});
ylabel('Group Norm')
xlabel('Lambda')
title('Group Norms vs. Lambda');

%%%%%%%%%%%%%%%%%
% Check with CVX
%%%%%%%%%%%%%%%%%
if exist('cvx_begin')
    % number of lambdas to check
    if ~exist('ncheck','var')
        ncheck = 3;
    end
    lambda_rprm = randperm(length(lambdas),ncheck);
    cvx_obj = nan(ncheck,1);
    obj_diff = nan(ncheck,1);
    obj_diff_rel = nan(ncheck,1);
    for li=1:ncheck
        stime = tic;
        lambda = lambdas(lambda_rprm(li));
        if intercept
            cvx_begin
                variables beta_hat(p,1) beta_0_hat
                minimize( .5*sum_square(Y - X*beta_hat(:) - beta_0_hat) + lambda*norm(beta_hat,1) + .5*lambdae*(beta_hat(:)'*beta_hat(:)) )
            cvx_end
        else
            beta_0_hat = 0;
            cvx_begin
                variables beta_hat(p,1)
                minimize( .5*sum_square(Y - X*beta_hat(:) - beta_0_hat) + lambda*norm(beta_hat,1) + .5*lambdae*(beta_hat(:)'*beta_hat(:)) )
            cvx_end
        end
        cvx_obj(li) = .5*sum_square(Y - X*beta_hat(:) - beta_0_hat) + lambda*norm(beta_hat,1) + .5*lambdae*(beta_hat(:)'*beta_hat(:));
        obj_diff(li) = abs(objs(lambda_rprm(li))-cvx_obj(li));
        obj_diff_rel(li) = obj_diff(li)/cvx_obj(li);
        fprintf('\t# lambda:%g, obj_diff: %g, obj_diff_rel: %g, elapsed:%f \n', lambda, obj_diff(li), obj_diff_rel(li), toc(stime));
    end
end