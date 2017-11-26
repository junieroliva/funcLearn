%%%%%%%%%%%%%
% Parameters
%%%%%%%%%%%%%
% number of basis, support, dims
M = 50;
M_n = M;
s = 5;
p = 1000;
N = 500;
n = 50;
std_y = .5;
std_f = .5;
% multiplier for lambda_e (0 for no elastic)
if ~exist('l_mult','var')
    l_mult = 0;
end
% multiplier for lambda_e (0 for no elastic)
if ~exist('intercept','var')
    intercept = true;
end
% smoothness parameters
c_k = (1:M)';
c_k(3:2:M) = c_k(3:2:M)-1;
c_k2 = c_k.^2;
% generate beta 
beta = rand(M,s);
beta = sqrt(bsxfun(@times,beta.^2,1./c_k2));
beta = bsxfun(@times,beta,1./sqrt(sum(beta.^2)));
beta = beta.*(2*((rand(size(beta))>.5)-.5));
beta = [beta zeros(M,p-s)];
beta = beta(:);
beta_0 = rand;
% get true A matrix
A = nan(N,M*p);
for i = 1:N
    alpha = rand(M,p);
    alpha = sqrt(bsxfun(@times,alpha,1./c_k2));
    alpha = bsxfun(@times,alpha,1./sqrt(sum(alpha.^2)));
    alpha = alpha.*(2*((rand(size(alpha))>.5)-.5));
    A(i,:) = alpha(:);
end
% get response
Y = A*beta+beta_0+std_y*randn(N,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate noise function observations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evaluate basis functions at grid
phi_k = @(x,k) (k==1)+(k>1)*(mod(k,2)==0)*sqrt(2)*cos(2*pi*(k/2)*x)+(k>1)*(mod(k,2)~=0)*sqrt(2)*sin(2*pi*((k-1)/2)*x);
lin_x = (1:n)/n;
Phi = ones(M,n);
for k=2:M
    Phi(k,:) = phi_k(lin_x,k);
end
f = nan(n,p,N);
for i=1:N
    alpha = reshape(A(i,:)',M,[]);
    for j=1:p
        % generate noisy grid points
        f(:,j,i) = alpha(:,j)'*Phi+ std_f*randn(1,n);
    end
end
opts.do_cv = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get matrix of estimated projection coefficients, CV # of basis funcs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[tA,B_cv] = get_multi_pc_reg(f,lin_x',opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot an example true input func / noisy observation / estimated func
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = size(B_cv,1);
% plot one function
figure;
plot(lin_x, alpha(:,1)'*Phi);
hold on;
scatter(lin_x, f(:,1,end));
plot(lin_x, tA(end,1:m)*Phi(1:m,:),'r');
title('Example Input Function');
legend('Truth','Noisy Grid', 'Estimate');
xlabel('x')
ylabel('f(x)')
clear A;

%%%%%%%%%%%%%%%%%
% Get objectives 
%%%%%%%%%%%%%%%%%
lambdae = l_mult*rand*max(sqrt(sum(reshape(tA'*(Y-mean(Y)),m,[]).^2,1)));
fopts.verbose = true;
fopts.lambdae = lambdae;
fopts.intercept = intercept;
%[objs,onorms,lambdas] = eval_FuSSO(Y,tA,p,fopts);
[objs,onorms,lambdas] = eval_grplasso(Y,tA,p,fopts);
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
% if exist('cvx_begin')
%     % number of lambdas to check
%     if ~exist('ncheck','var')
%         ncheck = 3;
%     end
%     lambda_rprm = randperm(length(lambdas),ncheck);
%     cvx_obj = nan(ncheck,1);
%     obj_diff = nan(ncheck,1);
%     obj_diff_rel = nan(ncheck,1);
%     for li=1:ncheck
%         stime = tic;
%         lambda = lambdas(lambda_rprm(li));
%         if intercept
%             cvx_begin
%                 variables beta_hat(m,p) beta_0_hat
%                 minimize( .5*sum_square(Y - tA*beta_hat(:) - beta_0_hat) + lambda*sum( norms(beta_hat) ) + .5*lambdae*(beta_hat(:)'*beta_hat(:)) )
%             cvx_end
%         else
%             beta_0_hat = 0;
%             cvx_begin
%                 variables beta_hat(m,p)
%                 minimize( .5*sum_square(Y - tA*beta_hat(:) - beta_0_hat) + lambda*sum( norms(beta_hat) ) + .5*lambdae*(beta_hat(:)'*beta_hat(:)) )
%             cvx_end
%         end
%         cvx_obj(li) = .5*sum_square(Y - tA*beta_hat(:) - beta_0_hat) + lambda*sum( norms(beta_hat) ) + .5*lambdae*(beta_hat(:)'*beta_hat(:));
%         obj_diff(li) = abs(objs(lambda_rprm(li))-cvx_obj(li));
%         obj_diff_rel(li) = obj_diff(li)/cvx_obj(li);
%         fprintf('\t# lambda:%g, obj_diff: %g, obj_diff_rel: %g, elapsed:%f \n', lambda, obj_diff(li), obj_diff_rel(li), toc(stime));
%     end
% end