%%%%%%%%%%%%%
% Parameters
%%%%%%%%%%%%%
% number of basis, support, dims
M = 50;
M_n = M;
s = 5;
nelas = 5;
p = 1000;
N = 500;
n = 50;
std_y = .1;
std_f = .1;
% multiplier for lambda_e (0 for no elastic)
l_mults = [0 1/8 .25 .5 1 2 4 8];
nl_mults = length(l_mults);

% multiplier for lambda_e (0 for no elastic)
if ~exist('intercept','var')
    intercept = false;
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
beta_0 = 0;
% get true A matrix
A = nan(N,M*p);
for i=1:N
    alpha = rand(M,s);
    alpha = sqrt(bsxfun(@times,alpha,1./c_k2));
    alpha = bsxfun(@times,alpha,1./sqrt(sum(alpha.^2)));
    alpha = alpha.*(2*((rand(size(alpha))>.5)-.5));
    alpha_base = alpha;
    
    for j=1:nelas-1
        alpha = [alpha, bsxfun(@times,alpha_base,1+std_f*randn(1,s))];
    end
    
    alpha_nsupp = rand(M,p-nelas*s);
    alpha_nsupp = sqrt(bsxfun(@times,alpha_nsupp,1./c_k2));
    alpha_nsupp = bsxfun(@times,alpha_nsupp,1./sqrt(sum(alpha_nsupp.^2)));
    alpha_nsupp = alpha_nsupp.*(2*((rand(size(alpha_nsupp))>.5)-.5));
    
    alpha = [alpha, alpha_nsupp];
    A(i,:) = alpha(:);
end
% get response
Y = A*beta+beta_0+std_y*randn(N,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate noise function observations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evaluate basis functions at grid
%phi_k = @(x,k) (k==1)+(k>1)*(mod(k,2)==0)*sqrt(2)*cos(2*pi*(k/2)*x)+(k>1)*(mod(k,2)~=0)*sqrt(2)*sin(2*pi*((k-1)/2)*x);
phi_k = @(x,k) (k==1)+(k>1)*sqrt(2)*cos(pi*(k-1)*x);
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
%[tA,B_cv] = get_multi_pc_reg(f,lin_x',opts);
f = permute(f, [3 1 2]);
[tA, inds_cv, B_cv, fhat] = osfe(lin_x', f);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot an example true input func / noisy observation / estimated func
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = size(B_cv,1);
% plot one function
figure;
plot(lin_x, alpha(:,1)'*Phi);
hold on;
scatter(lin_x, f(end,:,1));
plot(lin_x, tA(end,1:m)*Phi(1:m,:),'r');
title('Example Input Function');
legend('Truth','Noisy Grid', 'Estimate');
xlabel('x')
ylabel('f(x)')
clear A;

%%%%%%%%%%%%%%%%%
% Get objectives 
%%%%%%%%%%%%%%%%%
M_cv = length(inds_cv);
colors = distinguishable_colors(s+1);
all_plambdas = [];
all_alphas = [];
all_adiff = [];
all_bdiff = [];
all_ubound_loose = [];
all_ubound = [];
figure;
for i=1:nl_mults
    lambdae = l_mults(i)*max(sqrt(sum(reshape(tA'*(Y-mean(Y)),m,[]).^2,1)));
    fopts.verbose = true;
    fopts.lambdae = lambdae;
    fopts.intercept = intercept;
    %[objs,onorms,lambdas] = eval_FuSSO(Y,tA,p,fopts);
    [objs,onorms,lambdas,supps,betas] = eval_grplasso(Y,tA,p,fopts);
    % plot group norms
    hs = nan(s,1);
    gnames = cell(s,1);
    subplot(1,nl_mults,i);
    for j=0:(s-1)
        h = plot( lambdas, onorms(:,mod(1:nelas*s,s)==j), 'color', colors(j+1,:) );
        hs(j+1) = h(1);
        gnames{j+1} = sprintf('G%g',j);
        hold on;
    end
    h2 = plot(lambdas,onorms(:,nelas*s+1:end),'color', colors(end,:));
    %legend([hs(:)', h2(1)],[gnames{:}, {'Non-support'}]);
    ylabel('Group Norm')
    xlabel('Lambda')
    title(sprintf('Lambda_e=%g',lambdae));
    
    betas = full(cat(2,betas{:}));
    if intercept
        Ypreds = [tA ones(N,1)]*betas;
    else
        Ypreds = tA*betas;
    end
    resids = bsxfun(@minus,Ypreds,Y);
    rMSEs = sqrt(mean(resids.^2,1));
    plambdas = (lambdas+lambdae)/N;
    alphas = lambdas./(lambdas+lambdae);
    for j=0:(s-1)
        jinds = repmat(mod(1:nelas*s,s)==j,M_cv,1);
        Aj = tA(:,jinds(:));
        betasj = betas(jinds(:),:);
        for k=1:nelas
            Ak = Aj(:,(k-1)*M_cv+1:k*M_cv);
            betak = betasj((k-1)*M_cv+1:k*M_cv,:);
            ubetak = bsxfun(@times,betak,1./sqrt(sum(betak.^2,1)));
            abetak = bsxfun(@times,alphas,ubetak)+bsxfun(@times,1-alphas,betak);
            for l=k+1:nelas
                Al = Aj(:,(l-1)*M_cv+1:l*M_cv);
                betal = betasj((l-1)*M_cv+1:l*M_cv,:);
                ubetal = bsxfun(@times,betal,1./sqrt(sum(betal.^2,1)));
                abetal = bsxfun(@times,alphas,ubetal)+bsxfun(@times,1-alphas,betal);
                
                norm_kl = norm(Ak-Al);
                ubound_loose = norm_kl*rMSEs./plambdas;
                rhs = bsxfun(@times, (Ak-Al)'*resids, 1./(N*plambdas));
                lhs = abetak-abetal;
                adiff = sqrt(sum(lhs.^2));
                bdiff = sqrt(sum((betak-betal).^2));
                ubound = sqrt(sum(rhs.^2));
                
                both_supp = ~isnan(adiff);
                all_plambdas = [all_plambdas, plambdas(both_supp)];
                all_alphas = [all_alphas, alphas(both_supp)];
                all_adiff = [all_adiff, adiff(both_supp)];
                all_bdiff = [all_bdiff, bdiff(both_supp)];
                all_ubound_loose = [all_ubound_loose, ubound_loose(both_supp)];
                all_ubound = [all_ubound, ubound(both_supp)];
            end
        end
        
    end
end

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
                variables beta_hat(m,p) beta_0_hat
                minimize( .5*sum_square(Y - tA*beta_hat(:) - beta_0_hat) + lambda*sum( norms(beta_hat) ) + .5*lambdae*(beta_hat(:)'*beta_hat(:)) )
            cvx_end
        else
            beta_0_hat = 0;
            cvx_begin
                variables beta_hat(m,p)
                minimize( .5*sum_square(Y - tA*beta_hat(:) - beta_0_hat) + lambda*sum( norms(beta_hat) ) + .5*lambdae*(beta_hat(:)'*beta_hat(:)) )
            cvx_end
        end
        cvx_obj(li) = .5*sum_square(Y - tA*beta_hat(:) - beta_0_hat) + lambda*sum( norms(beta_hat) ) + .5*lambdae*(beta_hat(:)'*beta_hat(:));
        obj_diff(li) = abs(objs(lambda_rprm(li))-cvx_obj(li));
        obj_diff_rel(li) = obj_diff(li)/cvx_obj(li);
        fprintf('\t# lambda:%g, obj_diff: %g, obj_diff_rel: %g, elapsed:%f \n', lambda, obj_diff(li), obj_diff_rel(li), toc(stime));
    end
end