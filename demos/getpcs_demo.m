% Script that illustrates how to get projection coefficients for functions
% under various different types of observation types using osfe.m
% MAKE SURE THAT GRID POINTS ARE IN [0,1]^d, IE project them if needed.

%%
% Generate the data, you can largely ignore this section.
% number of basis, support, dims
M = 40;
p = 10;
N = 15;
n = 50;
std_f = .05;
% smoothness parameters
c_k = (1:M)';
c_k(3:2:M) = c_k(3:2:M)-1;
c_k2 = c_k.^2;
% get true A matrix
A = nan(M,p,N);
for i=1:N
    alpha = rand(M,p);
    alpha = sqrt(bsxfun(@times,alpha,1./c_k2));
    alpha = bsxfun(@times,alpha,1./sqrt(sum(alpha.^2)));
    alpha = alpha.*(2*((rand(size(alpha))>.5)-.5));
    A(:,:,i) = alpha;
end
% fixed design
x_fixed = (1:n)'/n;
phix_fixed = eval_basis(x_fixed, (0:(M-1))');
y_fixed = permute(mtimesx(phix_fixed, A), [3, 1, 2]);
y_fixed = y_fixed + std_f*randn(size(y_fixed));
% each of p functions has fixed design across instances
x_pfixed = cell(1,p);
y_pfixed = cell(1,p);
for i=1:p
    x_pfixed{i} = rand(n+randi(10),1);
    phix = eval_basis(x_pfixed{i}, (0:(M-1))');
    y_pfixed{i} = mtimesx(phix, squeeze(A(:,i,:)))';
    y_pfixed{i} = y_pfixed{i} + std_f*randn(size(y_pfixed{i}));
end
i = randi(N);
j = randi(p);

%%
% Scenario 1 - fix design for all functions, we have:
%   x: n * d matrix of grid points (n is #points, d is #dims, which should
%      be no more than around 3) in [0,1]^d.
%   y: N * n * p tensor of function values (N is #instances, p is
%      #functions). That is, we have multiple instances of p functions,
%      each observed on the points x.
% by defualt [pcs, inds] = osfe(x, y) returns:
%   inds: M * d matrix of indices of set of basis functions (chosen with CV).
%   pcs: N * Mp matrix of the projection coefficients (concat for all
%        p functions)
x = x_fixed; 
y = y_fixed;
[pcs_fixed, inds_fixed] = osfe(x, y);

% plot one of the functions from estimated projection coefficients
x_fine = (0:100)'/100;
x_ij = x_fixed;
phix_fine = eval_basis(x_fine, inds_fixed); % matrix of basis function evaluations on grid
M_ij = size(inds_fixed,1);
pcs_ij = pcs_fixed(i,(j-1)*M_ij+1:j*M_ij)'; % proj coefs of ith instance's jth fucntion
yhat_ij = phix_fine*pcs_ij;
y_ij = y_fixed(i,:,j)';
% plot
figure;
scatter(x_ij,y_ij);
hold on;
plot(x_fine, yhat_ij);
legend('Observed','Prediction');
xlabel('x');
ylabel('y');
title('Fixed Grid Example');

%%
% Scenario 2 - fix design according to function index, we have:
%   x: 1 * p cell where x{j} is n_j * d matrix of grid points for jth function
%   y: 1 * p cell where y{j} is N * n_j matrix of function values for jth function
% by defualt [pcs, inds] = osfe(x, y) returns:
%   inds: 1 * p cell where inds{j} is M_j * d matrix of indices of set of 
%         basis functions (chosen with CV).
%   pcs: N * (sum_j M_j) matrix of the projection coefficients (concat for all
%        p functions)
x = x_pfixed; 
y = y_pfixed;
[pcs_pfixed, inds_pfixed] = osfe(x, y);

% plot one of the functions from estimated projection coefficients
x_ij = x_pfixed{j};
phix_fine = eval_basis(x_fine, inds_pfixed{j}); % matrix of basis function evaluations on grid
M_ij = size(inds_pfixed{j},1);
M_s = sum(cellfun(@(C)size(C,1),inds_pfixed(1:j-1)));
pcs_ij = pcs_pfixed(i,M_s+1:M_s+M_ij)'; % proj coefs of ith instance's jth fucntion
yhat_ij = phix_fine*pcs_ij;
y_ij = y_pfixed{j}(i,:)';
% plot
figure;
scatter(x_ij,y_ij);
hold on;
plot(x_fine, yhat_ij);
legend('Observed','Prediction');
xlabel('x');
ylabel('y');
title('Fixed Grid By Function Example');
