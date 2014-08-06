% fista.m - Junier Oliva
% Runs accelerated generalized gradient descent
% - Parameters -
%   Y: response vector or matrix
%   K: covariate (design) matrix
%   funcs : struct with the following function handles as fields
%       .opt_msg(iter,x_curr,opts): 
%           function to prints a msg on the optimization status depending
%           on iter (the iteration), x_curr (the variable being optimized),
%           and opts (a struct of options)
%       .g(Y,K,x,opts): the smooth part of the objective evaluated at x
%       .obj(Y, K, x, opts): the entire objective evaluated at x
%       .grad_g(Y,K,x,opts): the gradient of the smooth part evaluated at x
%       .prox(x,t,opts): the prox_t(x) function of the unsmooth part of the
%           objective
%   opts : struct with options for optimization, contains atleast
%       .beta: the back-off rate for backtracking
%       .maxIter: the maximum number of iterations
%       .epsilon: the minimum relative difference in objective values to
%           consider the optimization to have converged
%       .verbose: verbose option
% - Returns -
%   x: optimizer found
%   objs: objectives at each iteration (padded with nans)
function [x, objs] = fista( Y, K, funcs, opts )
p = size(K,2);
q = size(Y,2);

% set options
beta = opts.beta;
maxIter = opts.maxIter;
epsilon = opts.epsilon;
t_k = opts.t_k;
verbose = opts.verbose;

% initialize variables
x_curr = get_opt(opts,'x_0',zeros(p,q));
objs = nan(maxIter,1);
u_curr = x_curr;
obj_curr = funcs.obj(Y,K,x_curr,opts);
objs(1) = obj_curr;

if verbose
    fprintf('# Beginning Regression with %i co-variates \n', p);
end
% main loop
for k = 1:maxIter
    if verbose
        funcs.opt_msg(k,x_curr,opts);
    end
    x_prev = x_curr;
    obj_prev = obj_curr;
    % accelaration
    theta_k = 2/(k+1);
    y = (1-theta_k)*x_curr + theta_k*u_curr;
    % backtracking
    gx = inf; acceptable = -inf; t_k = t_k/beta;
    grad_gy = funcs.grad_g(Y,K,y,opts);
    gy = funcs.g(Y,K,y,opts);
    while gx > acceptable
        t_k = beta*t_k;
        x_curr = funcs.prox(y-t_k*grad_gy,t_k,opts);
        x_minus_y = x_curr-y;
        gx = funcs.g(Y,K,x_curr,opts);
        acceptable = gy + grad_gy(:)'*x_minus_y(:)+x_minus_y(:)'*x_minus_y(:)/(2*t_k);
        if verbose
            fprintf('\t{%d}\t[g_new: %f]\t[accept: %f]\n', t_k, gx, acceptable);
        end
    end
    u_curr = x_prev+(x_curr-x_prev)/theta_k;
    % update objective    
    obj_curr = funcs.obj(Y,K,x_curr,opts);
    objs(k+1) = obj_curr;
    if verbose
        fprintf('\t{%d}\t[obj_curr: %f]\n', t_k, obj_curr);
    end
    % check for convergence
    if k>1 && abs(obj_curr-obj_prev)/abs(obj_prev)<epsilon
        break;
    end
end
x = x_curr;
    
end