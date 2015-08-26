function [x, screen, strong_lambdas, obj] = fista_active( x_0, funcs, lambda_prev, lambda, strong_lambdas, screen, params, varargin )
% set options
x_curr = x_0;
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end
% screen strong rule, get current active set
[srule, screen, strong_lambdas] = funcs.strong_rule(params,x_curr,lambda_prev,lambda,strong_lambdas,screen);
active = funcs.get_active_set(x_curr,params);
% main loop
first_opt = true;
viol = false(size(active));
while first_opt || any(viol) 
    first_opt = false;
    % add violators
    active = active | viol;
    viol = false(size(active));
    % optimize over the current active set
    [act_funcs, active_inds] = funcs.make_active_funcs(active,params);
    if sum(active_inds)>0
        [x_curr(active_inds),obj] = fista(x_curr(active_inds), act_funcs, opts);
    else
        obj = act_funcs.obj(x_curr(active_inds));
    end
    % check for kkt violators in covariates that did not pass strong rule
    checklist = ~active(:) & ~srule(:);
    if any(checklist)
        viol(checklist) = funcs.viol_kkt(x_curr,params,active,checklist);
    end
    % check for kkt violators in covariates that passed strong rule
    checklist = ~active(:) & srule(:);
    if ~any(viol) && any(checklist)
        viol(checklist) = funcs.viol_kkt(x_curr,params,active,checklist);
    end
end
x = x_curr;
end