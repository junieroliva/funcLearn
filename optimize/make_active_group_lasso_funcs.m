function [ funcs ] = make_active_group_lasso_funcs()
%MAKE_GLASSO_FUNCS Summary of this function goes here
%   Detailed explanation goes here
funcs.strong_rule = @strong_rule;
funcs.get_active_set = @get_active_set;
funcs.get_active_inds = @get_active_inds;
funcs.make_active_funcs = @make_active_funcs;
funcs.viol_kkt = @viol_kkt;

function [act_funcs, active_inds] = make_active_funcs(active,params)
    [active_inds, K_active_inds] = get_active_inds(active,params);
    act_params = params;
    act_params.K = params.K(:,K_active_inds);
    act_funcs = make_group_lasso_funcs(act_params);
end

function [srule, screen, strong_lambdas] = strong_rule(params,x_prev,lambda_prev,lambda,strong_lambdas,screen)
    srule = screen<2*lambda-strong_lambdas;
    update = ~srule & strong_lambdas~=lambda_prev;
    if any(update) 
        [x_update_inds, K_update_inds] = get_active_inds(update,params);
        up_params = params;
        up_params.K = params.K(:,K_update_inds);
        screen(update) = get_screen(x_prev(x_update_inds), up_params);
        strong_lambdas(update) = lambda_prev;
        srule(update) = screen(update)<2*lambda-strong_lambdas(update);
    end
end

function [x_active_inds, K_active_inds] = get_active_inds(active,params)
    intercept = get_opt(params,'intercept',false);
    gsize = params.gsize;
    K_active_inds = repmat(active(:)',gsize,1);
    K_active_inds = K_active_inds(:);
    if intercept
        x_active_inds = [K_active_inds; true];
    else
        x_active_inds = K_active_inds;
    end
end

function active_set = get_active_set(x_curr,params)
    intercept = get_opt(params,'intercept',false);
    gsize = params.gsize;
    if intercept
        x_curr = x_curr(1:end-1);
    end
    active_set = reshape(x_curr,gsize,[]);
    active_set = (sum(active_set.^2,1)>0)';
end

function screen = get_screen(x, params, varargin)
    Y = params.Y;
    K = params.K;
    lambdae = params.lambdae;
    lambda1 = params.lambda1;
    gsize = params.gsize;
    intercept = get_opt(params,'intercept',false);
    
    if ~isempty(varargin)
        resid = varargin{1};
    else
        if intercept
            resid = K*x(1:end-1)+x(end)-Y;
        else
            resid = K*x-Y;
        end
    end
    y_s = K'*resid;
    if intercept
        y_s = y_s+lambdae*x(1:end-1);
    else
        y_s = y_s+lambdae*x;
    end
    
    if lambda1>0
        y_s = sign(y_s).*max(0,abs(y_s)-lambda1);
    end
    
%     if isnan(gsize)
%         screen = cumsum(y_s(:).^2);
%         screen = sqrt(screen(ginds)-[0; screen(ginds(1:end-1))]);
    y_s = reshape(y_s,gsize,[]);
    screen = sqrt(sum(y_s.^2,1));
end

function viol = viol_kkt(x_curr,params,active,checklist)
    Y = params.Y;
    K = params.K;
    intercept = get_opt(params,'intercept',false);
    [x_active_inds, K_active_inds] = get_active_inds(active,params);
    x_act = x_curr(x_active_inds);
    if intercept
        if size(x_act,1)>1
            resid = K(:,K_active_inds)*x_act(1:end-1)+x_act(end)-Y;
        else
            resid = x_act(end)-Y;
        end
    else
        if ~isempty(x_act)
            resid = K(:,K_active_inds)*x_act-Y;
        else
            resid = -Y;
        end
    end
    viol_params = params;
    [x_active_inds, K_active_inds] = get_active_inds(checklist,params);
    viol_params.K = K(:,K_active_inds);
    
    viol = get_screen(x_curr(x_active_inds), viol_params, resid);
    viol = viol>params.lambda2;
end

end

