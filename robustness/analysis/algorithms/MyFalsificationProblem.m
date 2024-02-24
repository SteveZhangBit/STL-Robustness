classdef MyFalsificationProblem < FalsificationProblem
properties
    dev_names
    dev_0
    dev_bounds
    dev_param_idx
end

methods
    function this = MyFalsificationProblem(dev_names, dev_0, dev_bounds, BrSys, phi)
        this = this@FalsificationProblem(BrSys, phi);
        this.dev_names = dev_names;
        this.dev_bounds = dev_bounds;
        this.dev_0 = this.normalize_delta(dev_0);

        this.dev_param_idx = zeros(size(dev_names));
        for i = 1:length(dev_names)
            this.dev_param_idx(i) = find(strcmp(BrSys.GetParamList, dev_names(i)), 1);
        end
    end

    function [obj, cval, x_stoch] = objective_fn(this, x)
        [obj, cval, x_stoch] = this.objective_fn@FalsificationProblem(x);
        
        delta = x(this.dev_param_idx, :);
        delta = this.normalize_delta(delta);
        dist = sqrt(sum((delta - this.dev_0).^2));
        obj = obj + dist;
    end

    function [result] = normalize_delta(this, delta)
        result = (delta - this.dev_bounds(:, 1)) ./ (this.dev_bounds(:, 2) - this.dev_bounds(:, 1));
    end

end
    
end