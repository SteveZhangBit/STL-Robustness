function [obj_best, obj_values, all_signal_values, violation_signal_values, all_param_values, violation_param_values] = one_layer_breach_falsification(env, phi, trials, evals)

    global dev_names;
    global dev_0;
    global dev_bounds;
    
    obj_best = inf;
    obj_values = zeros(trials, evals);
  
    all_signal_values = struct();
    violation_signal_values = struct();
  
    all_param_values = struct();
    violation_param_values = struct();
  
    for n = 1:trials
      phi = STL_Formula('phi', phi);
      falsif_pb = MyFalsificationProblem(dev_names, dev_0, dev_bounds, env, phi);
      falsif_pb.max_obj_eval = evals;
      falsif_pb.StopAtFalse = false;
      falsif_pb.setup_solver('cmaes');
      falsif_pb.solve();
  
      obj_best = min(obj_best, falsif_pb.obj_best);
      obj_values(n, :) = falsif_pb.obj_log;
  
      traces = falsif_pb.GetLog;
    %   summary = traces.GetSummary;
    %   violation_indices = find(summary.num_violations_per_trace > 0);
  
      signal_names = traces.GetSignalList;
      for i = 1:length(signal_names)
        if isfield(all_signal_values, signal_names{i})
          all_signal_values.(signal_names{i}) = [all_signal_values.(signal_names{i}); traces.GetSignalValues(signal_names{i})];
        else
          all_signal_values.(signal_names{i}) = traces.GetSignalValues(signal_names{i});
        end
  
        % if isempty(violation_indices) == false
        %   if isfield(violation_signal_values, signal_names{i})
        %     violation_signal_values.(signal_names{i}) = [violation_signal_values.(signal_names{i}); traces.GetSignalValues(signal_names{i}, violation_indices)];
        %   else
        %     violation_signal_values.(signal_names{i}) = traces.GetSignalValues(signal_names{i}, violation_indices);
        %   end
        % end
      end
  
      param_names = traces.GetParamList;
      for i = 1:length(param_names)
        if isfield(all_param_values, param_names{i})
          all_param_values.(param_names{i}) = [all_param_values.(param_names{i}); traces.GetParam(param_names{i})];
        else
          all_param_values.(param_names{i}) = traces.GetParam(param_names{i});
        end
        % if isempty(violation_indices) == false
        %   violation_param_values.(param_names{i}) = traces.GetParam(param_names{i}, violation_indices);
        % end
      end
    end
  
end