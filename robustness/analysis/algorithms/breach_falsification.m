function [obj_best, all_signal_values, violation_signal_values] = breach_falsification(env, phi, trials, evals)

  obj_best = inf;
  all_signal_values = struct();
  violation_signal_values = struct();

  for n = 1:trials
    phi = STL_Formula('phi', phi);
    falsif_pb = FalsificationProblem(env, phi);
    falsif_pb.max_obj_eval = evals;
    falsif_pb.StopAtFalse = false;
    falsif_pb.setup_solver('cmaes');
    falsif_pb.solve();
    obj_best = min(obj_best, falsif_pb.obj_best);

    traces = falsif_pb.GetLog;
    summary = traces.GetSummary;
    violation_indices = find(summary.num_violations_per_trace > 0);

    signal_names = traces.GetSignalList;
    for i = 1:length(signal_names)
      if isfield(all_signal_values, signal_names{i})
        all_signal_values.(signal_names{i}) = [all_signal_values.(signal_names{i}); traces.GetSignalValues(signal_names{i})];
      else
        all_signal_values.(signal_names{i}) = traces.GetSignalValues(signal_names{i});
      end

      if isempty(violation_indices) == false
        if isfield(violation_signal_values, signal_names{i})
          violation_signal_values.(signal_names{i}) = [violation_signal_values.(signal_names{i}); traces.GetSignalValues(signal_names{i}, violation_indices)];
        else
          violation_signal_values.(signal_names{i}) = traces.GetSignalValues(signal_names{i}, violation_indices);
        end
      end
    end
  end

end