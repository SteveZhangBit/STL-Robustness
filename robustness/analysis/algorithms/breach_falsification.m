function [obj_best] = breach_falsification(env, phi, trials, evals)

  obj_best = inf;

  for n = 1:trials
    phi = STL_Formula('phi', phi);
    falsif_pb = FalsificationProblem(env, phi);
    falsif_pb.max_obj_eval = evals;
    falsif_pb.setup_solver('cmaes');
    falsif_pb.solve();
    obj_best = min(obj_best, falsif_pb.obj_best);
  end

end