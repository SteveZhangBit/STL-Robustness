% agent = load('WTK_TD3_Agent_9_20.mat').agent;
% name = 'WTK_RL_breach';

T = 24;
Ts = 0.1;
h_ref_min = 0;
h_ref_max = 15;
% in_rate = 0.5;
% out_rate = 0.1;
% in_rate_range = [0.01 0.5];
% out_rate_range = [0.01 0.2];
global dev_names;
global dev_0;
global dev_bounds;
dev_names = {'in_rate', 'out_rate'};
dev_0 = [in_rate; out_rate];
dev_bounds = [in_rate_range; out_rate_range];
% RL parameters
% desire_WL = 10;

model = BreachSimulinkSystem(name, {'in_rate', 'out_rate'});
model.Sys.tspan = 0:Ts:T;

cp_num = 4;
input_gen.type = 'UniStep';
input_gen.cp = cp_num;

model.SetInputGen(input_gen);

for cpi = 0:input_gen.cp-1
    h_ref_sig = strcat('h_ref_u', num2str(cpi));
    model.SetParamRanges({h_ref_sig}, [h_ref_min h_ref_max]);
end

model.SetParamRanges('in_rate', in_rate_range);
model.SetParamRanges('out_rate', out_rate_range);

% pb = MyFalsificationProblem(dev_names, dev_0, dev_bounds, model, STL_Formula('phi','alw_[5,5.9](abs(h_error[t]) < 1) and alw_[11,11.9](abs(h_error[t]) < 1) and alw_[17,17.9](abs(h_error[t]) < 1) and alw_[23,23.9](abs(h_error[t]) < 1)'));
% pb.max_obj_eval = 30;
% pb.StopAtFalse = false;
% pb.setup_solver('cmaes');
% pb.solve();
% BreachSamplesPlot(pb.GetLog);
