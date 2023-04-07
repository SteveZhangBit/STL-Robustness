% agent = load('AFC_DDPG_Agent.mat').agent;
% name = 'AFC_T_breach';

T = 30;
Ts = 0.1;
fuel_inj_tol = 1.0;
% MAF_sensor_tol = 1.0; % [0.95, 1.05]
% AF_sensor_tol = 1.0; % [0.99 1.01]
pump_tol = 1;
kappa_tol = 1;
tau_ww_tol = 1;
fault_time = 50;
kp = 0.04;
ki = 0.14;

%The engine speed is constrained to the range [900,1100].
min_Engine_Speed = 900;
max_Engine_Speed = 1100;

%The pedal angle is constrained to the range [8.8,61.1].
min_Pedal_Angle = 8.8;
max_Pedal_Angle = 61.1;

% RL parameters
max_mu = 0.05;        % mu from STL
mu_tol = 0.1;         % mu value to terminate the episode
time_tol = 5;         % time tolerence, to avoid the initial large mu value

model = BreachSimulinkSystem(name, {'MAF_sensor_tol', 'AF_sensor_tol'});
model.Sys.tspan = 0:Ts:T;
model.SetParam('MAF_sensor_tol', MAF_sensor_tol);
model.SetParam('AF_sensor_tol', AF_sensor_tol);

cp_num = 5;
input_gen.type = 'UniStep';
input_gen.cp = cp_num;

model.SetInputGen(input_gen);

for cpi = 0:input_gen.cp-1
    Pedal_Angle_sig = strcat('Pedal_Angle_u',num2str(cpi));
    model.SetParamRanges({Pedal_Angle_sig}, [min_Pedal_Angle max_Pedal_Angle]);
    
    Engine_Speed_sig = strcat('Engine_Speed_u', num2str(cpi));
    model.SetParamRanges({Engine_Speed_sig}, [min_Engine_Speed max_Engine_Speed]);
end

% pb = FalsificationProblem(model, STL_Formula('phi','alw (mu[t] < 0.2)'));
% pb = FalsificationProblem(model, STL_Formula('phi','alw (AF[t] < 1.2*14.7 and AF[t] > 0.8*14.7)'));
% pb.setup_solver('cmaes');
% pb.max_obj_eval = 30;
% pb.solve();
% BreachSamplesPlot(pb.GetLog);