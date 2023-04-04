% agent = load('LKA_PPO_Agent_9_18.mat').agent;
% name = 'LKA_T_breach';

T = 15;
Ts = 0.1;

m = 1575;   % total vehicle mass (kg)
Iz = 2875;  % yaw moment of inertia (mNs^2)
lf = 1.2;   % longitudinal distance from center of gravity to front tires (m)
lr = 1.6;   % longitudinal distance from center of gravity to rear tires (m)
Cf = 19000; % cornering stiffness of front tires (N/rad)
Cr = 33000; % cornering stiffness of rear tires (N/rad)

global Vx e1_initial e2_initial
Vx = 15;    % longitudinal velocity (m/s)
e1_initial = 0;   % initial lateral deviation
e2_initial = 0;   % initial yaw angle

u_min = -0.5;   % maximum steering angle
u_max = 0.5;    % minimum steering angle

% Helper constants for RL agent
line_width = 3.7;   % highway lane width
avg_car_width = 2;  % average car width
max_late_dev = (line_width - avg_car_width) / 2 - 0.1;
max_rel_yaw_ang = 0.261799; % lateral deviation tolerence
terminate_error = 1.5;

% Define the curvature of the road head
% turn_pos1 = 27.19;
% turn_pos2 = 56.46;
time = 0:Ts:T;
md = getCurvature(Vx, time, turn_pos1, turn_pos2);

% rho = 0.001;        %  curvature of the road

% MPC parameters
% PredictionHorizon = 10;     % MPC prediction horizon
% RL parameters

sg = LKA_signal_gen(name, agent);
model = BreachSignalGen(sg);
model.SetTime(0:Ts:T);
model.SetParamRanges({'Vx', 'e1_initial','e2_initial'}, [12 18; -0.5 0.5; -0.1 0.1]);
% model.SetParamRanges({'e1_initial','e2_initial'}, [-0.5 0.5; -0.1 0.1]);

% pb = FalsificationProblem(model, STL_Formula('phi','alw (abs(lateral_deviation[t]) < 0.85)'));
% pb.setup_solver('cmaes');
% pb.max_obj_eval = 30;
% pb.solve();
% BreachSamplesPlot(pb.GetLog);