T = 50;
Ts = 0.1;
% Specify the linear model for ego car.
G_ego = tf(1,[0.5,1,0]);
t_gap = 1.4;
D_default = 10;
% Specify the driver-set velocity in m/s.
v_set = 30;
% ego car minimum/maximum velocity
vmax_ego = 50;
vmin_ego = 0;
% the acceleration is constrained to the range [-3,2] (m/s^2).
amin_ego = -3;
amax_ego = 2;
% the velocity and the position of lead car
% x0_lead = 70;
% v0_lead = 40;
% the velocity and the position of ego car
x0_ego = 10;
v0_ego = 20;
% the acceleration is constrained to the range [-1,1] (m/s^2).
amin_lead = -1;
amax_lead = 1;

model = BreachSimulinkSystem(name);
model.Sys.tspan = 0:Ts:T;

cp_num = 10;
input_gen.type = 'UniStep';
input_gen.cp = cp_num;

model.SetInputGen(input_gen);

for cpi = 0:input_gen.cp-1
    in_lead_sig = strcat('in_lead_u', num2str(cpi));
    model.SetParamRanges({in_lead_sig}, [amin_lead amax_lead]);
end
