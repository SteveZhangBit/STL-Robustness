function md = getCurvature(Vx, time, turn_pos1, turn_pos2)
% Get previewed curvature from desired X and Y positions for LKA
%
% Inputs:
%   Vx: longitudinal velocity
%   time: time vector
%
% Outputs:
%   md: previewed curvature

% Desired X position
Xref = Vx*time;

% Desired Y position
z1 = (2.4/50)*(Xref-turn_pos1)-1.2;
z2 = (2.4/43.9)*(Xref-turn_pos2)-1.2;
Yref = 8.1/2*(1+tanh(z1)) - 11.4/2*(1+tanh(z2));

% Desired curvature
DX = gradient(Xref,0.1);
DY = gradient(Yref,0.1);
D2Y = gradient(DY,0.1);
curvature = DX.*D2Y./(DX.^2+DY.^2).^(3/2);

% Stored curvature (as input for LKA)
md.time = time;
md.signals.values = curvature';
