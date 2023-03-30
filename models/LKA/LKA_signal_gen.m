classdef LKA_signal_gen < signal_gen
    % Wrapper class for LKA
    properties
        mdl
        agent
    end
    methods
        function this = LKA_signal_gen(mdl, agent)
            this.params = { ...
                'Vx', ...
                'e1_initial', ...   % initial lateral deviation
                'e2_initial'};      % initial yaw angle
            
            this.p0 = [15; 0; 0];
            % this.p0 = [0; 0];
            this.signals = {'lateral_deviation'}; % starting with bare minimum for benchmark
            this.mdl = mdl;
            this.agent = agent;
        end
        
        function X = computeSignals(this, p, t_vec)
            global Vx e1_initial e2_initial
            Vx  = p(1);  % longitudinal velocity (m/s)
            e1_initial = p(2);
            e2_initial = p(3);

            % e1_initial = p(1);
            % e2_initial = p(2);
                        
            sim(this.mdl);
            data = logsout.find('Name', 'lateral_deviation');
            X = data{1}.Values.Data';
        end
    end
end
