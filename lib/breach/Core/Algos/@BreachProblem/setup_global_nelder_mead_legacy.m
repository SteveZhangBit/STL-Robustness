function opt = setup_global_nelder_mead_legacy(this, gui, varargin)
this.solver = 'global_nelder_mead';
dim_pb = numel(this.params);
opt = struct( ...
    'use_param_set_as_init',false,...
    'start_at_trial', 0, ...
    'quasi_rand_seed', 0, ...
    'nb_max_corners', min(2^dim_pb, 10*dim_pb),...    
    'nb_new_trials',  min(2^dim_pb, 10*dim_pb)+10*numel(this.params), ...
    'nb_local_iter',  50, ...
    'local_optim_options', optimset() ...
    );

if this.use_parallel
    opt.use_parallel = true;
end

% checks with what we have already
if isstruct(this.solver_options)
    fn = fieldnames(this.solver_options);
    for ifn = 1:numel(fn)
        field = fn{ifn};
        if isfield(opt, field)
            opt.(field) = this.solver_options.(field);
        end
    end
end

if nargin > 2
    opt = varargin2struct_breach(opt, varargin{:});
end

if (nargin >= 2)&&gui
    choices = struct( ...
        'use_param_set_as_init','bool',...
        'start_at_trial', 'int', ...
        'quasi_rand_seed',  'int', ...
        'nb_new_trials',  'int', ...
        'nb_local_iter',  'int', ...
        'local_optim_options', 'string' ...
        );
    tips = struct( ...
        'use_param_set_as_init','Use the samples in the parameter set used to create the problem as initial trials. Otherwise, starts with corners, then quasi-random sampling.',...
        'start_at_trial', 'Skip the trials before that. Use 0 if this is the first time you are solving this problem.', ...
        'quasi_rand_seed','Seed for quasirandom sampling of initial conditions',...
        'nb_new_trials',  'Number of initial parameters used before going into local optimization.', ...
        'nb_local_iter',  'Number of iteration of Nelder-Mead algorithm for each trial.', ...
        'local_optim_options', 'Advanced local solver options. ' ...
        );
    gui_opt = opt;
    gui_opt.local_optim_options = 'default optimset()';
    
    opt = BreachOptionGui('Choose options for solver global_nelder_mead', gui_opt, choices, tips);
    close(opt.dlg);
    
    return;
    %gui_opt = gu.output;
    %gui_opt.local_optim_options = opt.local_optim_options;
    %opt = gui_opt;
end

this.solver_options = opt;


end