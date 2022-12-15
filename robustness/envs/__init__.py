class DeviatableEnv:
    '''
    This is the base class of a parametric environment. It defines
    all the functions that need to be overriden by the subclasses.
    '''

    def instantiate(self, delta):
        '''
        Given a parametr (deviation), instantiate the corresponding
        parametric environment.
        
        Returns the environment and the bound of the initial state
        of this instantiated environment.
        '''
        raise NotImplementedError()
    
    def get_dev_bounds(self):
        '''
        Returns the bound of the deviations (parameters). It is in
        the form of [[x1_low, x1_high], [x2_low, x2_high]].
        '''
        raise NotImplementedError()
    
    def get_delta_0(self):
        '''
        Returns the zero deviation (i.e., no deviation parameter).
        '''
        raise NotImplementedError()
    
    def observation_space(self):
        raise NotImplementedError()
