class Agent:
    '''
    This is the base class of all the control agents.
    '''

    def next_action(self, obs):
        '''
        Given some observation, returns the next action to perform.
        '''
        raise NotImplementedError()
    
    def reset(self):
        '''
        Some models like PID are stateful, call this method to reset
        the model's state.
        '''
        raise NotImplementedError()
