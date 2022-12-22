class Logger:
    def __init__(self):
        self.trials = []
        self.cur_trial = None
    
    def new_trial(self):
        self.cur_trial = []
        self.trials.append(self.cur_trial)
    
    def add_dev(self, delta, v, x0):
        self.cur_trial.append([delta, v, x0])
    
    def find_dev_result(self, i, delta):
        for item in self.trials[i]:
            if (delta == item[0]).all():
                return item[1:]
        return None
