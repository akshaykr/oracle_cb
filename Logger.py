import numpy as np
import Simulators
import scipy.misc

class LogEntry(object):
    def __init__(self, context, action, action_feedback, slate_feedback, all_feedback, slate_prob, action_probs):
        self.context = context
        self.action = action
        self.action_feedback = action_feedback
        self.slate_feedback = slate_feedback
        self.all_feedback = all_feedback
        self.slate_prob = slate_prob
        self.action_probs = action_probs

class Logger(object):
    def __init__(self,Sim):
        self.Sim = Sim
        
    def collect_uniform_log(self, n):
        self.data = []
        for i in range(n):
            x = self.Sim.get_new_context()
            K = self.Sim.get_num_actions()
            L = self.Sim.L
            A = np.random.choice(range(K), L, replace=False)

            rsub = self.Sim.get_base_rewards(A)
            r = self.Sim.get_slate_reward(A)
            all_r = self.Sim.get_all_rewards()

            self.data.append(LogEntry(x, A, rsub, r, all_r, 1.0/(scipy.misc.comb(K,L)*scipy.misc.factorial(L)), 1.0*L/K*np.ones(L)))
