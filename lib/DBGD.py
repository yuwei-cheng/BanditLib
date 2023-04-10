import numpy as np
from Users.LearningUsers import LinUser, norm2, get_eigen_vec
import matplotlib.pyplot as plt
import argparse

class DBGD:
    """
    Dueling Bandit Gradient Descent (DBGD)
    """
    def __init__(self, feature_dim, gamma=0.1, delta=0.1):
        self.d = feature_dim
        self.gamma = gamma # exploitation
        self.delta = delta # exploration
        self.best = norm2(np.ones(self.d)) # initial arm (w1)
        self.T = 1

    def gen_rand_direction(self):
        return norm2(np.random.randn(self.d))

    def simulate(self, user, T=10000, gamma=0.2, delta=0.1, plot=False, verbose=False, learning = True):
        self.T = T
        self.gamma = gamma
        self.delta = delta
        self.regret = []

        for t in range(1, T, 1):
            ut = self.gen_rand_direction()
            right = norm2(self.best + self.delta * ut) # wt'
            right_score, winner, _ = user.get_winner(self.best, right, learning=learning)
            if right_score: # If wt' wins
                self.best = norm2(self.best + self.gamma * ut)
            user.get_reward(winner)
            self.regret.append(1 - np.dot(user.theta_star, winner))

        if plot:
            plt.plot(np.cumsum(self.regret))
            plt.show()


if __name__ == '__main__':

    # Set seed for the purpose of replication
    SEED = 666
    np.random.seed(SEED)

    # environment settings
    parser = argparse.ArgumentParser(description="AES test")
    parser.add_argument('--alg', default='SMD')
    parser.add_argument('--d', default=5, help="dimension")
    parser.add_argument('--T', default=100000, help="time horizon")
    parser.add_argument('--gamma', default=0.2)
    # parser.add_argument('--delta', default=0.1)

    args = vars(parser.parse_args())

    d = int(args['d'])
    T = int(args['T'])
    gamma = float(args['gamma'])

    # user configuration
    user = LinUser(feature_dim=d, gamma = gamma, V0=1.0 * np.identity(n=d))
    # algorithm initialization
    dbgd = DBGD(feature_dim=d)
    regret = []
    # fig, ax = plt.subplots()
    user.reset()
    # run simulation
    # aes.simulate(user=user, T=T, T0=T0, delta=delta, cut_thres=cut_thres, verbose=False)
    dbgd.simulate(user=user, T=T, plot=True)

    # visualize result
    regret.append(np.cumsum(dbgd.regret))
    plt.plot(regret[-1], 'r-.', label='DBGD')
    plt.title('Regret curves of DBGD for learning users')
    plt.xlabel("t")
    plt.ylabel("Regret(t)")
    plt.legend(fontsize=12)
    plt.show()
