import numpy as np
from Users.LearningUsers import LinUser, norm2, get_eigen_vec
import matplotlib.pyplot as plt
import argparse
import scipy

class SMD:
    """
    Noisy Comparison-based Stochastic Mirror Descent (NC-SMD)
    """
    def __init__(self, feature_dim, eta=0.1, lmbd=0.1, mu=0.1, b=2):
        self.d = feature_dim
        self.eta = eta # learning rate
        self.lmbd = lmbd # tuning parameter
        self.mu = mu # tuning parameter
        self.b = b # self-concordant function
        self.a = np.ones(self.d) # self concordant function
        self.T = 1
        self.alpha = np.zeros((self.T, self.d))

    def hessian(self, left, t):
        """
        Compute the hessian matrix of the self concordant function
        :param left: at
        """
        k = self.b - np.dot(self.a, left)
        return k**2 * np.ones((self.d, self.d)) + (self.lmbd * self.eta*t + 2*self.mu)*np.eye(self.d)

    def gradient(self, left, t):
        """
        Compute the gradient of the self concordant function
        :param left: at
        :param alpha: previous selected arms
        :return:
        """
        k = self.b - np.dot(self.a, left)
        return np.ones(self.d) / k + self.lmbd * self.eta * np.sum(left - self.alpha[0:t, :], axis=0) + 2*self.mu*left

    def rt_square_root(self, A):
        """
        Find B such that A = BB, A is psd
        :param A: PSD matrix
        :return:
        """
        eig_values, eig_vectors = np.linalg.eig(A)
        sqrt_matrix = eig_vectors * np.sqrt(eig_values) @ np.linalg.inv(eig_vectors)
        return sqrt_matrix

    def gen_rand_direction(self):
        return norm2(np.random.randn(self.d))

    def simulate(self, user, T=1000, eta=0.1, lmbd=0.1, mu=0.1, plot=False, verbose=False):
        self.T = T
        self.eta = eta
        self.lmbd = lmbd
        self.mu = mu
        self.regret = []
        self.alpha = np.zeros((self.T, self.d)) # recommended arms

        R = lambda x: -np.log(self.b - np.dot(self.a, x / np.linalg.norm(x, 2)))
        left = scipy.optimize.minimize(R, np.ones(self.d))["x"]
        self.alpha[0, :] = norm2(left)

        for t in range(1, T, 1):
            left = self.alpha[t-1, :]
            ut = self.gen_rand_direction()
            H = self.hessian(left, t)
            G = self.gradient(left, t)

            right = norm2(left + np.linalg.inv(self.rt_square_root(H)) @ ut)
            right_score, winner, _ = user.get_winner(left, right, learning=False)
            gt = right_score * self.d * self.rt_square_root(H)@ut
            new = np.linalg.inv(H) @ (G - self.eta * gt)
            self.alpha[t, :] = norm2(new)
            user.get_reward(winner)
            self.regret.append(1 - np.dot(user.theta_star, winner))

        if plot:
            plt.plot(np.cumsum(self.regret))
            plt.show()


if __name__ == '__main__':

    # environment settings
    parser = argparse.ArgumentParser(description="AES test")
    parser.add_argument('--alg', default='SMD')
    parser.add_argument('--d', default=2, help="dimension")
    parser.add_argument('--T', default=10000, help="time horizon")
    # parser.add_argument('--delta', default=0.2)
    # parser.add_argument('--delta', default=0.2)

    args = vars(parser.parse_args())

    d = int(args['d'])
    T = int(args['T'])

    # user configuration
    user = LinUser(feature_dim=d, V0=1.0 * np.identity(n=d))
    # algorithm initialization
    smd = SMD(feature_dim=d)
    regret = []
    fig, ax = plt.subplots()
    user.reset()
    # run simulation
    # aes.simulate(user=user, T=T, T0=T0, delta=delta, cut_thres=cut_thres, verbose=False)
    smd.simulate(user=user, T=T, plot=True)

    # visualize result
    regret.append(np.cumsum(smd.regret))
    ax.plot(regret[-1], 'r-.', label='SMD')
    ax.set_title('Regret curves of SMD for learning users')
    ax.set_xlabel("t")
    ax.set_ylabel("Regret(t)")
    ax.legend(fontsize=12)
    plt.show()
