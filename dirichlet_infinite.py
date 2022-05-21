import numpy as np
import itertools

class DirichletInfiniteAgent:
    def __init__(self, num_agents, S, A, T, p, trans_p, reward):
        """
        S: state space
        A: action space
        s_1: starting state
        T: time horizon
        p: parameter between (0, 1]
        """
        self.num_agents = num_agents
        self.S = S
        self.A = A
        self.T = T
        self.phi = int(np.floor(S * np.log(S * A / p)))
        self.w = np.log(T / p)
        self.k = np.log(T / p)

        self.M = np.ones([S, A, S])

        self.eta = np.sqrt(T * S / A) + 12 * self.w * S ** 4
        self.trans_p = trans_p
        self.reward = reward

    def posterior_sample(self, transition_prob, M, S, A):
        dirichlet_trans_p = np.zeros(transition_prob.shape)
        for s, a in itertools.product(range(S), range(A)):
            dirichlet_trans_p[s, a] = np.random.dirichlet(M[s, a, :])
        return dirichlet_trans_p

    def compute_policy(self, trans_prob, S, A, phi, reward):
        # performs undiscounted value iteration to output an optimal policy
        value_func = np.zeros(S)
        policy = np.zeros(S)
        iter = 0
        gamma = 0.99
        while True:
            tolerance = 0.01
            diff = np.zeros(S)
            for s in range(S):
                value = value_func[s]
                action_returns = []
                for a in range(A):
                    action_return = reward[s, a] + gamma * np.sum(
                        [trans_prob[s, a, s_next] * value_func[s_next] for s_next in range(S)])  # computes the undiscounted returns
                    action_returns.append(action_return)
                value_func[s] = np.max(action_returns)
                policy[s] = np.argmax(action_returns)
                diff[s] = max(diff[s], np.abs(value - value_func[s]))
            iter += 1
            if iter % 10000 == 0:
                print("diff: ", diff)
            if np.max(diff) <= tolerance:
                break
        return policy

    def evaluate(self, policy, num_env, horizon, episodes=1):
        regrets = []
        for env in range(num_env):
            env_reward = self.reward
            env_trans_p = np.zeros([state, action, state])
            for i in range(state):
                for j in range(action):
                    sample = np.random.gamma(1, 1, state)
                    env_trans_p[i, j, :] = sample / np.sum(sample)
            cumulative_reward = 0
            max_reward = 0
            s_t = int(np.random.randint(0, self.S, 1))
            #get regret for one episode
            for t in range(horizon):
                a_t = int(policy[s_t])
                s_next = np.random.choice(range(0, self.S), size=1, p=env_trans_p[s_t, a_t, :])
                cumulative_reward += env_reward[s_t, a_t]
                max_reward += np.amax(env_reward[s_t, :])

            regret = max_reward - cumulative_reward
            regrets += [regret]

        avg_regret = np.sum(regrets)/num_env
        return avg_regret


    def train(self, epochs, s_t):
        phi = self.phi
        w = self.w
        k = self.k
        M = self.M
        T = self.T

        num_visits = np.zeros((self.S, self.A, self.S, self.num_agents))
        curr_states = np.zeros((self.num_agents), dtype=np.int)
        for i in range(len(curr_states)):
            curr_states[i] = int(s_t)
        t = 0

        evaluation_epoch_regret = np.zeros((epochs, self.num_agents))
        time_steps = np.zeros(epochs)

        for i in range(epochs):
            policies = []
            for agent in range(self.num_agents):
                trans_prob = self.posterior_sample(self.trans_p, M, self.S, self.A,)
                policy = self.compute_policy(trans_prob, self.S, self.A, phi, self.reward)  # computes the max gain policy
                policies += [policy]

            num_visits_next = np.copy(num_visits)
            while True:
                end_epoch = False
                for agent in range(self.num_agents):
                    s_t = curr_states[agent]
                    a_t = int(policies[agent][s_t])
                    s_next = np.random.choice(range(0,self.S), size=1, p=self.trans_p[s_t, a_t, :])
                    num_visits_next[s_t, a_t, s_next, agent] += 1

                    if np.sum(num_visits_next[s_t, a_t, :, agent]) >= 2 * np.sum(num_visits[s_t, a_t, :, agent]):
                        end_epoch = True
                    curr_states[agent] = s_next

                t += 1
                if t == T:
                    # print("evaluation: ", evaluation_epoch_regret)
                    num_epochs = np.sum([1 for i in time_steps if i != 0])
                    evaluation_epoch_regret = evaluation_epoch_regret[:int(num_epochs), :]
                    time_steps = time_steps[time_steps != 0]
                    epoch_regret_sum = np.sum(evaluation_epoch_regret, axis=1) / self.num_agents
                    # print("episodic: ", epoch_regret_sum)
                    bayesian_regret = np.average(epoch_regret_sum, weights=time_steps)
                    print("bayesian: ", bayesian_regret)
                    return bayesian_regret
                    break

                if end_epoch:
                    num_visits = num_visits_next
                    num_visits_current = np.sum(num_visits[:, :, :, :], axis=-1)
                    M = np.ones(M.shape) + num_visits_current
                    for agent in range(len(policies)):
                        evaluation_epoch_regret[i, agent] = self.evaluate(policies[agent], 50, 75) #TODO: find evaluation horizon
                    time_steps[i] = t
                    break
            if t == T:
                break


if __name__ == "__main__":
    #Define MDP
    state = 10
    action = 5
    seeds = range(130, 150)
    for seed in seeds:
        print("seed: ", seed)
        np.random.seed(seed)
        reward = np.abs(np.random.normal(0.0, 1.0, size=(state, action)))
        trans_p = np.zeros([state, action, state])
        for i in range(state):
            for j in range(action):
                sample = np.random.gamma(1, 1, state)
                trans_p[i, j, :] = sample / np.sum(sample)
    #end Define MDP

        total_regret = []

        #TODO: add agents to the list
        num_agents = [1, 2, 10, 20, 30]
        for i in num_agents:
            print("agents: ", i)
            psrl = DirichletInfiniteAgent(i, state, action, 20000, 0.75, trans_p, reward)
            regret = psrl.train(1000, int(np.random.randint(0, state, 1)))
            total_regret += [regret]

        # np.savetxt("result" + str(seed) + ".csv", np.column_stack((num_agents, total_regret)), delimiter=",")