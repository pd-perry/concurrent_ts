import numpy as np
import torch
import itertools

class DirichletFiniteAgent:
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
        self.M = np.ones([S, A, S])
        self.trans_p = trans_p
        self.reward = reward
        self.R_mean = 1.0
        self.R_var = 0

    def update(self, R, mean, variance, tau_0, mu_0, sample_size):
        #tau_0 and mu_0 are priors of mean
        updated_mean = (variance * mean + mu_0 * R)/(sample_size * variance + mu_0)
        updated_variance = variance * tau_0/(sample_size* variance + tau_0)
        return updated_mean, updated_variance

    def posterior_sample(self, transition_prob, M, S, A):
        dirichlet_trans_p = np.zeros(transition_prob.shape)
        for s, a in itertools.product(range(S), range(A)):
            dirichlet_trans_p[s, a] = np.random.dirichlet(M[s, a, :])
        return dirichlet_trans_p

    def compute_policy(self, trans_prob, S, A, reward):
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
                break
            if np.max(diff) <= tolerance:
                # print(value_func)
                break

        return policy

    def evaluate(self, policy, num_env, horizon, episodes=1):
        regrets = []
        for env in range(num_env):
            #samples environment
            #TODO: ASK JERRY SAMPLE EACH TIME OR SAMPLE AT THE BEGINNING AND USE FOR EACH EVAL
            # env_reward = np.abs(np.random.normal(0.0, 1.0, size=(state, action, state)))
            env_reward = self.reward
            env_trans_p = np.zeros([state, action, state])
            for i in range(state):
                for j in range(action):
                    # sample = np.random.gamma(1, 1, state)
                    # env_trans_p[i, j, :] = sample / np.sum(sample)
                    env_trans_p[i, j, :] = np.random.dirichlet(np.ones(self.S))
            cumulative_reward = 0
            max_reward = 0
            s_t = int(np.random.randint(0, self.S, 1))
            #get regret for one episode
            for t in range(horizon):
                a_t = int(policy[s_t])
                s_next = np.random.choice(range(0, self.S), size=1, p=env_trans_p[s_t, a_t, :])
                cumulative_reward += env_reward[s_t, a_t]
                max_reward += np.amax(env_reward[s_t, :]) #TODO: make reward for zero transition probabilities correspond to some negative value

            regret = max_reward - cumulative_reward
            regrets += [regret]

        avg_regret = np.sum(regrets)/num_env
        return avg_regret

    def train(self, episodes, horizon, s_t):
        M = self.M

        t = 1
        #initialize num_visits and state tracking for each agent
        num_visits = np.zeros((self.S, self.A, self.S, self.num_agents))
        curr_states = np.zeros(self.num_agents, dtype=np.int)
        evaluation_episodic_regret = np.zeros((episodes, self.num_agents))
        R = []

        for i in range(episodes):
            for a in range(self.num_agents):
                curr_states[a] = int(np.random.randint(0, self.S, 1))

            # trans_prob_tensor = torch.tensor([self.num_agents, self.S, self.A, self.S])
            # policies = torch.tensor([self.num_agents, self.S])
            # for _ in range(horizon):
            #     for agent in range(self.num_agents):
            #         trans_prob_tensor[agent, :, :, :] = self.posterior_sample(self.trans_p, M, self.S, self.A)
            #         policies[agent, :] = self.compute_policy(trans_prob_tensor[agent, :, :, :], self.S, self.A,
            #                                                  self.reward)
            #     s_t = curr_states
            #     a_t = policies.gather(1, s_t)
            #     s_next = np.random.choice(range(0, self.S), size=1, p=self.trans_p[s_t, a_t, :])
            #     num_visits[s_t, a_t, s_next, agent] += 1
            #     curr_states[agent] = int(s_next)

            for agent in range(self.num_agents):
                #evaluation as in sample from M multiple times
                trans_prob = self.posterior_sample(self.trans_p, M, self.S, self.A)
                reward = np.random.normal(self.R_mean, self.R_var, size=(state, action))
                policy = self.compute_policy(trans_prob, self.S, self.A, reward)
                for _ in range(horizon):
                    s_t = curr_states[agent]
                    a_t = int(policy[s_t])
                    s_next = np.random.choice(range(0, self.S), size=1, p=self.trans_p[s_t, a_t, :])
                    R += [reward[s_t, a_t]]
                    num_visits[s_t, a_t, s_next, agent] += 1
                    curr_states[agent] = int(s_next)
                evaluation_episodic_regret[i, agent] = self.evaluate(policy, 50, horizon)

            # compute a num visits parameter for dirichlet
            num_visits_current = np.sum(num_visits[:, :, :, :], axis=-1)
            M = np.ones(M.shape) + num_visits_current
            #update posterior for reward
            self.R_mean, self.R_var = np.mean(R), np.var(R, ddof=1)
            t += horizon
        # print("evaluation: ", evaluation_episodic_regret)
        episodic_regret_avg_over_agent = np.sum(evaluation_episodic_regret, axis=1)/self.num_agents
        # print("episodic: ", episodic_regret_avg_over_agent)
        bayesian_regret = np.sum(episodic_regret_avg_over_agent)/t
        print("bayesian: ", bayesian_regret)
        return bayesian_regret


if __name__ == "__main__":
    #Define MDP
    state = 10
    action = 5
    #TODO: scale up the state and action
    #uniform sample over all the state
    #set horizon=1, initial state for each agent drawn from uniform distribution across all states
    seeds = range(100, 101)
    for seed in seeds:
        print("seed: ", seed)
        np.random.seed(seed)
        reward = np.random.normal(0.0, 1.0, size=(state, action))
        trans_p = np.zeros([state, action, state])
        for i in range(state):
            for j in range(action):
                # sample = np.random.gamma(1, 1, state)
                # trans_p[i, j, :] = sample / np.sum(sample)
                trans_p[i, j, :] = np.random.dirichlet(np.ones(state))
        #generate straight from dirichlet
        #end Define MDP

        total_regret = []

        num_agents = [1, 2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for agents in num_agents:
            print("agent: ", agents)
            psrl = DirichletFiniteAgent(agents, state, action, 100, 0.75, trans_p, reward)
            regret = psrl.train(30, 1, int(np.random.randint(0, state, 1)))
            total_regret += [regret]

        np.savetxt("evaluation_finite/result" + str(seed) + ".csv", np.column_stack((num_agents, total_regret)), delimiter=",")