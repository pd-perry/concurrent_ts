import numpy as np
import torch
import itertools

class DirichletFiniteAgent:
    def __init__(self, num_agents, num_env, S, A, trans_p, reward):
        """
        S: state space
        A: action space
        s_1: starting state
        T: time horizon
        p: parameter between (0, 1]
        """
        self.num_agents = num_agents
        self.num_env = num_env
        self.S = S
        self.A = A
        self.M = np.ones([num_env, S, A, S])
        self.trans_p = trans_p
        self.reward = reward
        self.R_mean = np.full((num_env, S, A), 0.0) #mu_0 =0, sigma_0=1 confirmed, x=sample mean, sigma=1, n is max(1, num_visit to (s,a) pair)

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

    def train(self, episodes, horizon):
        num_env = self.num_env
        M = self.M
        num_visits = np.zeros((num_env, self.S, self.A, self.S, self.num_agents))
        curr_states = np.zeros((num_env, self.num_agents), dtype=np.int)
        evaluation_episodic_regret = np.zeros((num_env, episodes, self.num_agents))
        max_reward = np.zeros((num_env, self.num_agents))
        cumulative_reward = np.zeros((num_env, self.num_agents))
        R = np.zeros((num_env, self.S, self.A))
        t = np.ones(num_env)
        for i in range(episodes):
            for env in range(num_env):
                # initialize num_visits and state tracking for each agent
                for a in range(self.num_agents):
                    curr_states[env, a] = int(np.random.randint(0, self.S, 1))

                for agent in range(self.num_agents):
                    #evaluation as in sample from M multiple times
                    trans_prob = self.posterior_sample(self.trans_p, M[env], self.S, self.A)
                    reward = np.zeros((self.S, self.A))
                    for s in range(self.S):
                        for a in range(self.A):
                            reward[s, a] = np.abs(np.float(np.random.normal(self.R_mean[env, s, a], 1, size=1))) #TODO: each s,a pair has its own posterior
                    policy = self.compute_policy(trans_prob, self.S, self.A, reward)
                    for _ in range(horizon):
                        s_t = curr_states[env, agent]
                        a_t = int(policy[s_t])
                        s_next = np.random.choice(range(0, self.S), size=1, p=self.trans_p[s_t, a_t, :])
                        R[env, s_t, a_t] += reward[s_t, a_t] #TODO: check if there needs to be agent dimension
                        max_reward[env, agent] += np.amax(reward[s_t, :])
                        cumulative_reward[env, agent] += reward[s_t, a_t]
                        num_visits[env, s_t, a_t, s_next, agent] += 1
                        curr_states[env, agent] = int(s_next)
                    # evaluation_episodic_regret[i, agent] = self.evaluate(policy, 50, horizon)
                    evaluation_episodic_regret[env, i, agent] = max_reward[env, agent] - cumulative_reward[env, agent]
                    max_reward[env, agent] = 0
                    cumulative_reward[env, agent] = 0

                # compute a num visits parameter for dirichlet
                num_visits_current = np.sum(num_visits[env, :, :, :, :], axis=-1)
                M[env] = np.ones(M[env].shape) + num_visits_current
                #update posterior for reward
                n = np.maximum(np.ones(num_visits[env, :, :, 0, 0].shape), np.sum(num_visits[env], axis=(-2, -1)))
                self.R_mean[env] = np.multiply(np.divide(1, (np.divide(1, n) + 1)), np.divide(R[env], n))  #TODO: CHANGE
                t[env] += horizon
        # print("evaluation: ", evaluation_episodic_regret)
        avg_over_env_reg = np.sum(evaluation_episodic_regret, axis=0)/num_env
        episodic_regret_avg_over_agent = np.sum(avg_over_env_reg, axis=-1)/self.num_agents
        # print("episodic: ", episodic_regret_avg_over_agent)
        bayesian_regret = np.sum(episodic_regret_avg_over_agent)/np.mean(t)
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
        reward = np.abs(np.random.normal(0.0, 1.0, size=(state, action)))
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
            psrl = DirichletFiniteAgent(agents, 10, state, action, trans_p, reward)
            regret = psrl.train(30, 1)
            total_regret += [regret]

        np.savetxt("evaluation_finite/result" + str(seed) + ".csv", np.column_stack((num_agents, total_regret)), delimiter=",")