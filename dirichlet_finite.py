import numpy as np
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
        self.phi = int(np.floor(S * np.log(S * A / p)))
        self.w = np.log(T / p)
        self.k = np.log(T / p)
        self.M = np.zeros([S, A, S, num_agents])

        for i in range(num_agents):
            for s, a in itertools.product(range(S), range(A)):
                self.M[s, a, :, i] = np.random.dirichlet(np.ones((S)))
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
                    action_return = np.sum(
                        [trans_prob[s, a, s_next] * (reward[s, a, s_next] + gamma * value_func[s_next]) for s_next in range(S)])  # computes the undiscounted returns
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
            env_reward = np.random.normal(0.0, 1.0, size=(state, action, state))
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
                cumulative_reward += env_reward[s_t, a_t, s_next]
                max_reward += np.amax(env_reward[s_t, :, :]) #TODO: make reward for zero transition probabilities correspond to some negative value

            regret = max_reward - cumulative_reward
            regrets += [regret]

        avg_regret = np.sum(regrets)/num_env
        return avg_regret


    def train(self, episodes, horizon, s_t):
        phi = self.phi
        w = self.w
        k = self.k
        M = self.M
        eta = self.eta

        t = 1
        #initialize num_visits and state tracking for each agent
        num_visits = np.zeros((self.S, self.A, self.S, self.num_agents))
        curr_states = np.zeros((self.num_agents), dtype=np.int)
        # M = np.zeros_like(num_visits)
        for i in range(len(curr_states)):
            curr_states[i] = int(s_t)

        # cumulative_rewards = np.zeros((self.num_agents))
        # max_rewards = np.zeros((self.num_agents))
        evaluation_episodic_regret = np.zeros((episodes, self.num_agents))

        #loops through episode
        for i in range(episodes):
            #compute a num visits parameter for dirichlet
            num_visits =  np.sum(num_visits[:, :, :, :], axis=-1) #TODO: check if sum is correct
            num_visits = np.expand_dims(num_visits, axis=-1).repeat(repeats=self.num_agents, axis=-1)
            #for each agent sample a dirichlet mdp and compute the max policy
            for agent in range(self.num_agents):
                #Check whether to use M or N
                trans_prob = self.posterior_sample(self.trans_p, M[:, :, :, agent], self.S, self.A)
                policy = self.compute_policy(trans_prob, self.S, self.A, phi, self.reward)  # computes the max gain policy
                for _ in range(horizon):
                    s_t = curr_states[agent]
                    a_t = int(policy[s_t])
                    s_next = np.random.choice(range(0, self.S), size=1, p=self.trans_p[s_t, a_t, :])
                    num_visits[s_t, a_t, s_next, agent] += 1
                    # cumulative_rewards[agent] += reward[s_t, a_t, s_next]
                    # max_rewards[agent] += np.amax(reward[s_t, :, :])
                    curr_states[agent] = int(s_next)
                # M[:, :, :, agent] = 1 / k * (num_visits[:, :, :, agent] + w) #TODO: max num visits or one
                M = np.maximum(np.ones(num_visits.shape), num_visits)
                evaluation_episodic_regret[i, agent] = self.evaluate(policy, 50, horizon)
            t += horizon
        print("evaluation: ", evaluation_episodic_regret)
        episodic_regret_sum = np.sum(evaluation_episodic_regret, axis=1)/self.num_agents
        print("episodic: ", episodic_regret_sum)
        bayesian_regret = np.mean(episodic_regret_sum)
        print("bayesian: ", bayesian_regret)
        # regret = np.sum(max_rewards - cumulative_rewards)
        # print("cumulative regret at episode", str(i), " ", regret)
        return bayesian_regret


if __name__ == "__main__":
    #Define MDP
    state = 5
    action = 5
    seeds = range(100, 101)
    for seed in seeds:
        print("seed: ", seed)
        np.random.seed(seed)
        reward = np.random.normal(0.0, 1.0, size=(state, action, state))
        trans_p = np.zeros([state, action, state])
        for i in range(state):
            for j in range(action):
                sample = np.random.gamma(1, 1, state)
                trans_p[i, j, :] = sample / np.sum(sample)
        #end Define MDP

        total_regret = []

        num_agents = [1, 2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for agents in num_agents:
            print("agent: ", agents)
            psrl = DirichletFiniteAgent(agents, state, action, 100, 0.75, trans_p, reward)
            regret = psrl.train(30, 75, int(np.random.randint(0, state, 1)))
            total_regret += [regret]

        np.savetxt("result" + str(seed) + ".csv", np.column_stack((num_agents, total_regret)), delimiter=",")