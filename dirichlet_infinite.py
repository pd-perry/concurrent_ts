import numpy as np
import itertools

class DirichletInfiniteAgent:
    def __init__(self, num_agents, num_env, S, A, T, trans_p, reward):
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
        self.T = T
        self.M = np.ones([num_env, S, A, S])

        self.trans_p = trans_p
        self.reward = reward

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
            if np.max(diff) <= tolerance:
                break
        return policy


    def train(self, epochs, s_t):
        M = self.M
        T = self.T
        num_env = self.num_env

        num_visits = np.zeros((num_env, self.S, self.A, self.S, self.num_agents))
        curr_states = np.zeros((num_env, self.num_agents), dtype=np.int)
        for i in range(self.num_agents):
            for env in range(self.num_env):
                curr_states[env, i] = int(s_t)
        max_reward = np.zeros((num_env, self.num_agents))
        cumulative_reward = np.zeros((num_env, self.num_agents))
        t = np.zeros(num_env)

        end_regret = np.zeros(num_env)
        time_steps = np.zeros((num_env, epochs))

        for i in range(epochs):
            end_epoch = np.zeros(num_env)
            for env in range(num_env):
                #makes environment that finishes first wait
                if int(end_epoch[env]) == 1 or int(t[env]) == T:
                    continue
                policies = []
                for agent in range(self.num_agents):
                    trans_prob = self.posterior_sample(self.trans_p[env], M[env], self.S, self.A,)
                    policy = self.compute_policy(trans_prob, self.S, self.A, self.reward[env])  # computes the max gain policy
                    policies += [policy]

                num_visits_next = np.copy(num_visits[env])
                while True:
                    for agent in range(self.num_agents):
                        s_t = curr_states[env, agent]
                        a_t = int(policies[agent][s_t])
                        s_next = np.random.choice(range(0,self.S), size=1, p=self.trans_p[env, s_t, a_t, :])
                        # R[env, s_t, a_t] += reward[s_t, a_t]  # TODO: check if there needs to be agent dimension
                        max_reward[env, agent] += np.amax(self.reward[env, s_t, :])
                        cumulative_reward[env, agent] += self.reward[env, s_t, a_t]
                        num_visits_next[s_t, a_t, s_next, agent] += 1

                        if np.sum(num_visits_next[s_t, a_t, :, agent]) >= 2 * np.sum(num_visits[env, s_t, a_t, :, agent]):
                            end_epoch[env] = 1
                        curr_states[env, agent] = s_next

                    t[env] += 1
                    if t[env] == T:
                        # print("evaluation: ", evaluation_epoch_regret)
                        # num_epochs = np.sum([1 for i in time_steps[env] if i != 0])
                        end_env_regret = np.sum(max_reward[env] - cumulative_reward[env])/self.num_agents

                        env_time_steps = len(time_steps[env][time_steps[env] != 0])
                        end_regret[env] = end_env_regret / env_time_steps
                        # print("episodic: ", epoch_regret_sum)
                        break

                    if end_epoch[env]:
                        num_visits[env] = num_visits_next
                        num_visits_current = np.sum(num_visits[env, :, :, :, :], axis=-1)
                        M[env] = np.ones(M[env].shape) + num_visits_current

                        # evaluation_epoch_regret[env, i] = max_reward[env] - cumulative_reward[env]
                        time_steps[env, i] = t[env]
                        break
            if int(np.sum(t)) == T * self.num_env: #TODO: CHECK IF EACH ENVIRONMENT OR ALL ENVIRONMENT
                bayesian_regret = np.average(end_regret)
                print(bayesian_regret)
                return np.mean(bayesian_regret)
                break


if __name__ == "__main__":
    #Define MDP
    state = 10
    action = 5
    seeds = range(130, 150)
    for seed in seeds:
        print("seed: ", seed)
        np.random.seed(seed)

        T = 4000
        num_env = 10
        all_env_rewards = np.zeros((num_env, state, action))
        all_env_trans_p = np.zeros((num_env, state, action, state))

        for env in range(num_env):
            reward = np.abs(np.random.normal(0.0, 1.0, size=(state, action)))
            trans_p = np.zeros([state, action, state])
            for i in range(state):
                for j in range(action):
                    # sample = np.random.gamma(1, 1, state)
                    # trans_p[i, j, :] = sample / np.sum(sample)
                    trans_p[i, j, :] = np.random.dirichlet(np.ones(state))
            all_env_rewards[env] = reward
            all_env_trans_p[env] = trans_p
        total_regret = []

        num_agents = [1, 2, 10, 20, 30]
        for i in num_agents:
            print("agents: ", i)
            psrl = DirichletInfiniteAgent(i, num_env, state, action, T, all_env_trans_p, all_env_rewards)
            regret = psrl.train(1000, int(np.random.randint(0, state, 1)))
            total_regret += [regret]

        # np.savetxt("result" + str(seed) + ".csv", np.column_stack((num_agents, total_regret)), delimiter=",")