import numpy as np
import random
import itertools
import torch

_device_ddtype_tensor_map = {
    'cpu': {
        torch.float32: torch.FloatTensor,
        torch.float64: torch.DoubleTensor,
        torch.float16: torch.HalfTensor,
        torch.uint8: torch.ByteTensor,
        torch.int8: torch.CharTensor,
        torch.int16: torch.ShortTensor,
        torch.int32: torch.IntTensor,
        torch.int64: torch.LongTensor,
        torch.bool: torch.BoolTensor,
    },
    'cuda': {
        torch.float32: torch.cuda.FloatTensor,
        torch.float64: torch.cuda.DoubleTensor,
        torch.float16: torch.cuda.HalfTensor,
        torch.uint8: torch.cuda.ByteTensor,
        torch.int8: torch.cuda.CharTensor,
        torch.int16: torch.cuda.ShortTensor,
        torch.int32: torch.cuda.IntTensor,
        torch.int64: torch.cuda.LongTensor,
        torch.bool: torch.cuda.BoolTensor,
    }
}

# spec = [
#     ('num_agents', int32),               # a simple scalar field
#     ('num_env', int32),
#     ('S', int32),
#     ('A', int32),
#     ('M', int64[:, :, :, :]),
#     ('trans_p', float32[:, :, :, :]),
#     ('reward', float32[:, :, :]),
#     ('R_mean', float32[:, :, :])         # an array field float32[:]
# ]
#
# @jitclass(spec)


def policy_from_mdp(trans_prob, reward, S, A, tolerance=0.01):
    # performs undiscounted value iteration to output an optimal policy
    value_func = torch.zeros(S)
    policy = torch.zeros(S)
    iter = 0
    gamma = 0.99
    while True:
        diff = torch.zeros(S)
        value = value_func
        action_return = reward + gamma * torch.einsum('ijk,k->ij', trans_prob, value_func)  # [S, A]
        value_func, policy = torch.max(action_return, dim=-1)  # [S]
        diff = torch.maximum(diff, torch.abs(value - value_func))  # [S]

        if torch.any(diff.max() <= tolerance) or iter >= 10000:
            break

    return policy  # [S]


class DirichletFiniteAgent:
    def __init__(self, num_agents, num_envs, S, A, trans_p, rewards, optimal_policy):
        """
        num_agents: number of agents
        num_envs: number of envs
        S: size of the state space
        A: size of the action space
        trans_p: transitions probabilities, size [n_env, n_agents, n_S, n_A, n_S]
        rewards: rewards, size [n_envs, n_agents, n_S, n_A]
        policy: optimal policies for all envs, size [n_envs, n_S]
        """
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.S = S
        self.A = A
        self.trans_p = trans_p
        self.rewards = rewards
        self.optimal_policy = optimal_policy
        self.alpha = torch.ones(num_envs, S, A, S)  # concentration parameters of Dirichlet posterior of transition_p
        self.reward_mean = torch.zeros(num_envs, S, A) #mu_0 =0, sigma_0=1 confirmed, x=sample mean, sigma=1, n is max(1, num_visit to (s,a) pair)
        self.reward_scale = torch.ones(num_envs, S, A)

    def posterior_sample(self, alpha, mu, scale, n_sample):
        dist_trans_p = torch.distributions.dirichlet.Dirichlet(alpha)
        dist_reward = torch.distributions.normal.Normal(mu, scale)
        trans_p = dist_trans_p.sample([n_sample])
        rewards = dist_reward.sample([n_sample])
        return torch.transpose(trans_p, 0, 1), torch.transpose(rewards, 0, 1)

    # def posterior_sample(self, transition_prob, alpha, S, A):
    #     dirichlet_trans_p = torch.zeros(transition_prob.shape)
    #     for s, a in itertools.product(range(S), range(A)):
    #         dirichlet_trans_p[s, a] = np.random.dirichlet(alpha[s, a, :])
    #     return dirichlet_trans_p

            # for s in range(S):
            #     value = value_func[s]
            #     action_returns = []
            #     for a in range(A):
            #         action_return = reward[s, a] + gamma * torch.sum(
            #             [trans_prob[s, a, s_next] * value_func[s_next] for s_next in range(S)])  # computes the undiscounted returns
            #         action_returns.append(action_return)
            #     value_func[s] = np.max(action_returns)
            #     policy[s] = np.argmax(action_returns)
            #     diff[s] = max(diff[s], np.abs(value - value_func[s]))

            # iter += 1
            # if iter % 10000 == 0:
            #     print("diff: ", diff)
            #     break
            # if np.max(diff) <= tolerance:
            #     # print(value_func)
            #     break


    def train(self, episodes, horizon):
        # curr_states = torch.zeros((self.num_envs, self.num_agents), dtype=np.int64)
        evaluation_episodic_regret = torch.zeros((self.num_envs, episodes, self.num_agents))
        # optimal_return = torch.zeros(self.num_envs)
        # cum_return = torch.zeros((self.num_envs, self.num_agents))
        cum_regret = torch.zeros(self.num_envs, self.num_agents)
        num_visits = torch.zeros((self.num_envs, self.S, self.A, self.S))
        model_reward = torch.zeros((self.num_envs, self.S, self.A))
        t = torch.ones(self.num_envs)

        for i in range(episodes):
            # initialize num_visits and state tracking for each agent in each env
            curr_states = torch.randint(0, self.S, size=[self.num_envs, self.num_agents])

            # (Alg) sample MDP's from the posterior 
            # [n_envs, n_agents, n_S, n_A, n_S], [n_envs, n_agents, n_S, n_A]
            sampled_trans_p, sampled_reawrds = self.posterior_sample(
                self.alpha, self.reward_mean, self.reward_scale, self.num_agents)  

            # extract optimal policies from sampled MDP's: [n_envs, n_agents, n_S]
            policy = torch.zeros((self.num_envs, self.num_agents, self.S), dtype=torch.int64)
            for env in range(self.num_envs):
                for agent in range(self.num_agents):
                    policy[env, agent, :] = policy_from_mdp(
                        sampled_trans_p[env, agent], sampled_reawrds[env, agent], self.S, self.A)

            for _ in range(horizon):
                s_t = curr_states  # [n_envs, n_agents]
                a_t = policy[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1),
                    torch.arange(self.num_agents).unsqueeze(0).unsqueeze(-1),
                    s_t.unsqueeze(-1)].squeeze()  # [n_envs, n_agents]
                optimal_a_t = self.optimal_policy[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1),
                    torch.arange(self.num_agents).unsqueeze(0).unsqueeze(-1),
                    s_t.unsqueeze(-1)].squeeze()  # [n_envs, n_agents]
                # categorical_prob = self.trans_p 
                trans_p = self.trans_p[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 
                    torch.arange(self.num_agents).unsqueeze(0).unsqueeze(-1).unsqueeze(-1), 
                    s_t.unsqueeze(-1).unsqueeze(-1), 
                    a_t.unsqueeze(-1).unsqueeze(-1)].squeeze()  # [n_envs, n_agents, n_S]
                s_next = torch.distributions.categorical.Categorical(trans_p).sample()  # [n_envs, n_agents]
                curr_states = s_next
                reward = self.rewards[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 
                    torch.arange(self.num_agents).unsqueeze(0).unsqueeze(-1).unsqueeze(-1), 
                    s_t.unsqueeze(-1).unsqueeze(-1), 
                    a_t.unsqueeze(-1).unsqueeze(-1)].squeeze()  # [n_envs, n_agents]
                optimal_reward = self.rewards[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 
                    torch.arange(self.num_agents).unsqueeze(0).unsqueeze(-1).unsqueeze(-1), 
                    s_t.unsqueeze(-1).unsqueeze(-1), 
                    optimal_a_t.unsqueeze(-1).unsqueeze(-1)].squeeze()  # [n_envs, n_agents]

                cum_regret = optimal_reward - reward

                # record observed transitions and rewards
                for agent in range(self.num_agents):
                    num_visits[
                        torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                        s_t[:, agent].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                        a_t[:, agent].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                        s_next[:, agent].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)] += 1
                    model_reward[
                        torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1),
                        s_t[:, agent].unsqueeze(-1).unsqueeze(-1),
                        a_t[:, agent].unsqueeze(-1).unsqueeze(-1)] += reward[:, agent].unsqueeze(-1).unsqueeze(-1)

            # update the posterior of the Dirichlet alpha of transitions
            self.alpha = torch.ones(self.alpha.shape) + num_visits
            # update the posterior of the Gaussian of rewards
            count = torch.ones(self.num_envs, self.S, self.A) + torch.sum(num_visits, dim=-1)  # [n_envs, n_S, n_A]
            self.reward_mean = model_reward / count
            self.reward_scale = 1 / torch.sqrt(count)

        # evaluate episodic regret
        per_step_regret = cum_regret / (episodes * horizon)  # [n_envs, n_agents]
        per_step_per_agent_regret = per_step_regret / self.num_agents  # [n_envs]
        per_step_per_agent_Bayesian_regret = per_step_per_agent_regret.mean()
        print("bayesian regret: ", per_step_per_agent_Bayesian_regret)

        return per_step_per_agent_Bayesian_regret



            #for env in range(self.num_envs):
            #    # initialize num_visits and state tracking for each agent
            #    for agent in range(self.num_agents):
            #        curr_states[env, agent] = int(np.random.randint(0, self.S, 1))

            #    for agent in range(self.num_agents):
            #        #evaluation as in sample from M multiple times
            #        trans_prob = self.posterior_sample(self.trans_p[env], self.alpha[env], self.S, self.A)
            #        reward = np.zeros((self.S, self.A))
            #        for s in range(self.S):
            #            for a in range(self.A):
            #                reward[s, a] = np.abs(np.float(np.random.normal(self.R_mean[env, s, a], 1, size=1))) #each s,a pair has its own posterior
            #        if horizon != 1:
            #            policy = self.compute_policy(trans_prob, self.S, self.A, reward, horizon)

            #        for _ in range(horizon):
            #            s_t = curr_states[env, agent]
            #            if horizon != 1:
            #                a_t = int(policy[s_t])
            #            else:
            #                a_t = int(np.argmax(reward[s_t, :]))
            #            s_next = np.random.choice(range(0, self.S), size=1, p=self.trans_p[env, s_t, a_t, :])
            #            R[env, s_t, a_t] += self.reward[env, s_t, a_t]
            #            max_reward[env, agent] += np.amax(self.reward[env, s_t, :]) #max reward is the maximum reward of the god generated MDP
            #            cumulative_reward[env, agent] += self.reward[env, s_t, a_t]
            #            num_visits[env, s_t, a_t, s_next, agent] += 1
            #            curr_states[env, agent] = int(s_next)
            #        # evaluation_episodic_regret[i, agent] = self.evaluate(policy, 50, horizon)
            #        evaluation_episodic_regret[env, i, agent] = max_reward[env, agent] - cumulative_reward[env, agent]
            #        max_reward[env, agent] = 0
            #        cumulative_reward[env, agent] = 0

            #    # compute a num visits parameter for dirichlet
            #    num_visits_current = np.sum(num_visits[env, :, :, :, :], axis=-1)
            #    self.alpha[env] = torch.ones(self.alpha[env].shape) + num_visits_current
            #    #update posterior for reward
            #    n = np.maximum(np.ones(num_visits[env, :, :, 0, 0].shape), np.sum(num_visits[env], axis=(-2, -1)))
            #    self.R_mean[env] = np.multiply(np.divide(1, (np.divide(1, n) + 1)), np.divide(R[env], n))  #TODO: CHANGE
            #    t[env] += horizon
        ## print("evaluation: ", evaluation_episodic_regret)
        #avg_over_env_reg = np.sum(evaluation_episodic_regret, axis=0)/self.num_envs
        #episodic_regret_avg_over_agent = np.sum(avg_over_env_reg, axis=-1)/self.num_agents
        ## print("episodic: ", episodic_regret_avg_over_agent)
        #bayesian_regret = np.sum(episodic_regret_avg_over_agent)/np.mean(t)
        #print("bayesian: ", bayesian_regret)
        #return bayesian_regret


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type(
            _device_ddtype_tensor_map['cuda'][torch.get_default_dtype()])
        torch.cuda.set_device(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True

    num_states = 20
    num_actions = 10
    num_envs = 3
    num_episodes = 4
    horizon = 10
    #TODO: scale up the state and action
    #uniform sample over all the state
    #set horizon=1, initial state for each agent drawn from uniform distribution across all states
    # seeds = range(100, 101)
    seeds = (100,)
    for seed in seeds:
        # deterministic settings for current seed
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print("seed: ", seed)

        # define MDP
        all_env_rewards = torch.abs(torch.randn(num_envs, num_states, num_actions))
        dist_trans_p = torch.distributions.dirichlet.Dirichlet(torch.ones(num_states))
        all_env_trans_p = dist_trans_p.sample([num_envs, num_states, num_actions])

        # compute optimal policy for each env
        all_env_optimal_policy = torch.zeros((num_envs, num_states), dtype=torch.int64)
        for env in range(num_envs):
            all_env_optimal_policy[env] = policy_from_mdp(
                all_env_trans_p[env], all_env_rewards[env], num_states, num_actions)

        regrets = []

        # list_num_agents = [1, 2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        list_num_agents = [2, 3]
        for num_agents in list_num_agents:
            print("num of agents: ", num_agents)
            # [n_envs, n_agents, n_S, n_A, n_S]
            all_env_agent_rewards = all_env_rewards.unsqueeze(1).repeat(1, num_agents, 1, 1)
            all_env_agent_trans_p = all_env_trans_p.unsqueeze(1).repeat(1, num_agents, 1, 1, 1)
            all_env_agent_optimal_policy = all_env_optimal_policy.unsqueeze(1).repeat(1, num_agents, 1)
            psrl = DirichletFiniteAgent(
                num_agents, num_envs, num_states, num_actions, 
                all_env_agent_trans_p, all_env_agent_rewards, all_env_agent_optimal_policy)
            regret = psrl.train(num_episodes, horizon)
            regrets.append(regret)

        total_regret = torch.stack(regrets)
        total_regret_np = total_regret.cpu().detach().numpy()
        np.savetxt("result" + str(seed) + ".csv", np.column_stack((list_num_agents, total_regret_np)), delimiter=",")
