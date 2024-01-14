from pathlib import Path

import gym
import d4rl
import numpy as np
import itertools
import os
import torch
from tqdm import trange



from pex.algorithms.pex import PEX
from pex.algorithms.iql_online import IQL_online
from pex.networks.policy import GaussianPolicy
from pex.networks.value_functions import DoubleCriticNetwork, ValueNetwork
from pex.utils.util import (
    set_seed, ReplayMemory, torchify, eval_policy, torchify, DEFAULT_DEVICE,
    get_batch_from_dataset_and_buffer,
    eval_policy, set_default_device, get_env_and_dataset, DEFAULT_DEVICE, epsilon_greedy_sample,
                            extract_sub_dict)

# from pex.utils.util import (DEFAULT_DEVICE, epsilon_greedy_sample,
#                             extract_sub_dict)

#########################################
import torch.nn.functional as F
import time
import utils
from coolname import generate_slug

#VAE
from vae import VAE
#########################################
############## logger
from log import Logger
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def homeostasis(x_t, time_step, x_bar, x_squared_bar, x_plus_bar, rho, device):
    #print("x_t : ", x_t)
    #print('time_step : ', time_step)
    #print('x_bar : ', x_bar)
    #print('x_squared_bar : ', x_squared_bar)
    #print('x_plus_bar : ', x_plus_bar)
    #print('rho : ', rho)
    Tau = np.minimum(time_step, 100/rho)
    #print("Tau : ", Tau)
    x_bar = (1 - 1/Tau) * x_bar + 1/Tau * x_t
    #print("x_bar : ", x_bar)
    x_squared_bar = (1 - 1/Tau) * x_squared_bar + 1/Tau * ((x_t - x_bar) ** 2)
    #print("x_squared_bar : ", x_squared_bar)
    x_plus = np.exp((x_t - x_bar)/ np.square(x_squared_bar))
    #print("x_plus : ", x_plus)
    x_plus_bar = (1 - 1/Tau) * x_plus_bar + 1/Tau * x_plus
    ##y_t = torch.bernoulli(np.minimum(1, rho*x_plus/x_plus_bar))
    #print("rho*x_plus/x_plus_bar : ", rho*x_plus/x_plus_bar)
    y_t_input = np.minimum(1, rho*x_plus/x_plus_bar)
    #print("y_t_input : ", y_t_input)
    y_t = torch.bernoulli(torch.tensor(y_t_input, dtype=torch.float32).to(device))
    #print("y_t : ", y_t)
    #print("homeostasis_y_t : ", y_t.cpu().numpy())
    return x_bar, x_squared_bar, x_plus_bar, y_t



def main(args):
    torch.set_num_threads(1)

    if os.path.exists(args.log_dir):
        print(f"The directory {args.log_dir} exists. Please specify a different one.")
        return
    else:
        print(f"Creating directory {args.log_dir}")
        os.mkdir(args.log_dir)


    env, dataset, reward_transformer = get_env_and_dataset(args.env_name, args.max_episode_steps)
    dataset_size = dataset['observations'].shape[0]
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]

    #### VAE
    state_dim = obs_dim
    action_dim = act_dim
    max_action = float(env.action_space.high[0])


    #####################################################
    # make directory
    base_dir = 'runs'
    utils.make_dir(base_dir)
    base_dir = os.path.join(base_dir, args.work_dir)
    utils.make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, args.env + '_' + args.dataset)
    utils.make_dir(args.work_dir)

    # ts = time.gmtime()
    # ts = time.strftime("%m-%d-%H:%M", ts)
    #             # + str(args.batch_size) + '-s' + str(args.seed) + '-b' + str(args.beta) + \
    # exp_name = str(args.env) + '-' + str(args.dataset) + '-' + ts + '-bs' \
    #            + str(args.batch_size_vae) + '-s' + str(args.seed_vae) + '-b' + str(args.beta_vae) + \
    #            '-h' + str(args.hidden_dim_vae) + '-lr' + str(args.lr_vae) + '-wd' + str(args.weight_decay_vae)
    # exp_name += '-' + generate_slug(2)
    # if args.notes is not None:
    #     exp_name = args.notes + '_' + exp_name
    # args.work_dir = args.work_dir + '/' + exp_name
    # utils.make_dir(args.work_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    utils.snapshot_src('.', os.path.join(args.work_dir, 'src'), '.gitignore')
    logger = Logger(args.work_dir, use_tb=True)

    #######################################################


    # VAE
    #vae = VAE(state_dim, action_dim, args.latent_dim if args.latent_dim else 2 * action_dim, max_action).to(device)
    # vae = VAE(obs_dim, action_dim, args.latent_dim_vae if args.latent_dim_vae else 2 * action_dim, max_action).to(device)
    # vae.load_state_dict(torch.load(args.vae_model_path))
    #
    # vae_online = VAE(obs_dim, action_dim, args.latent_dim_vae if args.latent_dim_vae else 2 * action_dim, max_action).to(device)
    # vae_online.load_state_dict(torch.load(args.vae_model_path))
    # vae_online_optimizer = torch.optim.Adam(vae_online.parameters(), lr=args.lr_vae, weight_decay=args.weight_decay_vae)

    #vae.eval()
    ######################################################

    if args.seed is not None:
        set_seed(args.seed, env=env)

    if torch.cuda.is_available():
        set_default_device()

    action_space = env.action_space
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, action_space=action_space, scale_distribution=False, state_dependent_std=False)

    algorithm_option = args.algorithm.upper()

    if algorithm_option == "SCRATCH":
        double_buffer = False
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=None
        )

    elif algorithm_option == "BUFFER":
        double_buffer = True
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=None
        )

    elif algorithm_option == "DIRECT":
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path
        )

    elif algorithm_option == "PEX":
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = PEX(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path,
            inv_temperature=args.inv_temperature,
        )

    memory = ReplayMemory(args.replay_size, args.seed)

    total_numsteps = 0

    ##############################
    count_vae = 0
    count_vae_online = 0
    ##############################

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        ##############################################################
        ##action = alg.policy_offline.act(torchify(state).to(DEFAULT_DEVICE).unsqueeze(0), deterministic=True)
        ##action = action.squeeze(0).detach().cpu().numpy()

        reward = 0

        action_m = 0
        explore_count = 1

        ##### comment out by johnny #####
        #####off_policy_count = 0
        #####exploration_policy_count = 0

        explore_flag = False
        exploit_count = 1
        x_bar, x_squared_bar, x_plus_bar = 0, 1, 1
        ##self.target_rate_rho = 0.01  #### 0.1, 0.01, 0.001, 0.0001
        ##self.target_rate_rho = 0.0001  #### 0.1, 0.01, 0.001, 0.0001
        target_rate_rho = 0.025  #### 0.1, 0.01, 0.001, 0.0001
        ##self.target_rate_rho = 0.001  #### 0.1, 0.01, 0.001, 0.0001
        explore_fixed_steps = 30
        ##self.update_timestep = 30
        update_timestep = 90
        old_value, old_m_value, old_h_target_value, old_l_target_value = 0, 0, 0, 0
        value_h_reward, value_h_reward_m, value_l_reward = 0, 0, 0
        gamma_tigeer = 1
        gamma = 0.99  # discount factor

        ##############################################################

        while not done:

            ######################################################################

            if explore_flag == True:
                # print("if self.explore_flag == True:")
                if explore_count % explore_fixed_steps == 0:
                    # print("self.explore_count : ", self.explore_count)
                    # self.explore_count = 0
                    explore_count = 1
                    x_bar, x_squared_bar, x_plus_bar = 0, 1, 1
                    explore_flag = False
                else:
                    # print("if self.explore_count_else")
                    # print("self.explore_count : ", self.explore_count)
                    explore_count += 1

            if explore_flag == False:
                #print("if self.explore_flag == False:")
                if  exploit_count % update_timestep == 0:
                    #next_goal_t = torch.min(torch.max(actor_target_h(next_state), -max_goal), max_goal)
                    #q_target_1, q_target_2 = critic_target_h(next_state, next_goal_t)
                    #q_target_h = torch.min(q_target_1, q_target_2)
                    #q_target_h_n = q_target_h.detach().cpu().numpy()[0][0]

                    ##q_target_h_n = self.agent.read_target_value(self.replay_iter, self.global_step)

                    ##### comment out by johnny #####
                    ####q_target_h_n = agent.read_target_value(time_step.observation,
                    ####                                            meta,
                    ####                                            global_step)
                    ####print("torchify(state).to(DEFAULT_DEVICE).unsqueeze(0) : ", torchify(state).to(DEFAULT_DEVICE).unsqueeze(0))
                    action = alg.policy_offline.act(torchify(state).to(DEFAULT_DEVICE).unsqueeze(0), deterministic=True)
                    q_target_h_n = alg.critic_offline.min(torchify(state).to(DEFAULT_DEVICE).unsqueeze(0), action)

                    # print("action : ", action)
                    # print("q_target_h_n : ", q_target_h_n)
                    with torch.no_grad():
                        #############q_target_h_n = self.agent_exploration.read_target_value(self.replay_iter, self.global_step)
                        value_h_promise_discrepancy = old_h_target_value - value_h_reward - (gamma_tigeer * q_target_h_n)
                        #print("value_h_promise_discrepancy : ", value_h_promise_discrepancy)
                        abs_value_h_promise_discrepancy = np.absolute(value_h_promise_discrepancy.cpu().numpy())

                    # print("abs_value_h_promise_discrepancy : ", abs_value_h_promise_discrepancy)
                    ##value_h_promise_discrepancy = apply_normalizer(abs_value_h_promise_discrepancy, value_h_promise_discrepancy_normalizer)

                    ##record_logger(args=[value_h_promise_discrepancy], option='only_h_values_variance', step=t-start_timestep)
                    ##record_logger(args=[abs_value_h_promise_discrepancy], option='only_h_values_variance', step=t-start_timestep)
                    old_h_target_value = q_target_h_n
                    value_h_reward = 0
                    gamma_tigeer = 1

                    ##homeostasis(x_t, time_step, x_bar, x_squared_bar, x_plus_bar, rho, device)
                    #print("homeostasis")
                    x_bar, x_squared_bar, x_plus_bar, y_t = homeostasis(abs_value_h_promise_discrepancy, exploit_count, x_bar, x_squared_bar, x_plus_bar, target_rate_rho, device)
                    ##print("y_t : ", y_t)
                    y_t_n = y_t.item()
                    #print("y_t_n : ", y_t_n)
                    ##if y_t_n == 1:
                    if y_t_n == 0:
                        #print("if y_t_n == 0:")
                        action_m = 1
                        ##exploit_count += 1
                    else:
                        #print("self.exploit_count = 1")
                        action_m = 0
                        exploit_count = 1
                        explore_flag = True

                else:
                    #print("if explore_flag == False: if y_t == 1: else_222 ")
                    ##########################################################################
                    ############temp_episode_reward_h = episode_reward_h.cpu().numpy()[0]
                    #############temp_episode_reward_h = intr_sf_reward.mean().item()
                    #############temp_episode_reward_h = intr_ent_reward.mean().item()
                    ##########################################################################
                    #intr_sf_reward = self.agent.compute_intr_sf_reward(replay_buffer_task, replay_buffer_next_obs, self.global_step)
                    #intr_ent_reward = self.agent.compute_intr_ent_reward(replay_buffer_task, replay_buffer_next_obs, self.global_step)

                    ##### comment out by johnny #####
                    # obs = torch.as_tensor(time_step.observation, device=device).unsqueeze(0)
                    # h = agent.encoder(obs)
                    # intr_sf_reward = agent.compute_intr_sf_reward(torch.as_tensor([meta['task']],device=device), h, global_step)

                    ##intr_ent_reward = self.agent_exploration.compute_intr_ent_reward(meta['task'], h, self.global_step)

                    ##### comment out by johnny #####
                    # temp_episode_reward_h = intr_sf_reward

                    temp_episode_reward_h = reward
                    temp_episode_reward_h = gamma_tigeer * temp_episode_reward_h
                    value_h_reward += temp_episode_reward_h
                    gamma_tigeer = gamma_tigeer * gamma
                    #print("self.exploit_count += 1")
                    exploit_count += 1

            ######################################################################


            #action = alg.select_action(torchify(state).to(DEFAULT_DEVICE)).detach().cpu().numpy()
            ##### comment out by johnny #####
            #####action, count_vae, count_vae_online, action_original = alg.select_action_vae(vae, vae_online, count_vae, count_vae_online, args.beta, torchify(state).to(DEFAULT_DEVICE))

            # sample action
            if action_m == 1:
                action = alg.policy_offline.act(torchify(state).to(DEFAULT_DEVICE).unsqueeze(0), deterministic=True)
                #####action = action.squeeze(0).detach().cpu().numpy()
                count_vae = count_vae + 1
                #count_vae_online = count_vae_online + 1
            else:
                dist = alg.policy(torchify(state).to(DEFAULT_DEVICE).unsqueeze(0))
                # if evaluate:
                #     a2 = epsilon_greedy_sample(dist, eps=0.1)
                # else:
                #     a2 = epsilon_greedy_sample(dist, eps=1.0)
                action = epsilon_greedy_sample(dist, eps=1.0)
                #####action = alg.select_action(torchify(state).to(DEFAULT_DEVICE)).detach().cpu().numpy()
                count_vae_online = count_vae_online + 1
                #count_vae = count_vae + 1

            action = action.squeeze(0).detach().cpu().numpy()

            ####################################################
            # Variational Auto-Encoder Training

            ### recon, mean, std = vae(train_states, train_actions)
            #print("state : ", state)
            #print("action : ", action)
            #print("state : ", torch.from_numpy(state))
            #print("action : ", torch.from_numpy(action))
            ######observations = torchify(state).to(DEFAULT_DEVICE)
            ######observations = observations.unsqueeze(0)
            ######recon, mean, std = vae_online(observations, action_original)

            #recon_loss = F.mse_loss(recon, train_actions)
            ######recon_loss = F.mse_loss(recon, action_original)
            ######KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            ######vae_loss = recon_loss + args.beta * KL_loss

            #logger.log('train/recon_loss', recon_loss, step=step)
            #logger.log('train/KL_loss', KL_loss, step=step)
            #logger.log('train/vae_loss', vae_loss, step=step)

            #optimizer.zero_grad()
            ######vae_online_optimizer.zero_grad()
            ######vae_loss.backward()
            #optimizer.step()
            ######vae_online_optimizer.step()
            ####################################################

            if len(memory) > args.initial_collection_steps:
                for i in range(args.updates_per_step):
                    alg.update(*get_batch_from_dataset_and_buffer(dataset, memory, args.batch_size, double_buffer))

            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            reward_for_replay = reward_transformer(reward)

            ######################## Logger
            logger.log('train/reward', reward, step=total_numsteps)
            logger.log('train/count_vae', count_vae, step=total_numsteps)
            logger.log('train/count_vae_online', count_vae_online, step=total_numsteps)
            ##################################

            terminal = 0 if episode_steps == env._max_episode_steps else float(done)
            memory.push(state, action, reward_for_replay, next_state, terminal)
            state = next_state

            if total_numsteps % args.eval_period == 0 and args.eval is True:

                print("Episode: {}, total env-steps: {}".format(i_episode, total_numsteps))
                #eval_policy(env, args.env_name, alg, args.max_episode_steps, args.eval_episode_num)
                return_mean, return_std, normalized_return_mean, normalized_return_std = eval_policy(env, args.env_name, alg, args.max_episode_steps, args.eval_episode_num)
                logger.log('train_ep/return_mean', return_mean, step=total_numsteps)
                logger.log('train_ep/return_std', return_std, step=total_numsteps)
                logger.log('train_ep/normalized_return_mean', normalized_return_mean, step=total_numsteps)
                logger.log('train_ep/normalized_return_std', normalized_return_std, step=total_numsteps)

        if total_numsteps > args.total_env_steps:
            break


        env.close()

    torch.save(alg.state_dict(), args.log_dir + '/{}_online_ckpt'.format(args.algorithm))

    logger._sw.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--algorithm', required=True)  # ['direct', 'buffer', 'pex']
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--target_update_rate', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=10.0,
                        help='IQL inverse temperature')
    parser.add_argument('--ckpt_path', default=None,
                    help='path to the offline checkpoint')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--total_env_steps', type=int, default=1000001, metavar='N',
                        help='total number of env steps (default: 1000000)')
    parser.add_argument('--initial_collection_steps', type=int, default=5000, metavar='N',
                        help='Initial environmental steps before training starts (default: 5000)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--inv_temperature', type=float, default=10, metavar='G',
                        help='inverse temperature for PEX action selection (default: 10)')
    parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--eval_period', type=int, default=10000)
    parser.add_argument('--eval_episode_num', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max_episode_steps', type=int, default=1000)

    ####################################################################################
    # dataset
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert

    # work dir
    parser.add_argument('--work_dir', type=str, default='train_vae')

    # model
    parser.add_argument('--model', default='VAE', type=str)
    parser.add_argument('--hidden_dim_vae', type=int, default=750)
    #parser.add_argument('--beta', type=float, default=0.5)

    # train
    #parser.add_argument('--num_iters', type=int, default=int(1e5))
    parser.add_argument('--batch_size_vae', type=int, default=256)
    parser.add_argument('--lr_vae', type=float, default=1e-3)
    parser.add_argument('--weight_decay_vae', default=0, type=float)
    #parser.add_argument('--scheduler', default=False, action='store_true')
    #parser.add_argument('--gamma', default=0.95, type=float)
    #parser.add_argument('--no_max_action', default=False, action='store_true')
    #parser.add_argument('--clip_to_eps', default=False, action='store_true')
    #parser.add_argument('--eps', default=1e-4, type=float)
    #parser.add_argument('--latent_dim', default=None, type=int, help="default: action_dim * 2")
    #parser.add_argument('--no_normalize', default=False, action='store_true', help="do not normalize states")


    parser.add_argument('--seed_vae', type=int, default=0)
    # VAE
    #parser.add_argument('--vae_model_path', default=None, type=str)
    #parser.add_argument('--vae_model_path', default='models/vae_trained_models/vae_model_antmaze_medium-diverse.pt', type=str)
    parser.add_argument('--vae_model_path', default='models/vae_trained_models/vae_model_halfcheetah_medium-replay.pt', type=str)

    parser.add_argument('--beta_vae', default=0.5, type=float)
    parser.add_argument('--latent_dim_vae', default=None, type=int)
    #parser.add_argument('--iwae', default=False, action='store_true')
    #parser.add_argument('--num_samples', default=1, type=int)


    main(parser.parse_args())
