from tqdm import tqdm
from car_environment import SimpleCarEnvironment
import matplotlib.pyplot as plt
from tf_agents.environments import tf_py_environment, wrappers
from tf_agents.networks import value_network
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.agents.ppo import ppo_clip_agent, ppo_agent, ppo_actor_network
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.utils import common
import reverb
import imageio
import os
import tensorflow as tf

num_iterations = 30 # @param {type:"integer"}
collect_episodes_per_iteration = 3 # @param {type:"integer"}
replay_buffer_capacity = 5000 # @param {type:"integer"}

fc_layer_params = (256, 128, 64, 64, 32)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 15 # @param {type:"integer"}
SimpleCarEnvironment.SIM_FPS = 10
SIM_DURATION_S = 30

if __name__ == '__main__':
  train_py_env = wrappers.TimeLimit(SimpleCarEnvironment(), SimpleCarEnvironment.SIM_FPS * SIM_DURATION_S)
  eval_py_env = wrappers.TimeLimit(SimpleCarEnvironment(), SimpleCarEnvironment.SIM_FPS * SIM_DURATION_S)

  train_env = tf_py_environment.TFPyEnvironment(train_py_env)
  eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

  # actor_net = actor_distribution_network.ActorDistributionNetwork(
  #     train_env.observation_spec(),
  #     train_env.action_spec(),
  #     fc_layer_params=fc_layer_params)

  actor_net = ppo_actor_network.PPOActorNetwork().create_sequential_actor_net(
    fc_layer_units=fc_layer_params,
    action_tensor_spec=train_env.action_spec(),
  )

  value_net = value_network.ValueNetwork(
      train_env.observation_spec()
  )

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  train_step_counter = tf.Variable(0)

  tf_agent = ppo_agent.PPOAgent(
      train_env.time_step_spec(),
      train_env.action_spec(),
      actor_net=actor_net,
      value_net=value_net,
      optimizer=optimizer,
      normalize_rewards=True,
      train_step_counter=train_step_counter
  )

  tf_agent.initialize()

  eval_policy = tf_agent.policy
  collect_policy = tf_agent.collect_policy

  def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

      time_step = environment.reset()
      episode_return = 0.0
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward
      total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


  table_name = 'uniform_table'
  replay_buffer_signature = tensor_spec.from_spec(
        tf_agent.collect_data_spec)
  replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)
  table = reverb.Table(
      table_name,
      max_size=replay_buffer_capacity,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(1),
      signature=replay_buffer_signature)

  reverb_server = reverb.Server([table])

  replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
      tf_agent.collect_data_spec,
      table_name=table_name,
      sequence_length=None,
      local_server=reverb_server)

  rb_observer = reverb_utils.ReverbAddEpisodeObserver(
      replay_buffer.py_client,
      table_name,
      replay_buffer_capacity
  )

  def collect_episode(environment, policy, num_episodes):

    driver = py_driver.PyDriver(
      environment,
      py_tf_eager_policy.PyTFEagerPolicy(
        policy, use_tf_function=True),
      [rb_observer],
      max_episodes=num_episodes)
    initial_time_step = environment.reset()
    driver.run(initial_time_step)

  # (Optional) Optimize by wrapping some of the code in a graph using TF function.
  tf_agent.train = common.function(tf_agent.train)

  # Reset the train step
  tf_agent.train_step_counter.assign(0)

  # Evaluate the agent's policy once before training.
  avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
  returns = [avg_return]

  print('training loop started')
  for _ in tqdm(range(num_iterations)):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(
        train_py_env, tf_agent.collect_policy, collect_episodes_per_iteration)
    # Use data from the buffer and update the agent's network.
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)
    train_loss = tf_agent.train(experience=trajectories)  

    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
      print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
      avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
      print('step = {0}: Average Return = {1}'.format(step, avg_return))
      returns.append(avg_return)

  policy_dir = os.path.join(os.getcwd(), 'policy')
  policy_saver = PolicySaver(tf_agent.policy)

  policy_saver.save(policy_dir)
  num_episodes = 5
  video_filename = 'imageio.mp4'
  with imageio.get_writer(video_filename, fps=SimpleCarEnvironment.SIM_FPS) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())
      while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(eval_py_env.render())

  #steps = range(0, num_iterations + 1, eval_interval)
  plt.plot(returns)
  plt.ylabel('Average Return')
  plt.xlabel('Step')
  plt.show()