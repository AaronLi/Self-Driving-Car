import os
from car_environment import SimpleCarEnvironment
from tf_agents.environments import wrappers, tf_py_environment
from tf_agents.policies import policy_loader
import tensorflow as tf
import imageio
from pygame import *
SimpleCarEnvironment.SIM_FPS = 30

eval_py_env = SimpleCarEnvironment()
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

policy = tf.saved_model.load(os.path.join(os.getcwd(), 'policy_archive', 'overnight_final'))

def make_video():
    num_episodes = 5
    video_filename = 'imageio.mp4'
    with imageio.get_writer(video_filename, fps=SimpleCarEnvironment.SIM_FPS) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())

def evaluate_forever():
    running = True

    time_step = eval_env.reset()

    eval_image = eval_py_env.draw()

    screen = display.set_mode(eval_image.get_size())

    clockity = time.Clock()

    while running:
        for e in event.get():
            if e.type == QUIT:
                running = False
                break
            if e.type == KEYDOWN:
                if e.key == K_SPACE:
                    time_step = eval_env.reset()

        screen.blit(eval_image, (0, 0))
        display.flip()
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        eval_image = eval_py_env.draw()
        if time_step.is_last():
            time_step = eval_env.reset()
        clockity.tick(SimpleCarEnvironment.SIM_FPS)

if __name__ == '__main__':
     evaluate_forever()
    #make_video()