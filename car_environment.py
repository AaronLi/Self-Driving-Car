from math import hypot
import time
import matplotlib.pyplot as plt
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from shapes import *
from scipy import interpolate
import random
import tqdm
from pygame import Rect, Surface, image, draw, surfarray
from tf_agents.trajectories import time_step as ts

class SimpleCarEnvironment(py_environment.PyEnvironment):
    SIM_FPS = 90
    def __init__(self):
        self.world = []
        self.car = Circle((0, 0), 20)
        self.car_velocity = [0, 0]
        self.car_direction = 0
        self._episode_ended = False
        self.steering_amount = 30
        self.max_speed = 100
        self.current_sim_steps = 0
        self.sim_score = 0
        self.travelled = 0
        self.left_wall = -500
        self.right_wall = 500
        self.top_wall = -500
        self.bottom_wall = 500
        self.last_command = [0.5, 0.5]
        self.terminate_simulation = False
        self._action_spec = array_spec.BoundedArraySpec(shape=(2, ), dtype=np.float32, minimum = 0, maximum=1, name='action') # speed, steering
        self._observation_spec = array_spec.BoundedArraySpec(shape=(65, ), dtype=np.float32, minimum = 0, maximum=1, name='observation')
        self.create_environment()
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def create_environment(self):
        top_wall = Line((self.left_wall, self.top_wall), (self.right_wall, self.top_wall))
        right_wall = Line((self.right_wall, self.top_wall), (self.right_wall, self.bottom_wall))
        bottom_wall = Line((self.left_wall, self.bottom_wall), (self.right_wall, self.bottom_wall))
        left_wall = Line((self.left_wall, self.top_wall), (self.left_wall, self.bottom_wall))
        self.world = [top_wall, right_wall, bottom_wall, left_wall]

        for i in range(20):
            obstacle = 0#random.randint(0, 2)

            radius = random.random() * 80 + 20
            radius_keepaway = np.ceil(radius) + 10
            shape_x = random.randint(self.left_wall, -radius_keepaway) if random.randint(0, 1) else random.randint(radius_keepaway, self.right_wall)
            shape_y = random.randint(self.top_wall, -radius_keepaway) if random.randint(0, 1) else random.randint(radius_keepaway, self.bottom_wall)
            if obstacle == 0:
                self.world.append(Circle((shape_x, shape_y), radius))
            # else:
            #     self.world.append(RegularPolygon((shape_x, shape_y), random.random() * 100, random.randint(3, 6)))

    def _reset(self):
        self.car_velocity = [0, 0]
        self.car = Circle((0, 0), 20)
        self.current_sim_steps = 0
        self.car_direction = 0
        self.travelled = 0
        self.last_command = [0.5, 0.5]
        self.sim_score = 0
        self.terminate_simulation = False
        self.create_environment()
        return ts.restart(np.array(self.car_observation(-32, 32)))
    
    def _step(self, action):
        if self.terminate_simulation:
            return self._reset()
        self.last_command = action
        self.car_velocity[0] -= self.car_velocity[0] * 0.9 * (1.0/self.SIM_FPS)
        self.car_velocity[1] -= self.car_velocity[1] * 0.9 * (1.0/self.SIM_FPS)

        self.car_direction += (2*action[1]-1) * self.steering_amount * (1.0/self.SIM_FPS)

        self.car_velocity[0] += (2*action[0]-1) * self.max_speed * np.cos(self.car_direction / 180 * np.pi) * (1.0/self.SIM_FPS)
        self.car_velocity[1] += (2*action[0]-1) * self.max_speed * np.sin(self.car_direction / 180 * np.pi) * (1.0/self.SIM_FPS)

        self.car.pos[0] += self.car_velocity[0] * (1.0/self.SIM_FPS)
        self.car.pos[1] += self.car_velocity[1] * (1.0/self.SIM_FPS)
        
        self.current_sim_steps += 1

        observation = self.car_observation(-32, 32)

        velocity_forward_ratio = np.dot(self.car_velocity, [np.cos(self.car_direction / 180 * np.pi), np.sin(self.car_direction / 180 * np.pi)]) / np.linalg.norm(self.car_velocity)

        if velocity_forward_ratio < 0:
            velocity_forward_ratio = (1 + velocity_forward_ratio) * 0.3

        # self.travelled += hypot(self.car_velocity[0], self.car_velocity[1])

        #self.sim_score += min((self.travelled / (float(self.current_sim_steps)/self.SIM_FPS)) / self.max_speed, 1) # distance over time

        reward = hypot(self.car_velocity[0], self.car_velocity[1]) * velocity_forward_ratio

        #reward = self.sim_score#hypot(self.car_velocity[0], self.car_velocity[1]) * velocity_forward_ratio
        if self.collisions():
            self.terminate_simulation = True
            return ts.termination(observation, reward)
        
        return ts.transition(observation, reward=reward, discount = 1.0)

        

        
    def collisions(self):
        for obj in self.world:
            if self.car.intersection(obj):
                return True
        return False


    def car_observation(self, from_degrees, to_degrees):
        observations = np.ndarray(shape=(to_degrees- from_degrees + 1, ), dtype=np.float32)
        for angle_offset in range(from_degrees, to_degrees+1):
            ray_start = self.car.pos
            ray_direction_y = np.sin((self.car_direction + angle_offset) / 180 * np.pi)
            ray_direction_x = np.cos((self.car_direction + angle_offset) / 180 * np.pi)
            ray = Ray((ray_start[0], ray_start[1]), (ray_direction_x, ray_direction_y))
            hits = [300]
            for obstacle in self.world:
                intersections = ray.intersection(obstacle)

                for intersection_hit in intersections:
                    hit_v = (intersection_hit[0] - ray_start[0], intersection_hit[1] - ray_start[1])
                    distance = min(hypot(hit_v[0], hit_v[1]), 300)
                    hits.append(distance)
            observations[angle_offset - from_degrees] = min(hits)/300
            
        return observations

    def draw(self) -> Surface:
        surface = Surface((self.right_wall - self.left_wall, self.bottom_wall - self.top_wall + 100))
        offset = (-self.left_wall, -self.top_wall)
        surface.fill((255, 255, 255))
        for obstacle in self.world:
            if isinstance(obstacle, Line):
                draw_points = [(p[0]+offset[0], p[1]+offset[1]) for p in (obstacle.p1, obstacle.p2)]
                draw.line(surface, (0, 0, 0), draw_points[0], draw_points[1], 3)
            elif isinstance(obstacle, Circle):
                draw.circle(surface, (0, 0, 0), (obstacle.pos[0]+offset[0], obstacle.pos[1]+offset[1]), obstacle.radius, 3)
        draw.circle(surface, (255, 0, 0), (self.car.pos[0] + offset[0], self.car.pos[1] + offset[1]), self.car.radius)

        start = time.time()
        intersect = self.car_observation(-32, 32)
        print('Elapsed: ',time.time()-start)
        for angle, distance in enumerate(intersect):
            angle_adjusted = angle - 32
            distance *= 300
            ray_start = self.car.pos
            ray_direction_y = np.sin((self.car_direction + angle_adjusted) / 180 * np.pi)
            ray_direction_x = np.cos((self.car_direction + angle_adjusted) / 180 * np.pi)
            draw.line(surface, (0, 255, 255), (ray_start[0] + offset[0], ray_start[1] + offset[1]), (ray_start[0] + ray_direction_x * distance+offset[0], ray_start[1] + ray_direction_y * distance+offset[1]))
        depth_visualization = interpolate.interp1d(np.arange(-32, 33, dtype=int), intersect, kind='cubic')

        for i in range(surface.get_width()):
            # draw a vertical line from the interpolation
            interpretation_point = (i-surface.get_width()//2) * 64 / surface.get_width()
            colour = 255 - min(int(depth_visualization(interpretation_point) * 255), 255)
            try:
                draw.line(surface, (colour, colour, colour), (i, surface.get_height()-100), (i, surface.get_height()-20))
            except ValueError:
                print(colour)
        draw.rect(surface, (0, 0, 0), (0, surface.get_height()-20, surface.get_width(), 20))
        throttle_rect = Rect(surface.get_width()//2, surface.get_height()-20, surface.get_width() * (self.last_command[0]-0.5), 10)
        steering_rect = Rect(surface.get_width()//2, surface.get_height()-10, surface.get_width() * (self.last_command[1]-0.5), 10)
        throttle_rect.normalize()
        steering_rect.normalize()
        draw.rect(surface, (0, 255, 0), throttle_rect)
        draw.rect(surface, (0, 255, 0), steering_rect)
        return surface

    def render(self, mode=None):
        return (surfarray.pixels3d(self.draw())).transpose((1, 0, 2))

if __name__ == '__main__':
    environment = SimpleCarEnvironment()
    #utils.validate_py_environment(wrappers.TimeLimit(environment, 100), episodes=3)

    plt.imshow(environment.render())
    plt.show()