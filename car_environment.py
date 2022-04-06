from math import hypot
import time
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from shapely.geometry import Point, LineString, Polygon
from scipy import interpolate
import random
import tqdm
from pygame import Surface, image, draw, surfarray
from tf_agents.trajectories import time_step as ts

class SimpleCarEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        self.world = []
        self.car_pos = [0, 0]
        self.car_radius = 5
        self.car_velocity = [0, 0]
        self.car_direction = 0
        self._episode_ended = False
        self.steering_amount = 30
        self.max_speed = 10
        self.sim_fps = 90
        self.current_sim_steps = 0
        self.travelled = 0
        self.left_wall = -300
        self.right_wall = 300
        self.top_wall = -300
        self.bottom_wall = 300
        self.terminate_simulation = False
        self._action_spec = array_spec.BoundedArraySpec(shape=(2, ), dtype=float, minimum = 0, maximum=1, name='action') # speed, steering
        self._observation_spec = array_spec.BoundedArraySpec(shape=(65, ), dtype=float, minimum = 0, maximum=1, name='observation')
        self.create_environment()
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def create_environment(self):
        top_wall = LineString([(self.left_wall, self.top_wall), (self.right_wall, self.top_wall)])
        right_wall = LineString([(self.right_wall, self.top_wall), (self.right_wall, self.bottom_wall)])
        bottom_wall = LineString([(self.left_wall, self.bottom_wall), (self.right_wall, self.bottom_wall)])
        left_wall = LineString([(self.left_wall, self.top_wall), (self.left_wall, self.bottom_wall)])
        self.world = [top_wall, right_wall, bottom_wall, left_wall]

        for i in range(20):
            obstacle = 0#random.randint(0, 2)

            shape_x = random.randint(self.left_wall, -50) if random.randint(0, 1) else random.randint(50, self.right_wall)
            shape_y = random.randint(self.top_wall, -50) if random.randint(0, 1) else random.randint(50, self.bottom_wall)
            if obstacle == 0:
                self.world.append(Point(shape_x, shape_y).buffer(random.random() * 100))
            # else:
            #     self.world.append(RegularPolygon((shape_x, shape_y), random.random() * 100, random.randint(3, 6)))

    def _reset(self):
        self.car_velocity = [0, 0]
        self.car_pos = [0, 0]
        self.current_sim_steps = 0
        self.car_direction = 0
        self.travelled = 0
        self.terminate_simulation = False
        return ts.restart(np.array(self.car_observation(-32, 32)))
    
    def _step(self, action):
        if self.terminate_simulation:
            return self._reset()
        self.car_velocity[0] *= 0.9 * (1/self.sim_fps)
        self.car_velocity[1] *= 0.9 * (1/self.sim_fps)

        self.car_direction += (2*action[1]-1) * self.steering_amount * (1/self.sim_fps)

        self.car_velocity[0] += (2*action[0]-1) * self.max_speed * np.cos(self.car_direction / 180 * np.pi) * (1/self.sim_fps)
        self.car_velocity[1] += (2*action[0]-1) * self.max_speed * np.sin(self.car_direction / 180 * np.pi) * (1/self.sim_fps)

        self.car_pos[0] += self.car_velocity[0]
        self.car_pos[1] += self.car_velocity[1]
        
        self.current_sim_steps += 1

        observation = self.car_observation(-32, 32)

        self.travelled += self.car_velocity[0]

        reward = self.current_sim_steps * min((self.travelled / (self.current_sim_steps/self.sim_fps)) / self.max_speed, 1) # distance over time 
        if self.collisions():
            self.terminate_simulation = True
            return ts.termination(np.array(observation), reward)
        
        return ts.transition(np.array(observation), reward = 1, discount = 1.0)

        

        
    def collisions(self):
        car = Point(self.car_pos).buffer(self.car_radius)
        for obj in self.world:
            if car.intersection(obj):
                return True
        return False


    def car_observation(self, from_degrees, to_degrees):
        observations = []
        for angle_offset in range(from_degrees, to_degrees+1):
            ray_start = self.car_pos
            ray_direction_y = np.sin((self.car_direction + angle_offset) / 180 * np.pi)
            ray_direction_x = np.cos((self.car_direction + angle_offset) / 180 * np.pi)
            ray = LineString([(ray_start[0], ray_start[1]), (ray_start[0] + ray_direction_x*300, ray_start[1] + ray_direction_y*300)])
            hits = [300]
            for obstacle in self.world:
                intersections = ray.intersection(obstacle)

                for intersection_hit in intersections.coords:
                    hit_v = (intersection_hit[0] - ray_start[0], intersection_hit[1] - ray_start[1])
                    distance = hypot(hit_v[0], hit_v[1])
                    hits.append(distance)
            observations.append(min(hits)/300)
            
        return observations

    def draw(self) -> Surface:
        surface = Surface((self.right_wall - self.left_wall, self.bottom_wall - self.top_wall + 100))
        offset = (-self.left_wall, -self.top_wall)
        surface.fill((255, 255, 255))
        for obstacle in self.world:
            if isinstance(obstacle, LineString):
                draw_points = [(p[0]+offset[0], p[1]+offset[1]) for p in obstacle.coords]
                draw.line(surface, (0, 0, 0), draw_points[0], draw_points[1], 3)
            elif isinstance(obstacle, Polygon):
                draw_points = [(p[0]+offset[0], p[1]+offset[1]) for p in obstacle.exterior.coords]
                draw.polygon(surface, (0, 0, 0), draw_points)
        draw.circle(surface, (255, 0, 0), (self.car_pos[0] + offset[0], self.car_pos[1] + offset[1]), self.car_radius)

        start = time.time()
        intersect = self.car_observation(-32, 32)
        print('Elapsed: ',time.time()-start)
        for angle, distance in enumerate(intersect):
            angle_adjusted = angle - 32
            distance *= 300
            ray_start = self.car_pos
            ray_direction_y = np.sin((self.car_direction + angle_adjusted) / 180 * np.pi)
            ray_direction_x = np.cos((self.car_direction + angle_adjusted) / 180 * np.pi)
            draw.line(surface, (0, 255, 255), (ray_start[0] + offset[0], ray_start[1] + offset[1]), (ray_start[0] + ray_direction_x * distance+offset[0], ray_start[1] + ray_direction_y * distance+offset[1]))
        depth_visualization = interpolate.interp1d(np.arange(-32, 33, dtype=int), intersect, kind='cubic')

        for i in range(surface.get_width()):
            # draw a vertical line from the interpolation
            interpretation_point = (i-surface.get_width()//2) * 64 / surface.get_width()
            colour = 255 - min(int(depth_visualization(interpretation_point) * 255), 255)
            try:
                draw.line(surface, (colour, colour, colour), (i, surface.get_height()-100), (i, surface.get_height()))
            except ValueError:
                print(colour)
        return surface

    def render(self, mode):
        return (surfarray.pixels3d(self.draw())).transpose((1, 0, 2))

if __name__ == '__main__':
    environment = SimpleCarEnvironment()
    utils.validate_py_environment(wrappers.TimeLimit(environment, 100), episodes=3)