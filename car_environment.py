from math import hypot
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from sympy import Circle, Line, RegularPolygon, Ray, pi
from sympy.geometry.util import intersection
from scipy import interpolate
import random
import tqdm
from pygame import Surface, image, draw

class SimpleCarEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        self.world = []
        self.car: Circle = Circle((0, 0), 5)
        self.car_velocity = [0, 0]
        self.car_direction = 0
        self._episode_ended = False
        self._action_spec = array_spec.BoundedArraySpec(shape=(2, ), dtype=float, minimum = 0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(65, ), dtype=float, minimum = 0, maximum=1, name='observation')
        self.create_environment()
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def create_environment(self):
        top_wall = Line((-500, -500), (500, -500))
        right_wall = Line((500, -500), (500, 500))
        bottom_wall = Line((-500, 500), (500, 500))
        left_wall = Line((-500, -500), (-500, 500))
        self.world = [top_wall, right_wall, bottom_wall, left_wall]

        for i in range(20):
            obstacle = 0#random.randint(0, 2)

            shape_x = random.randint(-500, -50) if random.randint(0, 1) else random.randint(50, 500)
            shape_y = random.randint(-500, -50) if random.randint(0, 1) else random.randint(50, 500)
            if obstacle == 0:
                self.world.append(Circle((shape_x, shape_y), random.random() * 100))
            else:
                self.world.append(RegularPolygon((shape_x, shape_y), random.random() * 100, random.randint(3, 6)))

    def _reset(self):
        self.car_velocity = [0, 0]
        self.car = Circle((0, 0), 5)
        self.car_direction = 0
    
    def _step(self, action):
        pass

    def car_observation(self):
        observations = []
        for i in tqdm.tqdm(range(65)):
            angle_offset = i-32
            ray_start = self.car.center
            ray_direction = Ray((0, 0), (1, 0)).rotate((self.car_direction + angle_offset) / 180 * pi)
            ray = Ray(ray_start, (float(ray_direction.direction.x), float(ray_direction.direction.y)))
            hits = [300]
            for obstacle in tqdm.tqdm(self.world, leave=False):
                intersections = intersection(ray, obstacle)
                for intersection_hit in intersections:
                    hit_v = (intersection_hit - ray_start)
                    distance = hypot(hit_v.x, hit_v.y)
                    hits.append(distance)
            observations.append(min(hits)/300)
            
        return observations

    def draw(self) -> Surface:
        surface = Surface((1000, 1100))
        offset = (500, 500)
        surface.fill((255, 255, 255))
        for obstacle in self.world:
            if isinstance(obstacle, Circle):
                draw.circle(surface, (0, 0, 0), obstacle.center + offset, int(obstacle.radius))
            elif isinstance(obstacle, Line):
                draw.line(surface, (0, 0, 0), obstacle.p1 + offset, obstacle.p2 + offset)
            elif isinstance(obstacle, RegularPolygon):
                draw.polygon(surface, (0, 0, 0), [vertex+offset for vertex in obstacle.vertices])
        draw.circle(surface, (255, 0, 0), self.car.center + offset, self.car.radius)

        intersect = self.car_observation()
        print(intersect)
        for angle, distance in enumerate(intersect):
            angle_adjusted = angle - 32
            distance *= 300
            ray_start = self.car.center
            ray_direction = Ray((0, 0), (1, 0)).rotate( (self.car_direction + angle_adjusted)/ 180  * pi)
            draw.line(surface, (0, 255, 255), ray_start + offset, ray_start + ray_direction.direction * distance + offset)
        depth_visualization = interpolate.interp1d(np.arange(-32, 33, dtype=int), intersect)

        for i in range(1001):
            # draw a vertical line from the interpolation
            colour = 255 - int(depth_visualization((i-500) * 32 / 500) * 255)
            draw.line(surface, (colour, colour, colour), (i, 1000), (i, 1100))
        return surface

if __name__ == '__main__':
    environment = SimpleCarEnvironment()
    environment.create_environment()
    environment_image = environment.draw()
    image.save(environment_image, "car_environment.png")