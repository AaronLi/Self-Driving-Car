import numpy as np
class Circle:
    def __init__(self, pos, radius):
        self.radius = radius
        self.pos = np.array(pos, dtype=np.float32)

    def hit_circle(self, C: 'Circle'):
        return np.linalg.norm(self.pos - C.pos) < self.radius + C.radius

    def hit_line(self, L: 'Line'):
        line_p1_adjusted = L.p1 - self.pos
        line_p2_adjusted = L.p2 - self.pos
        dx = line_p2_adjusted[0] - line_p1_adjusted[0]
        dy = line_p2_adjusted[1] - line_p1_adjusted[1]
        dr = np.hypot(dx, dy)
        D = line_p1_adjusted[0] * line_p2_adjusted[1] - line_p2_adjusted[0] * line_p1_adjusted[1]

        discriminant = (self.radius **2) * (dr **2) - (D**2)

        if discriminant < 0:
            return False
        else:
            return True

    def intersection(self, other):
        if isinstance(other, Circle):
            return self.hit_circle(other)
        elif isinstance(other, Line):
            return self.hit_line(other)
        else:
            raise Exception("Unknown object type")

class Line:
    def __init__(self, p1, p2):
        self.p1 = np.array(p1, dtype=np.float32)
        self.p2 = np.array(p2, dtype=np.float32)

class Ray:
    def __init__(self, start, direction):
        self.pos = np.array(start, dtype=np.float32)
        self.direction = np.array(direction, dtype=np.float32)

    def hit_circle(self, C: Circle):
        U = C.pos - self.pos
        ray_direction = self.direction / np.linalg.norm(self.direction)
        U1 = U.dot(ray_direction)*ray_direction

        if U1.dot(self.direction) < 0:
            return []

        U2 = U - U1

        d = np.linalg.norm(U2)


        if d < C.radius:
            m = np.sqrt(C.radius**2 - d**2).item()
            P1 = self.pos + U1 + m*ray_direction
            P2 = self.pos + U1 - m*ray_direction
            return [P1, P2]
        elif d == C.radius:
            m = np.sqrt(C.radius**2 - d**2).item()
            P1 = self.pos + U1 + m*ray_direction
            return [P1]
        else:
            return []

    def hit_line(self, L: Line):
        v1 = self.pos - L.p1
        v2 = L.p2 - L.p1

        v3 = np.array([-self.direction[1], self.direction[0]])

        t1 = np.cross(v2, v1) / np.dot(v2, v3)
        t2 = np.dot(v1, v3) / np.dot(v2, v3)
        if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
            return [self.pos + t1 * self.direction]
        return []

    def intersection(self, other):
        if isinstance(other, Circle):
            return self.hit_circle(other)
        elif isinstance(other, Line):
            return self.hit_line(other)