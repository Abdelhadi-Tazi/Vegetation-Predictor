import math
from functools import total_ordering

@total_ordering
class Point:
    def __init__(self, x: float, y: float, type: int = 0, neighborhood: int = -1):
        self.x = x
        self.y = y
        self.type = type
        self.neighborhood = neighborhood
    
    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y
    
    def __ne__(self, other) -> bool:
        return not (self.x == other.x and self.y == other.y)
    
    # def __ge__(self, other) -> bool:
    #     return self.x > other.x or (self.x == other.x and self.y >= other.y)

    # def __le__(self, other) -> bool:
    #     return self.x < other.x or (self.x == other.x and self.y <= other.y)

    # def __gt__(self, other) -> bool:
    #     return self.x > other.x or (self.x == other.x and self.y > other.y)

    def __lt__(self, other) -> bool:
        return self.x < other.x or (self.x == other.x and self.y < other.y)
    
    def dist(self, p):
        return math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)
    
    def translation(self, x: float, y: float):
        return Point(self.x + x, self.y + y, self.type, self.neighborhood)
