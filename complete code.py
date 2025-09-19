import heapq
from collections import deque
import random

class GridEnvironment:
    def __init__(self, grid, static_obstacles, dynamic_obstacles_schedule=None):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.static_obstacles = set(static_obstacles)
        self.dynamic_obstacles_schedule = dynamic_obstacles_schedule or {}
    
    def is_cell_blocked(self, r, c, time=None):
        if (r, c) in self.static_obstacles:
            return True
        if time and time in self.dynamic_obstacles_schedule:
            if (r, c) in self.dynamic_obstacles_schedule[time]:
                return True
        return False

    def neighbors(self, r, c):
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                yield (nr, nc)

def bfs(env, start, goal):
    queue = deque([start])
    came_from = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbor in env.neighbors(*current):
            if neighbor not in came_from and not env.is_cell_blocked(*neighbor):
                queue.append(neighbor)
                came_from[neighbor] = current
    return reconstruct_path(came_from, start, goal)

def uniform_cost_search(env, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in env.neighbors(*current):
            if env.is_cell_blocked(*neighbor):
                continue
            new_cost = cost_so_far[current] + env.grid[neighbor[0]][neighbor[1]]
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
                came_from[neighbor] = current
    return reconstruct_path(came_from, start, goal)

def manhattan_heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star_search(env, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in env.neighbors(*current):
            if env.is_cell_blocked(*neighbor):
                continue
            new_cost = cost_so_far[current] + env.grid[neighbor[0]][neighbor[1]]
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + manhattan_heuristic(neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
    return reconstruct_path(came_from, start, goal)

def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return []  # No path found
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def main():
    # Sample grid (terrain costs)
    grid = [
        [1, 1, 2, 1, 3],
        [1, 3, 2, 1, 1],
        [1, 1, 1, 4, 1],
        [4, 1, 1, 1, 1],
        [1, 1, 2, 1, 1]
    ]
    
    # Static obstacles (cells blocked)
    static_obstacles = {(1,1), (3,0)}
    
    # Dynamic obstacles schedule as time : set(cells)
    dynamic_obstacles_schedule = {
        3: {(2,3)},
        4: {(3,3), (4,2)}
    }
    
    env = GridEnvironment(grid, static_obstacles, dynamic_obstacles_schedule)
    
    start = (0, 0)
    goal = (4, 4)
    
    print("BFS path:")
    print(bfs(env, start, goal))
    
    print("Uniform Cost Search path:")
    print(uniform_cost_search(env, start, goal))
    
    print("A* Search path:")
    print(a_star_search(env, start, goal))

if __name__ == "__main__":
    main()
