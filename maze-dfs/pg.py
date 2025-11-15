import pygame
import numpy as np
from math import sqrt
from queue import PriorityQueue, deque
import heapq
import math

# Константы
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 40, 40
SPOT_WIDTH = WIDTH // COLS
SPOT_HEIGHT = HEIGHT // ROWS

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
GREY = (128, 128, 128)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Инициализация Pygame
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Coverage Path Planner")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 16)

class Spot:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = row * SPOT_WIDTH
        self.y = col * SPOT_HEIGHT
        self.color = WHITE
        self.state = "free"  # free, obstacle, visited, target, path

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, SPOT_WIDTH, SPOT_HEIGHT))
        pygame.draw.rect(win, GREY, (self.x, self.y, SPOT_WIDTH, SPOT_HEIGHT), 1)

    def update_state(self):
        # Обновляет цвет в зависимости от состояния
        state_color = {
            "free": WHITE,
            "obstacle": BLACK,
            "visited": GREEN,
            "target": BLUE,
            "path": YELLOW,
            "current": RED
        }
        self.color = state_color.get(self.state, WHITE)

class CoveragePlanner:
    def __init__(self, grid):
        self.grid = grid  # 0 - свободно, 1 - препятствие
        self.rows, self.cols = grid.shape
        self.visited = np.zeros_like(grid)
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # право, низ, лево, верх
        self.current_dir = 0  # начальное направление - вправо
        
    def snake_move(self, start_pos):
        path = []
        x, y = start_pos
        stuck_count = 0
        
        while stuck_count < 2:  # пробуем обойти препятствие
            # Пытаемся двигаться в текущем направлении
            dx, dy = self.directions[self.current_dir]
            new_x, new_y = x + dx, y + dy
            
            if self.is_valid_move(new_x, new_y):
                path.append((new_x, new_y))
                self.visited[new_x, new_y] = 1
                
                sq = 2
                for i in range(-sq, sq+1):
                    for j in range(-sq, sq+1):
                        dx = new_x + i
                        dy = new_y + j
                        if self.is_valid_move(dx, dy):
                            self.visited[dx, dy] = 1
                x, y = new_x, new_y
                stuck_count = 0
            else:
                # Пробуем повернуть направо
                self.current_dir = (self.current_dir + 1) % 4
                dx, dy = self.directions[self.current_dir]
                new_x, new_y = x + dx, y + dy
                
                if self.is_valid_move(new_x, new_y):
                    path.append((new_x, new_y))
                    self.visited[new_x, new_y] = 1
                    x, y = new_x, new_y
                    stuck_count = 0
                else:
                    # Пробуем повернуть налево
                    self.current_dir = (self.current_dir - 1) % 4
                    dx, dy = self.directions[self.current_dir]
                    new_x, new_y = x + dx, y + dy
                    
                    if self.is_valid_move(new_x, new_y):
                        path.append((new_x, new_y))
                        self.visited[new_x, new_y] = 1
                        x, y = new_x, new_y
                        stuck_count = 0
                    else:
                        stuck_count += 1
                        # Поворачиваем в исходное направление
                        self.current_dir = (self.current_dir + 1) % 4
        
        return path, (x, y)

    def is_valid_move(self, x, y):
        return (0 <= x < self.rows and 0 <= y < self.cols and 
                self.grid[x, y] == 0 and self.visited[x, y] == 0)
                
    def find_nearest_unvisited(self, current_pos):
        queue = deque([current_pos])
        visited_bfs = set([current_pos])
        
        while queue:
            x, y = queue.popleft()
            
            for dx, dy in self.directions:
                new_x, new_y = x + dx, y + dy
                
                if (0 <= new_x < self.rows and 0 <= new_y < self.cols and 
                    (new_x, new_y) not in visited_bfs):
                    
                    if self.grid[new_x, new_y] == 0 and self.visited[new_x, new_y] == 0:
                        return (new_x, new_y)
                    
                    if self.grid[new_x, new_y] == 0:
                        queue.append((new_x, new_y))
                        visited_bfs.add((new_x, new_y))
        
        return None  # все точки посещены
        
    def theta_star(self, start, goal):
        open_set = []
        closed_set = set()
        
        # Словари для хранения g-стоимости и родителей
        g_score = {start: 0}
        parent = {start: start}
        
        heapq.heappush(open_set, (self.heuristic(start, goal), start))
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                return self.reconstruct_path(parent, start, goal)
            
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Theta*: проверка прямой видимости к родителю текущей точки
                if parent[current] != current and self.has_line_of_sight(parent[current], neighbor):
                    # Прямая видимость есть - обновляем через родителя текущей точки
                    tentative_g = g_score[parent[current]] + self.distance(parent[current], neighbor)
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = parent[current]
                        f_score = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
                else:
                    # Стандартный A* шаг
                    tentative_g = g_score[current] + self.distance(current, neighbor)
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = current
                        f_score = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
        
        return None  # путь не найден

    def has_line_of_sight(self, start, end):
        """Проверка прямой видимости между двумя точками (алгоритм Брезенхема)"""
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        
        for _ in range(n):
            if self.grid[x, y] == 1:  # препятствие
                return False
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        
        return True

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        
        for dx, dy in self.directions:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.rows and 0 <= new_y < self.cols and 
                self.grid[new_x, new_y] == 0):
                neighbors.append((new_x, new_y))
        
        return neighbors

    def distance(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # манхэттенское расстояние

    def reconstruct_path(self, parent, start, goal):
        path = []
        current = goal
        
        while current != start:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return path
    

    def full_coverage_path(self, start_pos):
        full_path = [start_pos]
        current_pos = start_pos
        self.visited[start_pos] = 1
        
        while True:
            # Двигаемся змейкой пока возможно
            snake_path, current_pos = self.snake_move(current_pos)
            full_path.extend(snake_path)
            
            # Ищем ближайшую непосещенную точку
            next_target = self.find_nearest_unvisited(current_pos)
            
            if next_target is None:
                break  # все точки посещены
            
            # Строим путь до целевой точки с помощью Theta*
            theta_path = self.theta_star(current_pos, next_target)
            
            if theta_path is None:
                break  # не можем достичь целевой точки
            
            # Добавляем путь к полному маршруту
            full_path.extend(theta_path)
            
            # Обновляем текущую позицию и отмечаем точки как посещенные
            for pos in theta_path:
                self.visited[pos] = 1
            current_pos = next_target
        
        return full_path

def make_grid():
    grid = []
    for i in range(ROWS):
        grid.append([])
        for j in range(COLS):
            spot = Spot(i, j)
            grid[i].append(spot)
    return grid

def draw_grid(win, grid):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    
    # Отображение статистики
    total_cells = ROWS * COLS
    obstacle_count = sum(1 for row in grid for spot in row if spot.state == "obstacle")
    visited_count = sum(1 for row in grid for spot in row if spot.state == "visited")
    coverage = visited_count / (total_cells - obstacle_count) * 100 if total_cells > obstacle_count else 0
    
    stats_text = f"Coverage: {coverage:.1f}%  Visited: {visited_count}  Obstacles: {obstacle_count}"
    text_surface = font.render(stats_text, True, BLACK)
    win.blit(text_surface, (10, HEIGHT - 30))
    
    pygame.display.update()

def get_clicked_pos(pos):
    x, y = pos
    row = x // SPOT_WIDTH
    col = y // SPOT_HEIGHT
    return row, col

def convert_grid_to_numpy(grid):
    """Конвертирует сетку Pygame в numpy массив для алгоритма"""
    numpy_grid = np.zeros((ROWS, COLS))
    for i in range(ROWS):
        for j in range(COLS):
            if grid[i][j].state == "obstacle":
                numpy_grid[i][j] = 1
    return numpy_grid

def update_grid_from_path(grid, path, current_index):
    """Обновляет сетку на основе пройденного пути"""
    # Сбрасываем все посещенные клетки (кроме препятствий)
    for i in range(ROWS):
        for j in range(COLS):
            if grid[i][j].state not in ["obstacle", "current"]:
                grid[i][j].state = "free"
                grid[i][j].update_state()
    
    # Отмечаем пройденный путь
    for idx, (x, y) in enumerate(path[:current_index+1]):
        if 0 <= x < ROWS and 0 <= y < COLS:
            if idx == current_index:
                grid[x][y].state = "current"  # текущая позиция
            else:
                grid[x][y].state = "visited"  # посещенные клетки
            grid[x][y].update_state()

def coverage_algorithm(grid, start):
    """Запуск алгоритма покрытия с визуализацией"""
    # Конвертируем сетку в numpy
    numpy_grid = convert_grid_to_numpy(grid)
    
    # Создаем планировщик
    planner = CoveragePlanner(numpy_grid)
    
    # Запускаем алгоритм
    start_pos = (start.row, start.col)
    full_path = planner.full_coverage_path(start_pos)
    
    # Визуализируем путь
    running = True
    current_index = 0
    paused = False
    
    while running and current_index < len(full_path):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    current_index = min(current_index + 1, len(full_path) - 1)
                elif event.key == pygame.K_LEFT and paused:
                    current_index = max(current_index - 1, 0)
        
        if not paused:
            current_index += 1
            if current_index >= len(full_path):
                current_index = len(full_path) - 1
        
        # Обновляем отображение
        update_grid_from_path(grid, full_path, current_index)
        draw_grid(WIN, grid)
        
        # Отображаем статус паузы
        if paused:
            pause_text = font.render("PAUSED - Use arrows to navigate, SPACE to resume", True, RED)
            WIN.blit(pause_text, (WIDTH // 2 - 200, 20))
            pygame.display.update()
        
        clock.tick(10)  # Скорость анимации
    
    # Отображаем завершение
    completion_text = font.render("Coverage Complete! Press ESC to exit", True, BLUE)
    WIN.blit(completion_text, (WIDTH // 2 - 150, 20))
    pygame.display.update()
    
    # Ждем завершения
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

def main():
    grid = make_grid()
    start = None
    running = True
    algorithm_running = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not algorithm_running:
                if pygame.mouse.get_pressed()[0]:  # ЛКМ - ставим препятствия
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos)
                    if 0 <= row < ROWS and 0 <= col < COLS:
                        spot = grid[row][col]
                        if spot != start:
                            spot.state = "obstacle"
                            spot.update_state()

                elif pygame.mouse.get_pressed()[2]:  # ПКМ - ставим старт
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos)
                    if 0 <= row < ROWS and 0 <= col < COLS:
                        spot = grid[row][col]
                        if start:
                            start.state = "free"
                            start.update_state()
                        start = spot
                        start.state = "current"
                        start.update_state()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and start:
                        algorithm_running = True
                        coverage_algorithm(grid, start)
                        algorithm_running = False
                    elif event.key == pygame.K_c:  # Очистка сетки
                        grid = make_grid()
                        start = None
                    elif event.key == pygame.K_r:  # Случайные препятствия
                        grid = make_grid()
                        for i in range(ROWS):
                            for j in range(COLS):
                                if np.random.random() < 0.2:  # 20% chance
                                    grid[i][j].state = "obstacle"
                                    grid[i][j].update_state()
                        # Устанавливаем старт в центр
                        start = grid[ROWS//2][COLS//2]
                        start.state = "current"
                        start.update_state()

        draw_grid(WIN, grid)
        
        # Отображаем инструкции
        if not algorithm_running:
            instructions = [
                "LEFT CLICK: Place obstacles",
                "RIGHT CLICK: Set start position", 
                "SPACE: Start coverage algorithm",
                "R: Random obstacles",
                "C: Clear grid"
            ]
            for i, text in enumerate(instructions):
                text_surface = font.render(text, True, BLACK)
                WIN.blit(text_surface, (10, 10 + i * 20))

        pygame.display.update()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()