import numpy as np
from math import sqrt
from queue import PriorityQueue, deque
import heapq
import math
from ultralytics import YOLO
import cv2
import os
import rclpy
from rclpy.node import Node
import numpy as np
from queue import PriorityQueue
from rclpy.qos import QoSProfile
import math
from nav_msgs.msg import OccupancyGrid, Odometry 
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image
import time
import cv2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import os
from cv_bridge import CvBridge
import random

class CoveragePlanner:
    def __init__(self, grid):
        self.grid = grid  # 0 - свободно, 1 - препятствие
        self.rows, self.cols = grid.shape
        self.visited = np.zeros_like(grid)
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
        self.current_dir = 0  # начальное направление - вправо
        
    def snake_move(self, start_pos):
        path = []
        x, y = start_pos
        stuck_count = 0
        
        while stuck_count < 2:  # пробуем обойти препятствие
            dx, dy = self.directions[self.current_dir]
            new_x, new_y = x + dx, y + dy
            
            if self.is_valid_move(new_x, new_y):
                path.append((new_x, new_y))
                self.visited[new_x, new_y] = 1
                
                sq = 20
                for i in range(-sq, sq+1):
                    for j in range(-sq, sq+1):
                        dx = new_x + i
                        dy = new_y + j
                        if self.is_valid_move(dx, dy):
                            self.visited[dx, dy] = 1
                x, y = new_x, new_y
                stuck_count = 0
            else:
                self.current_dir = (self.current_dir + 1) % 4
                dx, dy = self.directions[self.current_dir]
                new_x, new_y = x + dx, y + dy
                
                if self.is_valid_move(new_x, new_y):
                    path.append((new_x, new_y))
                    self.visited[new_x, new_y] = 1
                    x, y = new_x, new_y
                    stuck_count = 0
                else:
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
        """Основной метод покрытия"""
        full_path = [start_pos]
        current_pos = start_pos
        self.visited[start_pos] = 1
        
        max_iterations = 1000  # защита от бесконечного цикла
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Двигаемся змейкой
            snake_path, current_pos = self.snake_move(current_pos)
            if snake_path:
                full_path.extend(snake_path)
            
            # Ищем следующую цель
            next_target = self.find_nearest_unvisited(current_pos)
            if next_target is None:
                break
                
            # Строим путь до цели
            theta_path = self.theta_star(current_pos, next_target)
            if theta_path is None:
                # Если путь не найден, отмечаем цель как недостижимую
                self.visited[next_target] = 1
                continue
                
            full_path.extend(theta_path)
            for pos in theta_path:
                self.visited[pos] = 1
            current_pos = next_target
        
        return full_path

expansion_size = 20

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
     
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
     
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
     
    return yaw_z

def costmap(data, width, height, resolution):
    """Улучшенная обработка карты"""
    grid = np.array(data, dtype=np.int8).reshape(height, width)
    
    # Создаем маску препятствий
    obstacles_mask = np.where(grid == 100, 255, 0).astype(np.uint8)
    
    # Расширяем препятствия
    kernel_size = expansion_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_obstacles = cv2.dilate(obstacles_mask, kernel)
    
    # Объединяем с оригинальной картой
    result = np.where(dilated_obstacles == 255, 100, grid)
    result[grid == -1] = -1  # сохраняем неизвестные области
    
    return result.flatten().tolist()



class Navigation(Node):
    def __init__(self):
        super().__init__('navigation_node')

        self.bridge = CvBridge()

        self.img_sub = self.create_subscription(
            Image, '/camera/image_raw',
            self.camera_callback,
            10
        )

        self.subscription_map = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        self.subscription_odom = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_marker_pub = self.create_publisher(Marker, '/path_marker', 10)
        self.world_to_grid_marker_pub = self.create_publisher(Marker, '/world_to_grid_marker', 10)

        self.model = YOLO('/home/sa/turtlebot3_ws/src/maze-dfs/maze-dfs/runs/detect/train3/weights/best.pt')

        self.data = None
        self.path_to_images = 'collected_images'
        self.img_count = 0
        self.timer = self.create_timer(2.0, self.nav_timer)
        self.viz_timer = self.create_timer(1.0, self.visualization_callback)  # 1 Hz

        self.map_initialized = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.path = []
        self.count = 0
        self.goal = None
        
        self.kp_distance = 0.3    # Уменьшено для плавности
        self.ki_distance = 0.001  # Уменьшено
        self.kd_distance = 0.1    # Уменьшено
        self.kp_angle = 0.5       # Уменьшено для плавных поворотов
        self.ki_angle = 0.005     # Уменьшено
        self.kd_angle = 0.1       # Уменьшено
        
        # Параметры движения
        self.max_linear_speed = 0.15
        self.max_angular_speed = 0.4
        self.goal_tolerance = 0.15     # Точность достижения точки
        self.angle_tolerance = 0.1    # Точность ориентации
        
        self.previous_distance = 0
        self.total_distance = 0
        self.previous_angle = 0
        self.total_angle = 0
        self.last_rotation = 0
        self.waypoint_reached = False
        self.last_time = self.get_clock().now()
        self.current_waypoint_index = 0
        self.navigating_to_waypoint = False
        self.path_to_viz = []

    def camera_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.data = cv_image

    def map_callback(self, msg):
        if not self.map_initialized:
            self.map_resolution = msg.info.resolution
            self.map_origin = [
                msg.info.origin.position.x,
                msg.info.origin.position.y
            ]
            self.width = msg.info.width
            self.height = msg.info.height
            self.grid = costmap(msg.data, self.width, self.height, self.map_resolution)
            self.grid = np.array(self.grid).reshape(self.height, self.width)
            self.grid = np.where((self.grid == 100) | (self.grid == -1), 1, 0).astype(np.int8)
            self.coverage_planner = CoveragePlanner(self.grid)

            time.sleep(10)
            self.map_initialized = True

            self.get_logger().info("Map initialized")


    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
    

    def improved_pid_navigate(self, goal_x, goal_y):
        """Улучшенный ПИД-контроллер с адаптивной скоростью"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        dt = max(dt, 0.01)  # Минимальный шаг времени
        
        move_cmd = Twist()
        
        # Вычисление ошибок
        dx = goal_x - self.x
        dy = goal_y - self.y
        distance_error = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        
        # Вычисление ошибки угла (нормализованной)
        angle_error = target_angle - self.yaw
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi
        
        # Адаптивная скорость на основе расстояния и угла
        if distance_error < self.goal_tolerance:
            # Достигли цели
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 0.0
            self.waypoint_reached = True
            return move_cmd, distance_error, True
        
        # ПИД для линейной скорости
        linear_vel = distance_error * self.kp_distance
        linear_vel = min(linear_vel, self.max_linear_speed)
        
        # Адаптивное уменьшение скорости при больших угловых ошибках
        speed_reduction = 1.0 - min(abs(angle_error) / (math.pi/2), 0.8)
        linear_vel *= speed_reduction
        
        # ПИД для угловой скорости
        angular_vel = angle_error * self.kp_angle
        
        # Ограничение угловой скорости
        if angular_vel > 0:
            angular_vel = min(angular_vel, self.max_angular_speed)
        else:
            angular_vel = max(angular_vel, -self.max_angular_speed)
        
        # Если угол слишком большой - сначала поворачиваем на месте
        if abs(angle_error) > math.pi/3:  # ~60 градусов
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = angular_vel
        else:
            move_cmd.linear.x = linear_vel
            move_cmd.angular.z = angular_vel
        
        self.last_time = current_time
        return move_cmd, distance_error, False

    def pure_pursuit_improved(self, lookahead_distance=0.3):
        """Улучшенный Pure Pursuit с поиском целевой точки"""
        if len(self.path) < 2:
            return 0.0, 0.0, None
        
        # Находим ближайшую точку на пути
        closest_point = None
        closest_dist = float('inf')
        closest_index = 0
        
        for i, (x, y) in enumerate(self.path):
            dist = math.hypot(self.x - x, self.y - y)
            if dist < closest_dist:
                closest_dist = dist
                closest_point = (x, y)
                closest_index = i
        
        # Ищем целевую точку на lookahead расстоянии
        target_point = None
        for i in range(closest_index, len(self.path)):
            x, y = self.path[i]
            dist = math.hypot(self.x - x, self.y - y)
            if dist >= lookahead_distance:
                target_point = (x, y)
                break
        
        # Если не нашли, берем последнюю точку
        if target_point is None:
            target_point = self.path[-1]
        
        # Вычисление управления
        target_x, target_y = target_point
        
        # Вектор к целевой точке
        dx = target_x - self.x
        dy = target_y - self.y
        
        # Угол к целевой точке
        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - self.yaw
        
        # Нормализация угла
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi
        
        # Вычисление кривизны
        distance_to_target = math.hypot(dx, dy)
        if distance_to_target < 0.001:
            return 0.0, 0.0, target_point
        
        # Закон управления Pure Pursuit
        curvature = 2.0 * dy / (distance_to_target ** 2)
        
        # Линейная скорость (адаптивная)
        linear_vel = min(self.max_linear_speed, distance_to_target * 0.5)
        
        # Угловая скорость
        angular_vel = curvature * linear_vel
        
        # Ограничения
        angular_vel = max(min(angular_vel, self.max_angular_speed), -self.max_angular_speed)
        
        return linear_vel, angular_vel, target_point

    def set_goal(self, grid):
        goal = None
        x = random.randint(0, grid.shape[1]-1)
        y = random.randint(0, grid.shape[0]-1)

        if grid[y][x]==0:
            goal = (x,y)
            return goal
        else:
            return self.set_goal(grid)

    def visualization_callback(self):
        if self.map_initialized and self.path_to_viz != []:
            self.visualize_path()

    def nav_timer(self):
        """Улучшенный основной цикл навигации"""
        if not self.map_initialized or self.coverage_planner is None:
            return

        # Проверка объектов
        if self.data is not None:
            results = self.model(self.data, verbose=False)
            if len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    if results[0].boxes.conf[i].cpu().numpy() > 0.95:
                        self.get_logger().info('Object detected - stopping!')
                        self.stop_robot()
                        self.navigating_to_waypoint = False
                        self.current_waypoint_index = 0
                        return

        if not self.navigating_to_waypoint:
            # Генерация пути (оставляем как было)
            current_grid = self.world_to_grid(self.x, self.y)
            if current_grid is None:
                return

            start_pos = (current_grid[1], current_grid[0])
            
            try:
                path = self.coverage_planner.full_coverage_path(start_pos)
                if not path:
                    return
                    
                self.path = []
                for grid_pos in path:
                    world_pos = self.grid_to_world(grid_pos[1], grid_pos[0])
                    if world_pos:
                        self.path.append(world_pos)

                self.path_to_viz = [self.grid_to_world_rviz(i[1], i[0]) for i in path]
                
                if not self.path:
                    return
                    
                self.reset_pid_state()
                self.current_waypoint_index = 0
                self.navigating_to_waypoint = True
                self.waypoint_reached = False
                
                self.get_logger().info(f"Generated path with {len(self.path)} points")
                
            except Exception as e:
                self.get_logger().error(f"Path planning error: {str(e)}")
                return

        # Навигация по точкам пути
        if self.navigating_to_waypoint and self.path:
            if self.current_waypoint_index >= len(self.path):
                self.stop_robot()
                self.get_logger().info("Path completed!")
                self.navigating_to_waypoint = False
                return

            current_waypoint = self.path[self.current_waypoint_index]
            
            # ВЫБОР КОНТРОЛЛЕРА - раскомментируйте нужный:
            
            # Вариант 1: Улучшенный ПИД
            twist, distance, reached = self.improved_pid_navigate(
                current_waypoint[0], current_waypoint[1]
            )
            
            # Вариант 2: Pure Pursuit (лучше для плавного движения)
            # linear_vel, angular_vel, _ = self.pure_pursuit_improved()
            # twist = Twist()
            # twist.linear.x = linear_vel
            # twist.angular.z = angular_vel
            # distance = math.hypot(current_waypoint[0] - self.x, current_waypoint[1] - self.y)
            # reached = (distance < self.goal_tolerance)
            
            self.cmd_vel_pub.publish(twist)

            # Переход к следующей точке
            if reached or distance < self.goal_tolerance:
                self.current_waypoint_index += 1
                self.reset_pid_state()
                self.waypoint_reached = False
                
                if self.current_waypoint_index < len(self.path):
                    next_wp = self.path[self.current_waypoint_index]
                    self.get_logger().info(
                        f"Reached waypoint {self.current_waypoint_index-1}/"
                        f"{len(self.path)}, moving to next: {next_wp}"
                    )

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        
    def world_to_grid(self, x_world, y_world, map_offset = (45, 15)):
        x_grid = int((x_world - self.map_origin[0]) / self.map_resolution) + map_offset[0]
        y_grid = int((y_world - self.map_origin[1]) / self.map_resolution) + map_offset[1]
        return (x_grid, y_grid)

    def grid_to_world(self, x_grid, y_grid, map_offset = (45,15)):
        x_world = (x_grid - map_offset[0])* self.map_resolution + self.map_origin[0]
        y_world = (y_grid - map_offset[1]) * self.map_resolution + self.map_origin[1]
        return (x_world, y_world)
    
    def grid_to_world_rviz(self, x_grid, y_grid, map_offset = (0,0)):
        x_world = (x_grid - map_offset[0])* self.map_resolution + self.map_origin[0]
        y_world = (y_grid - map_offset[1]) * self.map_resolution + self.map_origin[1]
        return (x_world, y_world)
    
    
    def visualize_path(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.g = 1.0
        marker.color.r = 0.0
        marker.color.b = 0.0

        for (x, y), i in zip(self.path_to_viz, range(len(self.path_to_viz))):
            marker.color.g = 1.0 - (1.0 - i/len(self.path_to_viz))
            marker.color.a = 1.0 - (1.0 - i/len(self.path_to_viz))

            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)

        self.path_marker_pub.publish(marker)


    def visualize_world_to_map(self, point):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "world_to_grid"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.world_to_grid_marker_pub.publish(marker)

    def reset_pid_state(self):
        """Сброс состояния ПИД регулятора"""
        self.previous_distance = 0
        self.total_distance = 0
        self.previous_angle = 0
        self.total_angle = 0
        self.last_rotation = 0

def main(args = None):
    rclpy.init(args=args)
    collector = Navigation()
    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        print("you stopped it")
    finally:
        collector.destroy_node()
        rclpy.shutdown()

if __name__== '__main__':
    main()
