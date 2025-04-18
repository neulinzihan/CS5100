import pygame
import sys
import json
import numpy as np
import os
import random
import time

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Drone colors for visualization
DRONE_COLORS = [
    (255, 0, 0),  # Red
    (0, 0, 255),  # Blue
    (0, 255, 0),  # Green
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (0, 128, 0),  # Dark Green
    (128, 128, 255)  # Light Blue
]

# Environment constants
UNEXPLORED = -1  # Cell hasn't been seen by any drone
OBSTACLE = 0  # Cell is an obstacle (no path)
SAFE = 1  # Cell is safe (has path)


class DronePathfinder:
    """
    Handles drone coverage visualization and emergency path finding.
    Uses matrix stitching to combine drone scan information.
    """

    def __init__(self, grid_size=20, cell_size=30):
        # Initialize PyGame
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen_w = self.grid_size * self.cell_size + 400  # Extra space for info panel
        self.screen_h = self.grid_size * self.cell_size + 100  # Extra padding
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Drone Coverage with Emergency Pathfinding")
        self.font = pygame.font.SysFont("Arial", 16)
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.clock = pygame.time.Clock()

        # Generate the actual environment (ground truth) with obstacles
        self.actual_env = self.generate_environment(grid_size)

        # Initialize the global grid with all cells unexplored
        self.global_grid = np.full((self.grid_size, self.grid_size), UNEXPLORED, dtype=np.int32)
        self.global_grid[0, 0] = SAFE  # Start location is always safe and known

        # Drone and animation state
        self.drone_positions = []
        self.drone_sizes = []
        self.drone_coverage_cells = []
        self.current_drone = -1
        self.animation_speed = 0.5
        self.animation_duration = 0.4
        self.animation_active = False
        self.expanding = False
        self.expansion_timer = 0.0
        self.current_expanded = set()
        self.coverage_history = [0]

        # Path finding state
        self.emergency_location = None
        self.path = None
        self.path_found = False

    def generate_environment(self, grid_size, obstacle_prob=0.2):
        """Generate a random environment with obstacles."""
        env = np.random.choice(
            [OBSTACLE, SAFE],
            size=(grid_size, grid_size),
            p=[obstacle_prob, 1 - obstacle_prob]
        ).astype(np.int32)

        # Ensure start position (0,0) is always SAFE
        env[0, 0] = SAFE

        return env


    def drone_scan(self, position, size):
        """Simulate a drone scan at the given position with the given size."""
        x, y = position
        half_size = (size - 1) // 2
        top_left = (max(0, x - half_size), max(0, y - half_size))

        # Extract the local information from the actual environment
        local_info = np.full((size, size), UNEXPLORED)
        for dx in range(size):
            for dy in range(size):
                nx, ny = top_left[0] + dx, top_left[1] + dy
                if (0 <= nx < self.grid_size and
                        0 <= ny < self.grid_size):
                    local_info[dx, dy] = self.actual_env[nx, ny]

        return local_info, top_left

    def stitch_information(self, local_info, top_left):
        """Stitch drone's local scan information into the global grid."""
        x_offset, y_offset = top_left

        for i in range(local_info.shape[0]):
            for j in range(local_info.shape[1]):
                x, y = x_offset + i, y_offset + j
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    if self.global_grid[x, y] == UNEXPLORED:
                        self.global_grid[x, y] = local_info[i, j]
                    elif self.global_grid[x, y] != local_info[i, j]:
                        # If there's a conflict, prioritize SAFE
                        if local_info[i, j] == SAFE:
                            self.global_grid[x, y] = SAFE

    def find_emergency_location(self):
        """Find a suitable emergency location in the SAFE cells."""
        safe_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.global_grid[x, y] == SAFE:
                    # Calculate distance from start
                    dist = abs(x) + abs(y)  # Manhattan distance
                    if dist > 5:  # At least a bit away from start
                        safe_cells.append((x, y, dist))

        if not safe_cells:
            print("No suitable emergency location found!")
            return None

        # Sort by distance (farthest first) and pick one of the top 5
        safe_cells.sort(key=lambda c: -c[2])
        if len(safe_cells) > 5:
            target_cell = random.choice(safe_cells[:5])
        else:
            target_cell = random.choice(safe_cells)

        return (target_cell[0], target_cell[1])

    def find_path_with_astar(self, start, goal):
        """Find a path using A* search algorithm."""
        if not goal:
            print("No emergency location set!")
            return None

        print(f"Finding path from {start} to {goal} using A*...")

        # A* implementation
        open_set = []  # Priority queue
        closed_set = set()

        # Track g_score (cost from start) and f_score (estimated total cost)
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        # Priority queue element: (f_score, position)
        import heapq
        heapq.heappush(open_set, (f_score[start], start))

        # Track the path
        came_from = {}

        # Direction vectors (up, right, down, left)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        while open_set:
            # Get the position with lowest f_score
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct and return the path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))  # Path from start to goal

            closed_set.add(current)

            # Check each neighbor
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # Skip if out of bounds or not safe
                if (not (0 <= neighbor[0] < self.grid_size and
                         0 <= neighbor[1] < self.grid_size) or
                        self.global_grid[neighbor] != SAFE or
                        neighbor in closed_set):
                    continue

                # Calculate tentative g_score
                tentative_g = g_score[current] + 1  # Cost of 1 per step

                if (neighbor not in g_score or
                        tentative_g < g_score[neighbor]):
                    # This path is better, record it
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)

                    # Add to open set if not already there
                    if neighbor not in [pos for _, pos in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return None

    def heuristic(self, a, b):
        """Manhattan distance heuristic for A*."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path_to_emergency(self):
        """Find a path from (0,0) to the emergency location."""
        # First make sure we have an emergency location
        if not self.emergency_location:
            self.emergency_location = self.find_emergency_location()
            if not self.emergency_location:
                print("Could not set emergency location!")
                return

        # Find the path
        start = (0, 0)
        path = self.find_path_with_astar(start, self.emergency_location)

        if path:
            self.path = path
            self.path_found = True
            print(f"Path found! Length: {len(path)}")
        else:
            print("No path found to emergency location!")
            self.path_found = False

    def place_next_drone(self):
        """Place the next drone and update the global grid with its scan."""
        if self.current_drone + 1 < len(self.drone_positions):
            self.current_drone += 1
            self.expanding = True
            self.expansion_timer = 0.0
            self.current_expanded.clear()

            # Get drone information
            position = self.drone_positions[self.current_drone]
            size = self.drone_sizes[self.current_drone]

            # Perform scan and stitch information
            local_info, top_left = self.drone_scan(position, size)
            self.stitch_information(local_info, top_left)

            return True
        return False

    def update_expansion(self, dt):
        """Update the drone coverage expansion animation."""
        if not self.expanding:
            return

        # Update animation timer
        self.expansion_timer += dt
        progress = min(1.0, self.expansion_timer / self.animation_duration)

        # If animation is complete
        if progress >= 1.0:
            self.expanding = False
            coverage = np.sum(self.global_grid != UNEXPLORED)
            self.coverage_history.append(coverage)

            # Auto-find emergency location if we've covered enough of the grid
            if not self.emergency_location:
                unexplored_percent = np.sum(self.global_grid == UNEXPLORED) / (self.grid_size * self.grid_size)
                if unexplored_percent < 0.3:  # If more than 70% explored
                    self.emergency_location = self.find_emergency_location()
                    if self.emergency_location:
                        print(f"Emergency location set to: {self.emergency_location}")

    def reset_animation(self):
        """Reset the animation to the beginning."""
        self.global_grid = np.full((self.grid_size, self.grid_size), UNEXPLORED, dtype=np.int32)
        self.global_grid[0, 0] = SAFE  # Start is always known
        self.coverage_history = [0]
        self.current_drone = -1
        self.animation_active = False
        self.expanding = False
        self.expansion_timer = 0.0
        self.current_expanded.clear()

        # Reset path finding
        self.emergency_location = None
        self.path = None
        self.path_found = False

    def draw_scene(self):
        """Draw the grid, drones, and path."""
        self.screen.fill(WHITE)

        # Draw the grid and cell contents
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * self.cell_size, x * self.cell_size,
                    self.cell_size, self.cell_size
                )

                # Draw cell based on global grid state
                if self.global_grid[x, y] == UNEXPLORED:
                    pygame.draw.rect(self.screen, (100, 100, 100), rect)  # Dark gray
                elif self.global_grid[x, y] == OBSTACLE:
                    pygame.draw.rect(self.screen, (180, 0, 0), rect)  # Red
                else:  # SAFE
                    pygame.draw.rect(self.screen, (200, 200, 255), rect)  # Light blue

                # Draw grid lines
                pygame.draw.rect(self.screen, BLACK, rect, 1)

        # Draw the path if found
        if self.path:
            for i, (x, y) in enumerate(self.path):
                if i == 0 or i == len(self.path) - 1:
                    continue  # Skip start and end points

                path_rect = pygame.Rect(
                    y * self.cell_size, x * self.cell_size,
                    self.cell_size, self.cell_size
                )
                # Fill with solid yellow instead of just an outline
                pygame.draw.rect(self.screen, YELLOW, path_rect)

                # Show step number for longer paths
                if len(self.path) > 10 and i % 5 == 0:
                    text = self.font.render(str(i), True, BLACK)
                    text_rect = text.get_rect(center=(
                        y * self.cell_size + self.cell_size // 2,
                        x * self.cell_size + self.cell_size // 2
                    ))
                    self.screen.blit(text, text_rect)

        # Draw start position (green)
        start_rect = pygame.Rect(0, 0, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, GREEN, start_rect, 4)

        # Draw emergency location if set (red)
        if self.emergency_location:
            x, y = self.emergency_location
            em_rect = pygame.Rect(
                y * self.cell_size, x * self.cell_size,
                self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.screen, RED, em_rect, 4)

        # Draw the drones
        for i in range(self.current_drone + 1):
            if i >= len(self.drone_positions):
                continue

            x, y = self.drone_positions[i]
            size = self.drone_sizes[i]
            color = DRONE_COLORS[i % len(DRONE_COLORS)]
            half = (size - 1) // 2

            # Calculate bounding box for drone coverage
            left = max(0, y - half)
            top = max(0, x - half)
            width = min(size, self.grid_size - left)
            height = min(size, self.grid_size - top)

            # Draw drone coverage area
            drone_rect = pygame.Rect(
                left * self.cell_size,
                top * self.cell_size,
                width * self.cell_size,
                height * self.cell_size
            )
            drone_surf = pygame.Surface((width * self.cell_size, height * self.cell_size), pygame.SRCALPHA)
            drone_surf.fill((color[0], color[1], color[2], 80))  # Semitransparent
            self.screen.blit(drone_surf, (drone_rect.x, drone_rect.y))

            # Draw drone center
            center_x = y * self.cell_size + self.cell_size // 2
            center_y = x * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, color, (center_x, center_y), self.cell_size // 4)

            # Show drone number
            label = self.font.render(str(i + 1), True, WHITE)
            label_rect = label.get_rect(center=(center_x, center_y))
            self.screen.blit(label, label_rect)

    def draw_info_panel(self):
        """Draw the information panel."""
        px = self.grid_size * self.cell_size + 10
        py = 10
        pw = 380
        ph = self.grid_size * self.cell_size

        # Panel background
        pygame.draw.rect(self.screen, GREY, (px, py, pw, ph))
        pygame.draw.rect(self.screen, BLACK, (px, py, pw, ph), 2)

        # Title
        title = self.title_font.render("Drone Coverage & Pathfinding", True, BLACK)
        self.screen.blit(title, (px + 10, py + 10))

        # Statistics
        total_cells = self.grid_size * self.grid_size
        explored = np.sum(self.global_grid != UNEXPLORED)
        safe_cells = np.sum(self.global_grid == SAFE)
        obstacle_cells = np.sum(self.global_grid == OBSTACLE)
        unexplored = total_cells - explored

        lines = [
            f"Grid Size: {self.grid_size}x{self.grid_size}",
            f"Total Cells: {total_cells}",
            f"Explored: {explored} ({explored * 100 / total_cells:.1f}%)",
            f"Safe Path Cells: {safe_cells}",
            f"Obstacle Cells: {obstacle_cells}",
            f"Unexplored: {unexplored}",
            f"Drones: {self.current_drone + 1}/{len(self.drone_positions)}"
        ]

        # Path information
        if self.emergency_location:
            lines.append("")
            lines.append(f"Emergency at: {self.emergency_location}")

            if self.path:
                lines.append(f"Path Found: {len(self.path)} steps")
                lines.append(f"Start: (0, 0)")

        # Display lines
        y_offset = 60
        for line in lines:
            text = self.font.render(line, True, BLACK)
            self.screen.blit(text, (px + 10, py + y_offset))
            y_offset += 25

        # Coverage chart
        chart_x = px + 20
        chart_y = py + 270
        chart_w = pw - 40
        chart_h = 150

        pygame.draw.rect(self.screen, WHITE, (chart_x, chart_y, chart_w, chart_h))
        pygame.draw.rect(self.screen, BLACK, (chart_x, chart_y, chart_w, chart_h), 1)

        chart_title = self.font.render("Coverage Progress", True, BLACK)
        self.screen.blit(chart_title, (chart_x, chart_y - 25))

        # Draw coverage history graph
        if len(self.coverage_history) > 1:
            max_cov = total_cells
            points = []

            for i, cov in enumerate(self.coverage_history):
                x = chart_x + (i * chart_w / (len(self.coverage_history) - 1))
                y = chart_y + chart_h - (cov * chart_h / max_cov)
                points.append((x, y))

            # Draw line connecting points
            if len(points) > 1:
                pygame.draw.lines(self.screen, RED, False, points, 2)

            # Draw points
            for point in points:
                pygame.draw.circle(self.screen, BLUE, (int(point[0]), int(point[1])), 3)

        # Instructions
        instructions = [
            "Space: Play/pause animation",
            "R: Reset animation",
            "P: Find path to emergency",
            "+/-: Speed up/slow down",
            "Esc: Exit"
        ]

        y_offset = py + ph - 140
        for instruction in instructions:
            text = self.font.render(instruction, True, BLACK)
            self.screen.blit(text, (px + 10, y_offset))
            y_offset += 22

    def run(self):
        """Main animation loop."""
        running = True
        time_acc = 0.0

        while running:
            dt = self.clock.tick(30) / 1000.0  # Delta time in seconds

            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.animation_active = not self.animation_active
                    elif event.key == pygame.K_r:
                        self.reset_animation()
                    elif event.key == pygame.K_p:
                        self.find_path_to_emergency()
                    elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                        self.animation_speed = max(0.05, self.animation_speed - 0.1)
                    elif event.key in [pygame.K_MINUS, pygame.K_UNDERSCORE]:
                        self.animation_speed = min(2.0, self.animation_speed + 0.1)

            # Auto-start if no drones placed yet
            if self.current_drone < 0 and not self.expanding and not self.animation_active:
                if len(self.drone_positions) > 0:
                    self.place_next_drone()

            # Update expansion animation
            self.update_expansion(dt)

            # Auto-advance to next drone when expansion is complete
            if not self.expanding and self.animation_active:
                time_acc += dt
                if time_acc > self.animation_speed:
                    time_acc = 0.0
                    advanced = self.place_next_drone()

                    # If all drones are placed
                    if not advanced:
                        self.animation_active = False

                        # Check if grid is fully explored
                        unexplored = np.sum(self.global_grid == UNEXPLORED)
                        if unexplored == 0 and not self.path_found:
                            print("Grid fully explored! Finding emergency path...")
                            self.find_path_to_emergency()
                        elif not self.path_found:
                            print(f"Grid exploration: {100 - (unexplored * 100 / self.grid_size ** 2):.1f}%")

            # Draw the scene
            self.draw_scene()
            self.draw_info_panel()
            pygame.display.flip()

        pygame.quit()


def run_visualization(grid_size=20, json_file=None):
    """Run the drone coverage and pathfinding simulation."""
    simulator = DronePathfinder(grid_size=grid_size)

    # Use drone positions from JSON file if provided
    if json_file and os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract drone data
            positions = []
            sizes = []

            for pos in data.get("drone_positions", []):
                positions.append(tuple(pos))

            for radius in data.get("drone_radii", []):
                sizes.append(radius)

            if len(positions) != len(sizes):
                print("Warning: Mismatched drone positions and sizes in JSON file")

            simulator.drone_positions = positions
            simulator.drone_sizes = sizes

            print(f"Loaded {len(positions)} drones from {json_file}")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            print("Cannot continue without JSON data")
            return
    else:
        print(f"Error: JSON file {json_file} not found")
        print("Cannot continue without drone position data")
        return

    # Calculate coverage cells for each drone
    simulator.drone_coverage_cells = []
    for i, (x, y) in enumerate(simulator.drone_positions):
        size = simulator.drone_sizes[i]
        half_size = (size - 1) // 2
        cells = []
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                gx, gy = x + dx, y + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    cells.append((gx, gy))
        simulator.drone_coverage_cells.append(cells)

    print(f"Using {len(simulator.drone_positions)} drones for {grid_size}×{grid_size} grid coverage")
    print("Starting simulation...")

    # Run the simulation
    simulator.run()


if __name__ == "__main__":
    # Parse command line argument for grid size
    grid_size = 7  # Default

    if len(sys.argv) > 1:
        try:
            grid_size = int(sys.argv[1])
        except ValueError:
            print(f"Invalid grid size: {sys.argv[1]}")
            print(f"Using default: {grid_size}x{grid_size}")

    # Always use the JSON file matching the grid size
    json_file = f"drone_coverage_results_{grid_size}.json"

    if os.path.exists(json_file):
        print(f"Using drone positions from: {json_file}")
        run_visualization(grid_size, json_file)
    else:
        print(f"Error: Required JSON file {json_file} not found")
        print("Please ensure the drone position data is available")