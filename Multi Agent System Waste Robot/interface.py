import pygame
import sys
import tkinter as tk
from tkinter import simpledialog
from environment import Environment
import asyncio
import os

# Initialize Tkinter to collect dimensions
def get_environment_size():
    root = tk.Tk()
    root.withdraw()
    
    # Reopen main window to center the pop-ups
    root.deiconify()
    root.geometry(f"1x1+{root.winfo_screenwidth() // 2}+{root.winfo_screenheight() // 2}")
    root.withdraw()
    
    rows = simpledialog.askinteger("Input", "Enter number of rows:", minvalue=1, parent=root)
    cols = simpledialog.askinteger("Input", "Enter number of columns:", minvalue=1, parent=root)
    root.destroy()  # Close the main Tkinter window
    return rows, cols

# Initial environment setup
rows, cols = 11, 11 #get_environment_size()
layout_2 = [[0,0,0,0,5,0,0,0,0,0,0],
            [5,5,5,0,5,0,5,0,5,5,5],
            [0,5,0,0,5,0,5,0,0,0,0],
            [0,5,0,0,0,0,0,0,0,5,0],
            [0,0,0,0,5,5,5,0,0,5,0],
            [0,5,0,0,5,0,0,0,0,5,0],
            [0,5,0,0,5,5,5,0,0,5,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,5,0,5,0,0,0,5,5,5,0],
            [5,5,0,5,0,5,5,5,0,5,0],
            [0,0,0,5,0,0,0,0,0,5,0],]

layout_1 = [[0,5,0,0,0,0,5,0,0,0,0],
            [0,5,0,5,5,0,5,5,5,5,0],
            [0,5,5,5,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,5,5],
            [5,5,0,0,5,0,5,0,0,0,0],
            [0,5,0,0,5,0,5,0,0,5,5],
            [0,5,0,0,5,5,5,5,0,0,0],
            [0,0,0,0,0,5,0,5,0,0,0],
            [5,5,5,0,0,5,0,5,0,0,0],
            [5,0,0,0,0,0,0,0,0,5,0],
            [0,0,5,5,0,0,0,0,0,5,0]]
env = Environment(cols, rows, layout_2)  # Create the environment

# Center the Pygame window
os.environ['SDL_VIDEO_CENTERED'] = '1'
# Initialize Pygame
pygame.init()
CELL_SIZE = 50
WIDTH, HEIGHT = cols * CELL_SIZE, rows * CELL_SIZE
screen = pygame.display.set_mode((WIDTH + 600, HEIGHT))
pygame.display.set_caption("Multi-Agent Autonomous Waste Collection System")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREY = (128, 128, 128)

# Truck
truck_image = pygame.image.load("Images/truck_image.png")
truck_image = pygame.transform.scale(truck_image, (CELL_SIZE - 5, CELL_SIZE - 5))  # Adjust to cell size

def apply_red_filter(image):
    # Create a surface with the same size as the original image and with alpha channel (transparency)
    red_surface = pygame.Surface(image.get_size(), pygame.SRCALPHA)
    
    # Fill the surface with red color and 50% transparency (128 is 50% of 255)
    red_surface.fill((255, 0, 0, 128))  # (R, G, B, A) -> (255, 0, 0, 128)
    
    # Overlay the original image on the red surface with transparency effect
    red_surface.blit(image, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    
    return red_surface

truck_image_modified = apply_red_filter(truck_image)

def draw_environment(grid, trucks, traffic_edges, bin_names):
    """Draw the environment."""
    for row in range(rows):
        for col in range(cols):
            color = WHITE
            if grid[row][col] == 1:  # Bin
                color = GREEN
            elif grid[row][col] == 2:   # Central
                color = BLUE
            elif grid[row][col] == -1:  # Obstacle
                color = BLACK
            elif grid[row][col] == 9:   # Roadblock
                color = RED
            elif grid[row][col] == 5:
                color = GREY

            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
            )

            pygame.draw.rect(
                    screen,
                    BLACK,
                    pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    1,
                )

            # Check if there's a name associated with the square
            if (row, col) in bin_names:
                name = bin_names[(row, col)]
                text_surface = pygame.font.Font(None, 24).render(name, True, (0, 0, 0))  # Black text
                text_rect = text_surface.get_rect(center=(col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2))
                screen.blit(text_surface, text_rect)  # Draw the text on screen

    # Draw lines that represent the edges that are traffic
    if traffic_edges:
        for (pos1, pos2) in traffic_edges:
            # pos1 and pos2 are (row, col) tuples
            x1 = pos1[1] * CELL_SIZE + CELL_SIZE // 2
            y1 = pos1[0] * CELL_SIZE + CELL_SIZE // 2
            x2 = pos2[1] * CELL_SIZE + CELL_SIZE // 2
            y2 = pos2[0] * CELL_SIZE + CELL_SIZE // 2

            # Draw a thick orange line between the centers of pos1 and pos2
            pygame.draw.line(screen, (255, 165, 0), (x1, y1), (x2, y2), 5)

            
    # Draw the trucks
    for truck in trucks:
        x,y  = truck.position

        # Just center the image without drawing background rectangle
        image_x = y * CELL_SIZE + (CELL_SIZE - truck_image.get_width()) // 2
        image_y = x * CELL_SIZE + (CELL_SIZE - truck_image.get_height()) // 2
        if truck.is_broken:
            screen.blit(truck_image_modified, (image_x, image_y))
        else:  
            screen.blit(truck_image, (image_x, image_y))

def draw_metrics(trucks, bins):
    """Draw metrics (truck and bin status) on the right side of the screen."""
    metrics_x = WIDTH + 20  # Initial position to the right of the map
    metrics_y = 0

    # Black background for metrics
    pygame.draw.rect(screen, BLACK, pygame.Rect(metrics_x, metrics_y, 600, HEIGHT))

    # Draw bin status
    metrics_y += 20
    title = pygame.font.Font(None, 36).render("Bin Status", True, (255, 255, 255))
    screen.blit(title, (metrics_x + 20, metrics_y))
    metrics_y += 40
    for bin_id, info in bins.items():
        color = (0, 255, 0) if info[0] < 0.4*info[1] else (255, 255, 0) if info[0] < 0.7*info[1] else (255, 0, 0)
        text = pygame.font.Font(None, 28).render(
            f"{bin_id}: {info[0]}/{info[1]}", True, color
        )
        screen.blit(text, (metrics_x+20, metrics_y))
        metrics_y += 30

    # Draw truck status
    metrics_y += 20
    title = pygame.font.Font(None, 36).render("Truck Status", True, (255, 255, 255))
    screen.blit(title, (metrics_x + 20, metrics_y))
    metrics_y += 40
    for truck_id, info in trucks.items():
        truck_info = f"{truck_id}: Cap {info[0]}/{info[1]}, Fuel {info[2]}/{info[3]}"
        text = pygame.font.Font(None, 28).render(truck_info, True, (255, 255, 255))
        screen.blit(text, (metrics_x+20, metrics_y))
        metrics_y += 30

def write_file(trucks, bins):
    filename = "Test_3_Layout_2.txt"
    with open(filename, "w") as file:
        title = f"Environment with constant waste accumulation (5s) and possibility of breaks.\n\n"
        file.write(title)
        for truck in trucks:
            # Create the string with truck data
            truck_data = (
                f"TRUCK: {truck.jid}:\n"
                f"Total waste collected: {truck.collected_waste}\n"
                f"Total fuel spent: {truck.total_fuel}\n"
                f"Total distance travelled: {truck.total_distance}\n"
                f"Total number of collabs: {truck.collab}\n\n"
            )
            # Write to file
            file.write(truck_data)

        for bin in bins.values():
            # Create the string with bin data
            if (len(bin.collection_time) != 0):
                bin_data = (
                    f"BIN {bin.jid}:\n"
                    f"Times: {bin.collection_time}\n"
                    f"Average collection time: {sum(bin.collection_time)/len(bin.collection_time)}\n\n"
                )
            else:
                bin_data = (
                    f"BIN {bin.jid}:\n"
                    f"Times: {bin.collection_time}\n\n"
                )
            # Write to file
            file.write(bin_data)
    
    print(f"Data has been saved to file '{filename}'.")

async def pygame_loop():
    """Execute the main Pygame loop."""
    running = True
    pressed_keys = set()
    roadblocks = []
    last_broke_truck_time = 0
    bins_status = {}
    truck_status = {}
    bin_names = {}
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                pressed_keys.add(event.key)
                
                # Traffic Level Controls
                if event.key == pygame.K_s:
                    await env.start_system()
                    print("System initialized")
                elif event.key == pygame.K_1:
                    await env.set_traffic(level=1)
                    print("Traffic level set to 1")
                elif event.key == pygame.K_2:
                    await env.set_traffic(level=2)
                    print("Traffic level set to 2")
                elif event.key == pygame.K_3:
                    await env.set_traffic(level=3)
                    print("Traffic level set to 3")
                elif event.key == pygame.K_4:
                    await env.set_traffic(level=4)
                    print("Traffic level set to 4")
                elif event.key == pygame.K_5:
                    await env.set_traffic(level=5)
                    print("Traffic level set to 5")
                elif event.key == pygame.K_0:
                    await env.set_traffic(level=0)
                    print("Traffic reset to level 0")
            elif event.type == pygame.KEYUP:
                pressed_keys.discard(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                grid_x, grid_y = y // CELL_SIZE, x // CELL_SIZE
                if event.button==1 :
                    if pygame.K_b in pressed_keys:  # Add bin
                        await env.add_bin((grid_y, grid_x))
                        bin_names[(grid_x, grid_y)] = str(len(bin_names)+1)
                    elif pygame.K_t in pressed_keys:  # Add truck
                        await env.add_truck((grid_y, grid_x))
                    elif pygame.K_r in pressed_keys: # Add a roadblock
                        await env.add_roadBlock((grid_y,grid_x))
                        roadblocks.append((grid_y,grid_x))
                elif(event.button==3):
                    if pygame.K_r in pressed_keys:
                        if (grid_y,grid_x) in roadblocks:
                            await env.remove_roadBlock((grid_y,grid_x))
                            roadblocks.remove((grid_y,grid_x))
                        else:
                            print("No roadblock exists at this position")

        # last_broke_truck_time is the time since the last truck breakdown
        last_broke_truck_time = env.break_truck(last_broke_truck_time)
        # Get the updated environment state
        grid, trucks, bins, traffic_edges = env.update_display()

        bins_status = {}
        for bin in bins.values():
            bins_status[bin.name] = [bin.current_waste, bin.max_capacity]

        truck_status = {}
        for truck in trucks:
            truck_status[truck.name] = [truck.load, truck.max_load, truck.fuel, truck.max_fuel]

        # Update the screen
        screen.fill(WHITE)
        draw_environment(grid, trucks, traffic_edges, bin_names)
        draw_metrics(truck_status, bins_status)
        pygame.display.flip()

        await asyncio.sleep(0.03)  # Limit FPS
    
    pygame.quit()
    grid, trucks, bins, traffic_edges = env.update_display()
    write_file(trucks, bins)
    os._exit(0)

async def main():
    """Execute the main loop."""
    await pygame_loop()

asyncio.run(main())