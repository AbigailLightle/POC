import random
import math
import os
import numpy as np
from collections import defaultdict
import multiprocessing
import time

os.system('cls')

NUMBER_OF_TRIALS = 100
RENDER = False  # Whether to render the screen or not
seed = 0
trial_number = 0

# Screen Definitions
WIDTH, HEIGHT, LEGION = 1024, 768, 64

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)                 # Immune
RED = (200, 0, 0)                   # Infected
BLUE = (0, 0, 200)                  # Susceptible
YELLOW = (255, 255, 0)              # Deceased
GRAY = (100, 100, 100)              # Background
LIGHT_MAGENTA = (224, 0, 224)       # Legend

# Simulation Parameters
POPULATION = 1000              # Total number of people
INFECTION_RADIUS = 5           # Infection spread distance
INFECTION_PROBABILITY = 0.97   # Chance of spreading infection
VACCINATED_PERCENT = 0.95      # Initial immune population proportion
VACCINE_EFFECTIVENESS = 0.97   # Vaccine effectiveness
DEATH_PROBABILITY = 0.02       # Probability an infected person dies
DAYS_TO_DEATH = 10             # Min days before death chance
DAYS_TO_DEATH_MULTIPLIER = 10  # Frame/Day multiplier
DAYS_TO_RECOVERY = 80          # Days to recover
RADIUS_OF_PERSON = 5           # Person radius
FRAME_RATE = 1000              # Simulation speed

# Initialize pygame once if rendering
if RENDER:
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + LEGION))
    pygame.display.set_caption("Measles in Motion: A Digital Epidemic Simulation by Abigail Lightle")
    legend_font = pygame.font.SysFont(None, 30)


class InstantClock:
    def tick(self, framerate=0): return 0
    def get_fps(self): return 0


class Person:
    __slots__ = ('x', 'y', 'status', 'speed_x', 'speed_y', 'infection_days')
    
    def __init__(self, x, y, status="susceptible"):
        self.x = x
        self.y = y
        self.status = status
        self.speed_x = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        self.speed_y = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        self.infection_days = 0

    def move(self):
        if self.status != "deceased":
            self.x += self.speed_x
            self.y += self.speed_y

            # Bounce off walls - avoid branching where possible
            if self.x < RADIUS_OF_PERSON:
                self.x = RADIUS_OF_PERSON
                self.speed_x = -self.speed_x
            elif self.x > WIDTH - RADIUS_OF_PERSON:
                self.x = WIDTH - RADIUS_OF_PERSON
                self.speed_x = -self.speed_x
                
            if self.y < RADIUS_OF_PERSON:
                self.y = RADIUS_OF_PERSON
                self.speed_y = -self.speed_y
            elif self.y > HEIGHT + LEGION//2 - RADIUS_OF_PERSON:
                self.y = HEIGHT + LEGION//2 - RADIUS_OF_PERSON
                self.speed_y = -self.speed_y

class Grid:
    """Optimized spatial partitioning grid for efficient proximity checking"""
    def __init__(self, width, height, cell_size):
        self.cell_size = cell_size
        self.grid_width = int(width / cell_size) + 1
        self.grid_height = int(height / cell_size) + 1
        self.grid = defaultdict(list)
    
    def clear(self):
        self.grid.clear()
    
    def add_person(self, person, person_idx):
        cell_x = int(person.x / self.cell_size)
        cell_y = int(person.y / self.cell_size)
        self.grid[(cell_x, cell_y)].append(person_idx)
    
    def get_nearby_indices(self, person):
        cell_x = int(person.x / self.cell_size)
        cell_y = int(person.y / self.cell_size)
        
        # Use a pre-allocated list for better performance
        nearby_indices = []
        # Only check cells that could contain people within infection radius
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_key = (cell_x + dx, cell_y + dy)
                if cell_key in self.grid:  # Faster check than .get() with default
                    nearby_indices.extend(self.grid[cell_key])
        
        return nearby_indices


def create_population(population_size):
    people = []
    for _ in range(population_size):
        x = random.randint(RADIUS_OF_PERSON, WIDTH-RADIUS_OF_PERSON)
        y = random.randint(RADIUS_OF_PERSON, HEIGHT-RADIUS_OF_PERSON)
        if random.random() < VACCINATED_PERCENT:
            people.append(Person(x, y, status="immune"))
        else:
            people.append(Person(x, y))
    
    # Infect one random person
    people[random.randint(0, population_size-1)].status = "infected"
    return people


def check_if_infected(people, spatial_grid):
    """Optimized infection checking using spatial partitioning"""
    total_infections = 0
    
    # Update spatial grid
    spatial_grid.clear()
    for idx, person in enumerate(people):
        spatial_grid.add_person(person, idx)
    
    # Check for infections
    for idx, person in enumerate(people):
        if person.status == "infected":
            person.infection_days += 1
            
            # Check for death or recovery
            if person.infection_days >= DAYS_TO_DEATH and random.random() < DEATH_PROBABILITY/DAYS_TO_DEATH_MULTIPLIER:
                person.status = "deceased"
            elif person.infection_days >= DAYS_TO_RECOVERY and person.status != "deceased":
                person.status = "immune"
            
            # Only check for spreading infection if still infected
            if person.status == "infected":
                nearby_indices = spatial_grid.get_nearby_indices(person)
                
                for other_idx in nearby_indices:
                    if idx == other_idx:
                        continue
                        
                    other = people[other_idx]
                    if other.status in ["susceptible", "immune"]:
                        dist = math.hypot(person.x - other.x, person.y - other.y)
                        
                        if dist < INFECTION_RADIUS:
                            if other.status == "susceptible" and random.random() < INFECTION_PROBABILITY:
                                other.status = "infected"
                                total_infections += 1
                            elif other.status == "immune" and random.random() > VACCINE_EFFECTIVENESS:
                                other.status = "infected"
                                total_infections += 1
    
    return total_infections


def count_status(people):
    """Count people in each status category"""
    susceptible = immune = infected = deceased = 0
    
    for person in people:
        if person.status == "susceptible": susceptible += 1
        elif person.status == "immune": immune += 1
        elif person.status == "infected": infected += 1
        elif person.status == "deceased": deceased += 1
    
    return [susceptible, immune, infected, deceased]


def append_trial_data_to_dataset(data):
    """Save trial results to CSV file"""
    dataset_file = "Proof_Of_Concept_DataSet/measles_dataset.csv"

    # Make sure directory exists
    os.makedirs(os.path.dirname(dataset_file), exist_ok=True)

    # Append the trial data to the csv file using lock to prevent race conditions
    with file_lock:
        if os.path.exists(dataset_file):
            with open(dataset_file, "a") as file:
                file.write(str(data) + "\n")
        else:
            with open(dataset_file, "a") as file:
                file.write("fps,p_rad,days_r,dm,d_t_d,d_prob,h_i_t,vac_p,inf_p,i_rad,pop,t_inf,n_inf,imm,sus,dec,frames,seed\n")
                file.write(str(data) + "\n")


def run_single_simulation(trial_seed):
    """Run a single simulation trial with the given seed"""
    random.seed(trial_seed)

    # Initialize population and spatial grid
    people = create_population(POPULATION)
    spatial_grid = Grid(WIDTH, HEIGHT, INFECTION_RADIUS * 2)
    
    # Setup timing
    clock = InstantClock()
    number_of_frames = 0
    total_infections = 1
    
    # Main simulation loop
    running = True
    while running:
        number_of_frames += 1
        
        # Process infections
        total_infections += check_if_infected(people, spatial_grid)
        
        # Move people
        for person in people:
            person.move()
        
        # Count people by status
        status = count_status(people)
        
        # Stop if no more infected
        if status[2] <= 0:
            running = False
        
        clock.tick(FRAME_RATE)
    
    # Prepare trial data
    trial01 = f"{FRAME_RATE: g},{RADIUS_OF_PERSON: g},{DAYS_TO_RECOVERY: g}"
    trial02 = f"{DAYS_TO_DEATH_MULTIPLIER: g},{DAYS_TO_DEATH: g},{DEATH_PROBABILITY: g}"
    trial03 = f"{0.93: g},{VACCINATED_PERCENT: g},{INFECTION_PROBABILITY: g}"  # Using 0.93 for HERD_IMMUNITY_THRESHOLD
    trial04 = f"{INFECTION_RADIUS: g},{POPULATION: g},{total_infections: g},{status[2]: g}"
    trial05 = f"{status[1]: g},{status[0]: g},{status[3]: g},{number_of_frames: g}, {trial_seed: g}"
    trial_data = f"{trial01},{trial02},{trial03},{trial04},{trial05}"
    
    return trial_data, trial_seed


def process_result(result):
    """Process and save completed simulation results"""
    global trial_number
    trial_data, trial_seed = result
    append_trial_data_to_dataset(trial_data)
    
    with counter_lock:
        trial_number += 1
        print(f"\rTrial {trial_number} of {NUMBER_OF_TRIALS} completed (seed: {trial_seed})", end="")


def run_simulation():
    global seed, trial_number, file_lock, counter_lock
    # Initialize locks regardless of render mode
    file_lock = multiprocessing.Lock()
    counter_lock = multiprocessing.Lock()
    start_time = time.time()
    if RENDER:
        # Run in single-thread mode when rendering
        for _ in range(NUMBER_OF_TRIALS):
            seed += 1
            random.seed(seed)

            # Initialize population and spatial grid
            people = create_population(POPULATION)
            spatial_grid = Grid(WIDTH, HEIGHT, INFECTION_RADIUS * 2)
            
            # Setup timing
            clock = pygame.time.Clock()
            number_of_frames = 0
            total_infections = 1
            
            # Main simulation loop
            running = True
            while running:
                number_of_frames += 1
                
                screen.fill(GRAY)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                # Process infections
                total_infections += check_if_infected(people, spatial_grid)
                
                # Move people and render
                for person in people:
                    person.move()
                    person.draw(screen)
                
                # Count people by status
                status = count_status(people)
                
                # Stop if no more infected
                if status[2] <= 0:
                    running = False
                
                # Update caption
                pygame.display.set_caption(f"Measles in Motion: A Digital Epidemic Simulation by Abigail Lightle\
                      (FPS: {int(clock.get_fps()+1)})   (Frames: {number_of_frames})   (Population: {POPULATION})   (Seed: {seed} of {NUMBER_OF_TRIALS})    (Vaccinated: {int(VACCINATED_PERCENT*100)}%)")
                
                # Draw legend
                pygame.draw.rect(screen, LIGHT_MAGENTA, (0, HEIGHT + LEGION//2, WIDTH, LEGION//2))
                
                pos_x = 0
                screen.blit(legend_font.render(f"Total Infections: {total_infections:04d}", True, BLACK), (pos_x + 10, HEIGHT + LEGION//2 + 8))
                
                pos_x = 280
                pygame.draw.circle(screen, RED, (pos_x, HEIGHT + LEGION//2 + 17), 5)
                screen.blit(legend_font.render(f"Infected: {status[2]:04d}", True, BLACK), (pos_x + 10, HEIGHT + LEGION//2 + 8))
                
                pos_x += 180
                pygame.draw.circle(screen, GREEN, (pos_x, HEIGHT + LEGION//2 + 17), 5)
                screen.blit(legend_font.render(f"Immune: {status[1]:04d}", True, BLACK), (pos_x + 10, HEIGHT + LEGION//2 + 8))
                
                pos_x += 175
                pygame.draw.circle(screen, BLUE, (pos_x, HEIGHT + LEGION//2 + 17), 5)
                screen.blit(legend_font.render(f"Susceptible: {status[0]:04d}", True, BLACK), (pos_x + 10, HEIGHT + LEGION//2 + 8))
                
                pos_x += 220
                pygame.draw.circle(screen, YELLOW, (pos_x, HEIGHT + LEGION//2 + 17), 5)
                screen.blit(legend_font.render(f"Deceased: {status[3]:04d}", True, BLACK), (pos_x + 10, HEIGHT + LEGION//2 + 8))
                
                pygame.display.flip()
                
                clock.tick(FRAME_RATE)
            
            # Save trial data
            trial01 = f"{FRAME_RATE: g},{RADIUS_OF_PERSON: g},{DAYS_TO_RECOVERY: g}"
            trial02 = f"{DAYS_TO_DEATH_MULTIPLIER: g},{DAYS_TO_DEATH: g},{DEATH_PROBABILITY: g}"
            trial03 = f"{0.93: g},{VACCINATED_PERCENT: g},{INFECTION_PROBABILITY: g}"  # Using 0.93 for HERD_IMMUNITY_THRESHOLD
            trial04 = f"{INFECTION_RADIUS: g},{POPULATION: g},{total_infections: g},{status[2]: g}"
            trial05 = f"{status[1]: g},{status[0]: g},{status[3]: g},{number_of_frames: g}, {seed: g}"
            trial_data = f"{trial01},{trial02},{trial03},{trial04},{trial05}"
            
            append_trial_data_to_dataset(trial_data)
            
            os.system('cls')
            trial_number += 1
            print(f"\nTrial {trial_number} of {NUMBER_OF_TRIALS} data appended to dataset.\n")
        print(f"\nAll simulations completed in {time.time() - start_time:.2f} seconds")
    else:
        # Run in multi-process mode for non-rendering simulations
        start_time = time.time()
        
        # Create locks for shared resources
        file_lock = multiprocessing.Lock()
        counter_lock = multiprocessing.Lock()
        
        # Create a process pool with as many processes as there are CPU cores
        num_processes = multiprocessing.cpu_count()
        print(f"Starting {NUMBER_OF_TRIALS} simulations using {num_processes} processes...")
        
        # Generate all seeds upfront
        seeds = list(range(seed + 1, seed + NUMBER_OF_TRIALS + 1))
        
        # Create process pool and run simulations
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Map the seeds to the simulation function and process results as they complete
            for _ in pool.imap_unordered(run_single_simulation, seeds):
                process_result(_)
        
        print(f"\nAll simulations completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    # Global locks for multiprocessing
    file_lock = None
    counter_lock = None
    
    run_simulation()
    if RENDER:
        pygame.quit()
