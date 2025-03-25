import pygame
import random
import math
import os; os.system('cls')


# Initialize Pygame
pygame.init()

# Screen Definitions
WIDTH, HEIGHT, LEGEND  = 1024, 768, 64
screen = pygame.display.set_mode((WIDTH, HEIGHT + LEGEND))
pygame.display.set_caption("Measles in Motion: A Digital Epidemic Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)                 # Immune
RED = (200, 0, 0)                   # Infected
BLUE = (0, 0, 200)                  # Susceptible
YELLOW = (255,255,0)                # Deceased
GRAY = (100, 100, 100)              # Background 
LIGHT_MAGENTA = (224, 0, 224)       # Legend

# Fonts
legend_font = pygame.font.SysFont(None, 30)

# Measles Herd Immunity Simulation Parameters
POPULATION = 500                # Total number of people in the simulation

INFECTION_RADIUS = 5            # Distance within which an infected 
                                # person can spread the infection

INFECTION_PROBABILITY = 0.97    # Represents the chance of spreading
                                # the infection when an infected person 
                                # is near a susceptible person

VACCINATED_PERCENT = 0.40       # Proportion of the population that
                                # starts as immune (vaccinated)

HERD_IMMUNITY_THRESHOLD = 0.93  # The percentage of immune people
                                # needed to stop uncontrolled spread

DEATH_PROBABILITY = 0.02        # Probability an infected person dies per day

DAYS_TO_DEATH = 10              # Minimum number of days an infected person must
                                # be sick before they have a chance to die

DAYS_TO_DEATH_MULTIPLIER = 10   # Frame/Day multiplier

DAYS_TO_RECOVERY = 80           # Number of days it takes to recover

RADIUS_OF_PERSON = 5            # Radius of each person in the simulation

FRAME_RATE = 10                 # use 10-60 to demo program and 1000
                                # to run full speed


# START - Individual Person Class  ****************************************************
class Person:
    def __init__(self, x, y, status="susceptible", radius=RADIUS_OF_PERSON):
        self.x = x
        self.y = y
        self.status = status
        
        self.color = BLUE if self.status == "susceptible" else\
                     GREEN if self.status == "immune" else\
                     RED if self.status == "infected" else\
                     YELLOW  # Deceased

        self.speed_x = -random.choice([1,-2,3,-4,5])
        self.speed_y = -random.choice([-1,2,-3,4,-5])

        self.radius = radius
        self.age = "adult"  # Child, adult, senior
        self.infection_days = 0  # Days since infected

    def move(self):
        if self.status != "deceased":
            self.x += self.speed_x
            self.y += self.speed_y

        # Have people bounce off the walls and ceiling
            if self.x - self.radius <= 0 or self.x + self.radius >= WIDTH:
                self.speed_x = -self.speed_x

            if self.y - self.radius <= 0 or self.y + self.radius >= HEIGHT + LEGEND//2:
                self.speed_y = -self.speed_y

    def draw(self):
        self.color = BLUE if self.status == "susceptible" else\
                     GREEN if self.status == "immune" else\
                     RED if self.status == "infected" else\
                     YELLOW  # Deceased
        
        if self.color == YELLOW:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)),
                RADIUS_OF_PERSON * 1.75)
        else:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)),
                RADIUS_OF_PERSON)
        
# END - Individual Person Class  ******************************************************

# Initialize population
people = []
for _ in range(POPULATION):
    x = random.randint(RADIUS_OF_PERSON, WIDTH-RADIUS_OF_PERSON)
    y = random.randint(RADIUS_OF_PERSON, HEIGHT-RADIUS_OF_PERSON)
    if random.random() < VACCINATED_PERCENT:
        people.append(Person(x, y, status="immune"))
    else:
        people.append(Person(x, y))

# Infect one random person
people[random.randint(0, POPULATION-1)].status = "infected"

def check_if_infected():
    for person in people:
        if person.status == "infected":
            person.infection_days += 1
            
            # If infected for long enough, introduce a probability of death
            if person.infection_days >= DAYS_TO_DEATH and random.random() < DEATH_PROBABILITY/DAYS_TO_DEATH_MULTIPLIER:
                person.status = "deceased"
                # continue        # GOTO THE NEXT PERSON
            if person.infection_days >= DAYS_TO_RECOVERY and person.status != "deceased":
                person.status = "immune"
                # continue        # GOTO THE NEXT PERSON
            
            for other in people:
                if other.status == "susceptible":
                    dist = math.hypot(person.x - other.x, person.y - other.y)
                    if dist < INFECTION_RADIUS and random.random() < INFECTION_PROBABILITY:
                        other.status = "infected"

# START: Main loop ----------------------------------------------------------------------------------------------------
running = True
clock = pygame.time.Clock()

number_of_frames = 0
days = 0
herd_immunity_achieved = False
while running:
    number_of_frames += 1
    screen.fill(GRAY)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.K_r:
            running = False
            running = True
    
    check_if_infected()

    # check person status and MOVE & DRAW each person
    status = [0,0,0,0]
    for person in people:
        if person.status == "susceptible": status[0] += 1
        if person.status == "immune": status[1] += 1
        if person.status == "infected": status[2] += 1
        if person.status == "deceased":
            status[3] += 1
        person.move()
        person.draw()
    
    # Display stats in Caption
    cap01 = f"Measles in Motion: A Digital Epidemic Simulation - STATUS: "
    cap02 = f"{status[0]} - immune(GREEN) {status[1]} - infected(RED) {status[2]}"
    cap03 = f" - deceased(YELLOW) {status[3]}      Frames: {number_of_frames}"
    pygame.display.set_caption(f"{cap01}{cap02}{cap03}")

    pygame.display.set_caption(f"Measles in Motion: A Digital Epidemic Simulation (Frames: {number_of_frames})")

    # Display legend
    pygame.draw.rect(screen, LIGHT_MAGENTA, (0, HEIGHT + LEGEND//2 , WIDTH, LEGEND//2))

    pos_x = 150
    pygame.draw.circle(screen, RED, (pos_x, HEIGHT + LEGEND//2 + 17), 5)
    screen.blit(legend_font.render(f"Infected: {status[2]:04d}", True, BLACK), (pos_x + 10, HEIGHT + LEGEND//2 + 8 ))

    pos_x += 180
    pygame.draw.circle(screen, GREEN, (pos_x, HEIGHT + LEGEND//2 + 17), 5)
    screen.blit(legend_font.render(f"Immune: {status[1]:04d}", True, BLACK), (pos_x + 10, HEIGHT + LEGEND//2 + 8))

    pos_x += 175
    pygame.draw.circle(screen, BLUE, (pos_x, HEIGHT + LEGEND//2 + 17), 5)
    screen.blit(legend_font.render(f"Susceptible: {status[0]:04d}", True, BLACK), (pos_x + 10, HEIGHT + LEGEND//2 +8))

    pos_x += 220
    pygame.draw.circle(screen, YELLOW, (pos_x, HEIGHT + LEGEND//2 + 17), 5)
    screen.blit(legend_font.render(f"Deceased: {status[3]:04d}", True, BLACK), (pos_x + 10, HEIGHT + LEGEND//2 +8))

    if status[2] <=0:
        running = False

    pygame.display.flip()
    clock.tick(FRAME_RATE)
    days += 1
# End of Main loop ----------------------------------------------------------------------------------------------------

# Append This Trial to Dataset for Future Data Science Analysis
# Not Yet Include

# wait for a key hit or "X" clicked to QUIT
waiting = True
while waiting:
    event = pygame.event.wait()  # Wait for an event
    if event.type == pygame.KEYDOWN:
        waiting = False  # Exit the loop when a key is pressed
    if event.type == pygame.QUIT:
        waiting = False  # Exit the loop when "X" is clicked

os.system('cls')
pygame.quit()