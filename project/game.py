import numpy as np
import enum
import tkinter as tk


class Apparence(enum.Enum):
    CACTUS = 0
    TWO_CACTUS = 1
    THREE_CACTUS = 2
    BIRD = 3
    CLOUD = 4


class Obstacle:
    def __init__(self, distance, height, width, z_height = 0, speed=0, apparence:Apparence=Apparence.CACTUS):
        self.distance = distance
        self.height = height
        self.width = width
        self.z_height = z_height
        self.speed = speed
        self.apparence = apparence

    @staticmethod
    def create_random_obstacle():
        distance = np.random.randint(100, 1000)
        height = np.random.randint(0, 100)
        width = np.random.randint(0, 100)
        speed = np.random.randint(0, 10)
        apparence = np.random.choice(list(Apparence))
        z_height = np.random.randint(0, 100)
        return Obstacle(distance, height, width, z_height, speed, apparence)

    def update(self, speed):
        self.distance -= (self.speed+speed)


class Dino:
    def __init__(self):
        self.height = 0
        self.speed = 1
        self.is_alive = True
        self.z_speed = 0
        self.gravity = 0.1
        self.generate_obstacle()
        self.root = None
        self.canvas = None
        self.DT = 0.01

    def generate_obstacle(self):
        self.obstacle = Obstacle.create_random_obstacle()

    def draw(self):
        if self.root is None:
            self.root = tk.Tk()
            self.root.geometry('1050x300')
            self.root.title('DINOSAURRRRr')

            self.canvas = tk.Canvas(self.root, width=1000, height=250, bg='white')
            self.canvas.pack(anchor=tk.CENTER, expand=True)

        self.canvas.delete('all')
        self.canvas.create_rectangle((50, 150), (70, 230), fill='red')
        self.canvas.create_rectangle((self.obstacle.distance-self.obstacle.width/2,230- self.obstacle.height), (self.obstacle.distance+self.obstacle.width/2, 50- self.obstacle.height), fill='green')
        self.root.update()


    def update(self):
        self.z_speed -= self.gravity
        self.height += self.z_speed
        if self.height < 0:
            self.height = 0
            self.z_speed = 0
        self.speed+=0.001
        self.obstacle.update(self.speed)
        if self.obstacle.distance < -50:
            self.generate_obstacle()
        self.draw()


d = Dino()

while d.is_alive:
    d.update()
    d.root.after(int(d.DT*1000))