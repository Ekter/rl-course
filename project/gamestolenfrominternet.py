# https://github.com/ZippyCodeYT/Zippy_Codes/tree/main

from ursina import *
import random as r

# app = Ursina()

class ShadowGame():
    def __init__(self):
        super().__init__()
        self.app = Ursina()
        self.dino = None
        self.label = None
        self.gordos = []
        self.cadoizos = []
        self.setup_game()

    def setup_game(self) :
        destroy(self.dino, delay=0)
        destroy(self.label, delay=0)
        for gordo in self.gordos:
            destroy(gordo, delay=0)
        for cadoizo in self.cadoizos:
            destroy(cadoizo, delay = 0)

        self.dino = Animation("assets/shad", x=-5)
        self.head_collider = Entity(
            model='sphere',
            collider='sphere',
            parent=self.dino,
            scale=(0.9, 0.9, 0.9),  # Taille pour la tête
            position=(0, 0, 0),
            enabled=False  # Actif par défaut
        )
        self.ground1 = Entity(model="quad", texture="assets/ground", scale=(50, 0.25, 1), z=1, y=-0.5)
        self.ground2 = duplicate(self.ground1, x=50)
        self.pair = [self.ground1, self.ground2]

        self.cadoizo = Entity(model="quad",texture = "assets/cadoizo2", x = 200, y = 0.1, collider = "sphere")
        self.cadoizo2 = Entity(model="quad",texture = "assets/cadoizo2", x = 10, y = 1,collider = "sphere", scale = (0.6,0.6,0))
        self.cadoizos = [self.cadoizo,self.cadoizo2]

        self.gordo1 = Animation(name="assets/gordo", x=10, collider="box")
        self.gordo2 = Animation(name="assets/gordo", x=14, collider="box")
        self.gordo3 = Animation(name="assets/gordo", x=20, collider="box")
        self.gordo4 = Animation(name="assets/gordo", x=26, collider="box")
        self.gordos = [self.gordo1, self.gordo2, self.gordo3, self.gordo4]

        self.label = Text(text=f"Points: {0}", color=color.black, position=(-0.5, 0.4))
        self.points = 0

    def update(self):
        self.points += 1
        self.label.text = f"Points: {self.points}"
        for self.ground in self.pair:
            self.ground.x -= 6 * time.dt
            if self.ground.x < -35:
                self.ground.x += 100
        for c in self.gordos:
            c.x -= 6 * time.dt
            if c.x<-10:
                c.x = r.randint(20,30)
        for d in self.cadoizos:
            d.x -=6 *time.dt
        if self.head_collider.intersects().hit:
            self.dino.texture = "assets/ohno"
            self.setup_game()
            self.points = 0

    def input(self,key):
        # if key in ("space", "j","up"):
            # if dino.y < 0.01:
                # sound.play()
                # dino.animate_y(2, duration=0.4, curve=curve.out_sine)
                # dino.animate_y(0, duration=0.4, delay=0.4, curve=curve.in_sine)
                
        print(key)
        if key.split(" ")[0] in ("space", "j","up"):
            # if dino.y < 0.01:
                self.dino.y +=0.5
        else:
            quit()

# def view():
    

Shadgame = ShadowGame()
window.fullscreen = True
window.color = color.white
camera.orthographic = True
camera.fov = 10
update = Shadgame.update
input = Shadgame.input
Shadgame.app.run()
print("37")