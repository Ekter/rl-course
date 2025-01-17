from ursina import *
import random as random

# app = Ursina()


class ShadowGame:
    def __init__(self):
        super().__init__()
        self.app = Ursina()
        self.shadow = None
        self.label = None
        self.gordos = []
        self.cadoizos = []
        self.setup_game()
        self.gravity = -9.8
        self.velocity = 0
        self.jump_speed = 6
        self.jumping = True

    def setup_game(self):
        self.velocity = 0
        self.jumping = False
        destroy(self.shadow, delay=0)
        destroy(self.label, delay=0)
        for gordo in self.gordos:
            destroy(gordo, delay=0)
        for cadoizo in self.cadoizos:
            destroy(cadoizo, delay=0)

        self.shadow = Animation("assets/shad", x=-5)
        self.head_collider = Entity(
            model="sphere",
            collider="sphere",
            parent=self.shadow,
            scale=(0.9, 0.9, 0.9),
            position=(0, 0, 0),
            enabled=True,
            visible = False
        )
        self.ground1 = Entity(
            model="quad", texture="assets/ground", scale=(50, 0.25, 1), z=1, y=-0.3
        )
        self.ground2 = duplicate(self.ground1, x=50)
        self.pair = [self.ground1, self.ground2]

        self.cadoizo = Entity(
            model="quad",
            texture="assets/cadoizo2",
            x=2,
            y=0.5,
            collider="sphere",
            scale=(0.6, 0.6, 0),
        )
        self.cadoizo2 = Entity(
            model="quad",
            texture="assets/cadoizo2",
            x=10,
            y=1.1,
            collider="sphere",
            scale=(0.6, 0.6, 0),
        )
        self.cadoizo.sped = 4
        self.cadoizo2.sped = 8
        self.cadoizos = [self.cadoizo, self.cadoizo2]

        self.gordo1 = Animation(name="assets/gordo", x=10, y=0.3)
        self.gordo2 = Animation(name="assets/gordo", x=14, y=0.3)
        self.gordo3 = Animation(name="assets/gordo", x=20, y=0.3)
        self.gordo4 = Animation(name="assets/gordo", x=26, y=0.3)
        self.gordos = [self.gordo1, self.gordo2, self.gordo3, self.gordo4]

        self.gordocollider1 = Entity(
            model="sphere",
            collider="sphere",
            parent=self.gordo1,
            scale=(0.6, 0.6, 0.6),
            position=(0, 0, 0),
            enabled=True,
            visible = False
        )
        self.gordocollider2 = Entity(
            model="sphere",
            collider="sphere",
            parent=self.gordo2,
            scale=(0.6, 0.6, 0.6),
            position=(0, 0, 0),
            enabled=True,
            visible = False
        )
        self.gordocollider3 = Entity(
            model="sphere",
            collider="sphere",
            parent=self.gordo3,
            scale=(0.6, 0.6, 0.6),
            position=(0, 0, 0),
            enabled=True,
            visible = False
        )
        self.gordocollider4 = Entity(
            model="sphere",
            collider="sphere",
            parent=self.gordo4,
            scale=(0.6, 0.6, 0.6),
            position=(0, 0, 0),
            enabled=True,
            visible = False
        )
        self.label = Text(text=f"Points: {0}", color=color.black, position=(-0.5, 0.4))
        self.points = 0

    def update(self):
        self.points += 1
        self.label.text = f"Points: {self.points}"
        for self.ground in self.pair:
            self.ground.x -= 6 * time.dt
            if self.ground.x < -35:
                self.ground.x += 100
        if self.jumping:
            self.shadow.y += self.velocity * time.dt
            self.velocity += self.gravity * time.dt
            if self.shadow.y <= 0:
                self.shadow.y = 0
                self.jumping = False
        for gordo in self.gordos:
            gordo.x -= 6 * time.dt
            if gordo.x < -10:
                gordo.x = random.randint(20, 30)
        for cadoizo in self.cadoizos:
            cadoizo.x -= cadoizo.sped * time.dt
            if cadoizo.x < -10:
                cadoizo.x = random.randint(20, 30)
        if self.head_collider.intersects().hit:
            self.shadow.texture = "assets/ohno"
            self.setup_game()
            self.points = 0
        
        with open("action.txt","r",encoding="utf-8") as file:
            character = file.read()
            print(character,"#########################################################################################################")
            match character:
                case "space" | "j" | "up" :
                    if not self.jumping:
                        self.jumping = True
                        self.velocity = self.jump_speed
                case "down" | "f":
                    if self.jumping:
                        self.velocity -= 13
                    elif self.shadow.y == 0:
                        self.shadow.y = -0.1
                case "q" | "escape":
                    quit()
                case _:
                        self.shadow.y = 0

    def input(self, key):
        print(key)
        

    def view(self):
        closest = min(self.gordos + self.cadoizos, key=lambda x: x.x)
        x = closest.x
        y_up = closest.y
        y_down = closest.y + closest.scale_y
        return x, y_up, y_down


if __name__ == "__main__":
    Shadgame = ShadowGame()
    window.fullscreen = False
    window.borderless = False
    window.size = (1900, 500)
    window.position=(0,0)
    window.color = color.white
    camera.orthographic = True
    camera.fov = 5
    camera.position = (0,2.1,-5)
    update = Shadgame.update
    input = Shadgame.input
    Shadgame.app.run()
