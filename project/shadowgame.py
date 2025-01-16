from ursina import *
import random as random

# app = Ursina()


class ShadowGame:
    def __init__(self):
        super().__init__()
        self.app = Ursina()
        self.dino = None
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
        destroy(self.dino, delay=0)
        destroy(self.label, delay=0)
        for gordo in self.gordos:
            destroy(gordo, delay=0)
        for cadoizo in self.cadoizos:
            destroy(cadoizo, delay=0)

        self.dino = Animation("assets/shad", x=-5)
        self.head_collider = Entity(
            model="sphere",
            collider="sphere",
            parent=self.dino,
            scale=(0.9, 0.9, 0.9),  # Taille pour la tête
            position=(0, 0, 0),
            enabled=True,  # Actif par défaut
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
            scale=(0.7, 0.7, 0.7),
            position=(0, 0, 0),
            enabled=True,
        )
        self.gordocollider2 = Entity(
            model="sphere",
            collider="sphere",
            parent=self.gordo2,
            scale=(0.7, 0.7, 0.7),
            position=(0, 0, 0),
            enabled=True,
        )
        self.gordocollider3 = Entity(
            model="sphere",
            collider="sphere",
            parent=self.gordo3,
            scale=(0.7, 0.7, 0.7),
            position=(0, 0, 0),
            enabled=True,
        )
        self.gordocollider4 = Entity(
            model="sphere",
            collider="sphere",
            parent=self.gordo4,
            scale=(0.7, 0.7, 0.7),
            position=(0, 0, 0),
            enabled=True,
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
            self.dino.y += self.velocity * time.dt
            self.velocity += self.gravity * time.dt
            if self.dino.y <= 0:
                self.dino.y = 0
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
            self.dino.texture = "assets/ohno"
            self.setup_game()
            self.points = 0

    def input(self, key):
        # if key in ("space", "j","up"):
        # if dino.y < 0.01:
        # sound.play()
        # dino.animate_y(2, duration=0.4, curve=curve.out_sine)
        # dino.animate_y(0, duration=0.4, delay=0.4, curve=curve.in_sine)

        print(key)
        if key.split(" ")[0] in ("space", "j", "up"):
            # if dino.y < 0.01:
            if not self.jumping:
                self.jumping = True
                self.velocity = self.jump_speed
        elif key.split(" ")[0] in ("down"):
            if self.jumping:
                self.velocity -= 13
                # self.dino.y -= 0.1
            elif self.dino.y == 0 and key != "down arrow up":
                self.crouching = True
                self.dino.y = -0.1
            elif key == "down arrow up":
                self.dino.y = 0
        else:
            quit()

    def view(self):
        closest = min(self.gordos + self.cadoizos, key=lambda x: x.x)
        x = closest.x
        y_up = closest.y
        y_down = closest.y + closest.scale_y
        return x, y_up, y_down


if __name__ == "__main__":
    Shadgame = ShadowGame()
    window.fullscreen = True
    window.color = color.white
    camera.orthographic = True
    camera.fov = 10
    update = Shadgame.update
    input = Shadgame.input
    Shadgame.app.run()
