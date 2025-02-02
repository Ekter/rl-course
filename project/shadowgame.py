import random as random
import sys

from ursina import *

# app = Ursina()


class ShadowGame:
    def __init__(self):
        super().__init__()
        self.app = Ursina()
        self.shadow = None
        self.label = None
        self.gordos = []
        self.cadoizos = []
        self.gravity = -9.8
        self.velocity = 0
        self.jump_speed = 6
        self.jumping = True
        self.end = False
        if len(sys.argv) > 1:
            self.i = sys.argv[1]
            self.filescore = f"data/discore{self.i}.txt"
            self.fileaction = f"data/action{self.i}.txt"
        else:
            self.filescore = "data/discore.txt"
            self.fileaction = "data/action.txt"
            self.i = None
            # print("NO I ############################################################")

        self.setup_game()

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
            visible=False,
        )

        self.placeholder = Entity(
            x=100, y=0, enabled=False, scale=(1, 1, 1), visible=False
        )

        self.ground1 = Entity(
            model="quad", texture="assets/ground", scale=(50, 0.25, 1), z=1, y=-0.3
        )
        self.ground2 = duplicate(self.ground1, x=50)
        self.pair = [self.ground1, self.ground2]

        self.cadoizo = Entity(
            model="quad",
            texture="assets/cadoizo2",
            x=random.randint(50, 70),
            y=0.5,
            collider="sphere",
            scale=(0.6, 0.6, 0),
        )
        self.cadoizo2 = Entity(
            model="quad",
            texture="assets/cadoizo2",
            x=random.randint(40, 90),
            y=1.1,
            collider="sphere",
            scale=(0.6, 0.6, 0),
        )
        self.cadoizo.sped = 4
        self.cadoizo2.sped = 8
        self.cadoizos = [self.cadoizo, self.cadoizo2]

        self.gordo1 = Animation(name="assets/gordo", x=random.randint(10, 20), y=0.3)
        self.gordo2 = Animation(name="assets/gordo", x=random.randint(20, 30), y=0.3)
        self.gordo3 = Animation(name="assets/gordo", x=random.randint(30, 40), y=0.3)
        self.gordo4 = Animation(name="assets/gordo", x=random.randint(40, 50), y=0.3)
        self.gordos = [self.gordo1, self.gordo2, self.gordo3, self.gordo4]

        self.gordocollider1 = Entity(
            model="sphere",
            collider="sphere",
            parent=self.gordo1,
            scale=(0.6, 0.6, 0.6),
            position=(0, 0, 0),
            enabled=True,
            visible=False,
        )
        self.gordocollider2 = Entity(
            model="sphere",
            collider="sphere",
            parent=self.gordo2,
            scale=(0.6, 0.6, 0.6),
            position=(0, 0, 0),
            enabled=True,
            visible=False,
        )
        self.gordocollider3 = Entity(
            model="sphere",
            collider="sphere",
            parent=self.gordo3,
            scale=(0.6, 0.6, 0.6),
            position=(0, 0, 0),
            enabled=True,
            visible=False,
        )
        self.gordocollider4 = Entity(
            model="sphere",
            collider="sphere",
            parent=self.gordo4,
            scale=(0.6, 0.6, 0.6),
            position=(0, 0, 0),
            enabled=True,
            visible=False,
        )

        self.label = Text(
            text=f"Points: {0}",
            color=color.black,
            position=(-0.5, 0.4),
            scale=(3, 3, 3),
        )
        self.points = 0

    def update(self):
        self.points += 1
        self.label.text = f"Points: {self.points}, run={self.i}"
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
            # self.setup_game()
            # self.points = 0
            self.end = True
        with open(self.filescore, "w", encoding="utf-8") as file:
            file.write(str(self.view()))
        # time.sleep(1/30)
        if self.end:
            sys.exit(0)

        with open(self.fileaction, "r", encoding="utf-8") as file:
            character = file.read()
        # print(character,"###################################################################")
        match character:
            case "space" | "j" | "up":
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
                if self.shadow.y < 0:
                    self.shadow.y = 0

    def input(self, key):
        print(key)

    def inputv2(self, key):
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
        elif key.split(" ")[0] in ("down", "s"):
            if self.jumping:
                self.velocity -= 13
                # self.dino.y -= 0.1
            elif self.shadow.y == 0 and key != "down arrow up":
                self.crouching = True
                self.shadow.y = -0.1
            elif key == "down arrow up":
                self.shadow.y = 0
        else:
            quit()

    def view(self):
        ordered = sorted(
            filter(
                lambda obj: obj.x > self.shadow.x - 0.3, self.gordos + self.cadoizos
            ),
            key=lambda x: x.x,
        )
        try:
            closest = ordered[0]
        except IndexError:
            closest = self.placeholder
        try:
            second = ordered[1]
        except IndexError:
            second = self.placeholder
            return 100, 0, 0, 100, 0, 0, self.points, self.end
        return (
            closest.x,
            closest.y,
            closest.y + closest.scale_y,
            second.x,
            second.y,
            second.y + second.scale_y,
            self.shadow.y,
            self.points,
            self.end,
        )


if __name__ == "__main__":
    Shadgame = ShadowGame()
    window.fullscreen = False
    window.borderless = False
    window.size = (1900, 500)
    window.position = (0, 0)
    window.color = color.white
    camera.orthographic = True
    camera.fov = 5
    camera.position = (0, 2.1, -5)
    update = Shadgame.update
    input = Shadgame.input
    Shadgame.app.run()
