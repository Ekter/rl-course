# https://github.com/ZippyCodeYT/Zippy_Codes/tree/main

from ursina import *
import random as r

app = Ursina()
window.fullscreen = True
window.color = color.white


dino = Animation("assets/dino", collider="box", x=-5)

ground1 = Entity(model="quad", texture="assets/ground", scale=(50, 0.25, 1), z=1, y=-0.5)
ground2 = duplicate(ground1, x=50)
pair = [ground1, ground2]


gordo1 = Animation(name="assets/gordo", x=10, collider="box")
gordo2 = Animation(name="assets/gordo", x=14, collider="box")
gordo3 = Animation(name="assets/gordo", x=20, collider="box")
gordo4 = Animation(name="assets/gordo", x=26, collider="box")
gordos = [gordo1, gordo2, gordo3, gordo4]


def newGordo():
    if gordos[0].X < 0:
        last = gordos.pop(0)
        last.x_setter(r.randint(20, 30))
        gordos.append(last)

    invoke(newGordo, delay=2)


newGordo()


label = Text(text=f"Points: {0}", color=color.black, position=(-0.5, 0.4))
points = 0


def update():
    global points
    points += 1
    label.text = f"Points: {points}"
    for ground in pair:
        ground.x -= 6 * time.dt
        if ground.x < -35:
            ground.x += 100
    for c in gordos:
        c.x -= 6 * time.dt
    if dino.intersects().hit:
        dino.texture = "assets/hit"
        application.pause()


# sound = Audio("assets/beep", autoplay=False)


def input(key):
    if key == "space":
        if dino.y < 0.01:
            # sound.play()
            dino.animate_y(2, duration=0.4, curve=curve.out_sine)
            dino.animate_y(0, duration=0.4, delay=0.4, curve=curve.in_sine)


camera.orthographic = True
camera.fov = 10

app.run()
