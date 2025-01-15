# https://github.com/ZippyCodeYT/Zippy_Codes/tree/main

from ursina import *
import random as r

app = Ursina()

window.fullscreen = True
window.color = color.white

dino = Animation("assets/shad", x=-5)


head_collider = Entity(
    model='sphere',
    collider='sphere',
    parent=dino,
    scale=(0.9, 0.9, 0.9),  # Taille pour la tête
    position=(0, 0, 0),
    enabled=False  # Actif par défaut
)

ground1 = Entity(model="quad", texture="assets/ground", scale=(50, 0.25, 1), z=1, y=-0.5)
ground2 = duplicate(ground1, x=50)
pair = [ground1, ground2]

cadoizo = Entity(model="quad",texture = "assets/cadoizo2", x = 200, y = 0.1, collider = "sphere")
cadoizo2 = Entity(model="quad",texture = "assets/cadoizo2", x = 10, y = 1,collider = "sphere", scale = (0.6,0.6,0))
cadoizos = [cadoizo,cadoizo2]

# collider_cad = Entity(
#     model = 'sphere',
#     collider='sphere',
#     parent=cadoizo2,
#     scale = (1,1,1),
#     position=(0,0,0),
#     enabled=True
# )




gordo1 = Animation(name="assets/gordo", x=10, collider="box")
gordo2 = Animation(name="assets/gordo", x=14, collider="box")
gordo3 = Animation(name="assets/gordo", x=20, collider="box")
gordo4 = Animation(name="assets/gordo", x=26, collider="box")
gordos = [gordo1, gordo2, gordo3, gordo4]




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
        if c.x<-10:
            c.x = r.randint(20,30)
    for d in cadoizos:
        d.x -=6 *time.dt
    if head_collider.intersects().hit:
        dino.texture = "assets/ohno"
        application.quit()
        points = 0




def input(key):
    # if key in ("space", "j","up"):
        # if dino.y < 0.01:
            # sound.play()
            # dino.animate_y(2, duration=0.4, curve=curve.out_sine)
            # dino.animate_y(0, duration=0.4, delay=0.4, curve=curve.in_sine)
            
    print(key)
    if key.split(" ")[0] in ("space", "j","up"):
        # if dino.y < 0.01:
            dino.y +=0.5
    else:
        application.quit()

    def view(self):
        closest = min(self.gordos + self.cadoizos, key=lambda x: x.x)
        x = closest.x
        y_up = closest.y
        y_down = closest.y + closest.scale_y
        return x, y_up, y_down



camera.orthographic = True
camera.fov = 10

app.run()
