# ...

from pynput.keyboard import Key, Controller, Listener
import time
from tqdm import tqdm


cont = Controller()

def jump():
    cont.tap(Key.space)

def down(t:time = 0.1):
    cont.press(Key.down)
    time.sleep(t)
    cont.release(Key.down)

def restart():
    time.sleep(3)
    cont.tap(Key.space)

print("warning! gonna start in 1s")
time.sleep(1)
print("I started")
# cont.press(Key.alt)
# cont.press(Key.tab)
# time.sleep(0.1)
# cont.release(Key.tab)
# cont.release(Key.alt)



for i in tqdm(range(10)):
    restart()
    for _ in range(5):
        jump()
        time.sleep(0.5)
    down(0.5)

cont.press(Key.alt)
cont.tap(Key.tab)
cont.release(Key.alt)
