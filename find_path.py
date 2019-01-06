import numpy as np
import time
import sys


try:
    import unicornhathd as u

except ModuleNotFoundError:
    print("No UnicornHatHD module found.")
    sys.exit(1)


def distance(p, g, divide=True):
    d = abs(g[0] - p[0]) + abs(g[1] - p[1])
    if divide:
        return 2 if d == 0 else 1 / d
    return d


def coord_to_int(x, y):
    return x * 16 + y


def build_walls(goal):
    n = 0
    while True:
        x = np.random.randint(2, 10)
        y = np.random.randint(2, 10)
        wall = [(x, y + i) for i in range(6)]
        x, y = wall[-1]
        wall.extend([(x + i, y) for i in range(6)])
        if not goal in wall:
            return wall
        n += 1


def draw_walls(walls):
    for x, y in walls:
        u.set_pixel_hsv(x, y, 0.1, 0.85, 0.33)


def get_action(p, qtable, eps):
    if np.random.random() > eps:
        row = coord_to_int(*p)
        val = max([qtable[row][col] for col in range(4)])
        return qtable[row].index(val)
    else:
        return np.random.randint(4)


def apply_action(p, a, g, wall):
    x, y = p
    if a == 0:
        y -= 1
    if a == 1:
        x -= 1
    if a == 2:
        y += 1
    if a == 3:
        x += 1
    p1 = (min(15, max(0, x)), min(15, max(0, y)))
    if p1 == p:
        return p1, -2
    elif p1 in wall:
        return p, -1
    return p1, distance(p1, g)


def show_path(path, goal):
    for p in reversed(path):
        dist = distance(p, goal, divide=False)
        h = max(0.1, 1 - (dist / 16))
        s = max(0.1, 1 - (dist / 16))
        b = max(0.1, ((h + s) + 0.001) / 2.)
        u.set_pixel_hsv(*p, h, s, b)
        u.show()
        time.sleep(0.2)


def reset():
    qtable = [[-1.] * 4 for _ in range(16 * 16)]
    start = (0, 0)
    goal = (np.random.randint(16), np.random.randint(16))
    goal_reward = 2.
    epsilon = 0.5
    best = -2.
    walls = build_walls(goal)
    return qtable, start, goal, goal_reward, epsilon, best, walls


def play(verbose=0):
    qtable, start, goal, goal_reward, epsilon, best, walls = reset()
    while True:
        p = start
        path = []
        dist = distance(p, goal, divide=False)
        for step in range(int(dist) + 8):
            u.clear()
            u.set_pixel_hsv(*goal, .22, .72, .4)
            draw_walls(walls)
            h = (dist / 16)
            s = 1 - (dist / 16)
            b = max(0.2, ((h + s) + 0.001) / 2.)
            u.set_pixel_hsv(*p, h, s, b)
            u.show()
            a = get_action(p, qtable, epsilon)
            p1, reward = apply_action(p, a, goal, walls)
            path.append(p1)
            if reward == goal_reward:
                qtable[coord_to_int(*p)][a] = reward
                epsilon = max(0.05, epsilon - 0.1)
                show_path(path, goal)
                if np.random.random() > 0.9:
                    qtable, start, goal, goal_reward, epsilon, best, walls = reset()
                    print("new goal: {}".format(goal))
                break
            if reward > best:
                best = reward
                if verbose:
                    print(p1, reward, epsilon)
            qtable[coord_to_int(*p)][a] = reward + 0.5 * max([qtable[coord_to_int(*p1)][col] for col in range(4)])
            p = p1


if __name__ == '__main__':
    play(verbose=1)
