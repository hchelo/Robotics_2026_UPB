"""
RRT* + RDP + Khepera I — Carrera simultánea → GIF
Guarda la animación completa como carrera_khepera.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
from PIL import Image
import io, os, time

# ─────────────────────────────────────────────
#  PARÁMETROS
# ─────────────────────────────────────────────
XDIM   = 100
YDIM   = 100
ETA    = 2
D      = 2
GAMMA  = 200
RADIUS = 3
DILAT  = 10

START  = np.array([5.0,  90.0])
TARGET = np.array([90.0,  5.0])

ROBOT_RADIUS    = 2.58
ROBOT_SPEED     = 0.8
WAYPOINT_THRESH = 1.5
RDP_EPSILON     = 3.0

# ─────────────────────────────────────────────
#  OBSTÁCULOS
# ─────────────────────────────────────────────
RAW_OBSTACLES = np.array([
    [0,  70, 68, 73],
    [30, 100, 38, 42],
    [60,  80, 33, 38],
    [60,  80,  0,  8],
], dtype=float)

def dilate_obstacles(obs, d):
    return np.clip(obs + np.array([-d, d, -d, d]), 0, None)

OBSTACLES = dilate_obstacles(RAW_OBSTACLES, DILAT)

# ─────────────────────────────────────────────
#  RRT*
# ─────────────────────────────────────────────
def point_in_obstacle(p, obs):
    x, y = p
    return (obs[0] <= x <= obs[1]) and (obs[2] <= y <= obs[3])

def obstacle_free(p):
    return not any(point_in_obstacle(p, ob) for ob in OBSTACLES)

def segment_free(p1, p2, steps=20):
    for t in np.linspace(0, 1, steps):
        if not obstacle_free(p1 + t*(p2-p1)):
            return False
    return True

def nearest(V, xrand):
    return np.argmin(np.linalg.norm(V - xrand, axis=1))

def steer(xnearest, xrand):
    d = np.linalg.norm(xrand - xnearest)
    return xrand.copy() if d < ETA else xnearest + ETA*(xrand-xnearest)/d

def near(V, xnew, r):
    return np.where(np.linalg.norm(V - xnew, axis=1) <= r)[0]

def run_rrt_star(max_iter=4000):
    V      = [START.copy()]
    parent = [-1]
    cost   = [0.0]
    edges  = []        # (p1, p2) para dibujar después
    found  = False
    goal_idx = -1

    for it in range(max_iter):
        xrand    = TARGET.copy() if np.random.rand() < 0.05 else \
                   np.array([XDIM*np.random.rand(), YDIM*np.random.rand()])
        Varr     = np.array(V)
        idx_nn   = nearest(Varr, xrand)
        xnearest = Varr[idx_nn]
        xnew     = steer(xnearest, xrand)

        if not segment_free(xnearest, xnew):
            continue

        n         = len(V)
        r         = min(ETA, (GAMMA*np.log(n)/n)**(1.0/D))
        near_idxs = near(Varr, xnew, r)
        xmin_idx  = idx_nn
        c_min     = cost[idx_nn] + np.linalg.norm(xnearest - xnew)

        for ni in near_idxs:
            xnear  = Varr[ni]
            c_cand = cost[ni] + np.linalg.norm(xnear - xnew)
            if segment_free(xnear, xnew) and c_cand < c_min:
                xmin_idx, c_min = ni, c_cand

        new_idx = len(V)
        V.append(xnew); parent.append(xmin_idx); cost.append(c_min)
        edges.append((np.array(V[xmin_idx]), xnew.copy()))

        for ni in near_idxs:
            xnear  = np.array(V[ni])
            c_thru = c_min + np.linalg.norm(xnew - xnear)
            if segment_free(xnew, xnear) and c_thru < cost[ni]:
                cost[ni] = c_thru; parent[ni] = new_idx

        if not found and np.linalg.norm(xnew - TARGET) < RADIUS:
            found = True; goal_idx = new_idx
            break

    nodes = []
    idx = goal_idx
    while idx != -1:
        nodes.append(V[idx]); idx = parent[idx]
    nodes.reverse()
    return np.array(nodes), cost[goal_idx], edges

# ─────────────────────────────────────────────
#  RDP
# ─────────────────────────────────────────────
def rdp_simplify(path, epsilon):
    def _rdp(pts, eps):
        if len(pts) < 3:
            return pts.tolist()
        dmax, idx = 0.0, 0
        p0, pn = pts[0], pts[-1]
        d_vec = pn - p0
        d_len = np.linalg.norm(d_vec)
        for i in range(1, len(pts)-1):
            if d_len < 1e-10:
                d = np.linalg.norm(pts[i] - p0)
            else:
                t = np.clip(np.dot(pts[i]-p0, d_vec)/(d_len**2), 0, 1)
                d = np.linalg.norm(pts[i] - (p0 + t*d_vec))
            if d > dmax:
                dmax, idx = d, i
        if dmax > eps:
            return _rdp(pts[:idx+1], eps)[:-1] + _rdp(pts[idx:], eps)
        return [pts[0].tolist(), pts[-1].tolist()]
    return np.array(_rdp(path, epsilon))

# ─────────────────────────────────────────────
#  COLORES / ESTILO
# ─────────────────────────────────────────────
BG    = '#0a0a0a'
PANEL = '#0d0d0d'
C_RRT = '#f0c040'
C_RDP = '#ff44ff'
C_WIN = '#00ff88'

# ─────────────────────────────────────────────
#  DIBUJAR ENTORNO
# ─────────────────────────────────────────────
def draw_environment(ax, title, title_color):
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, XDIM); ax.set_ylim(0, YDIM)
    ax.set_aspect('equal')
    ax.tick_params(colors='#555')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.set_title(title, color=title_color,
                 fontfamily='monospace', fontsize=10, pad=6)

    for ob in RAW_OBSTACLES:
        w, h = ob[1]-ob[0], ob[3]-ob[2]
        ax.add_patch(patches.Rectangle(
            (ob[0], ob[2]), w, h,
            lw=1, edgecolor='#ff4444',
            facecolor='#2a0808', alpha=0.9, zorder=2))

    for center, color, label in [
        (START,  '#00ff88', 'START'),
        (TARGET, '#ff4466', 'GOAL')
    ]:
        ax.add_patch(plt.Circle(center, RADIUS,
                                color=color, alpha=0.4, zorder=5))
        ax.add_patch(plt.Circle(center, RADIUS,
                                color=color, fill=False, lw=2, zorder=6))
        ax.text(center[0]+RADIUS+0.5, center[1]+RADIUS+0.5, label,
                color=color, fontsize=7, zorder=7)

# ─────────────────────────────────────────────
#  HELPERS — ruedas
# ─────────────────────────────────────────────
def _make_wheel(ax, zorder=22):
    W = ROBOT_RADIUS * 0.55
    H = ROBOT_RADIUS * 0.22
    rect = patches.Rectangle(
        (-W/2, -H/2), W, H,
        linewidth=0.8, edgecolor='#555555',
        facecolor='#1a1a1a', zorder=zorder)
    ax.add_patch(rect)
    return rect

def _update_wheel(rect, cx, cy, angle_rad):
    t = (transforms.Affine2D()
         .translate(cx, cy)
         .rotate_around(cx, cy, angle_rad))
    rect.set_transform(t + rect.axes.transData)

# ─────────────────────────────────────────────
#  ROBOT
# ─────────────────────────────────────────────
class KheperaRobot:
    def __init__(self, ax, path, color, label):
        self.ax     = ax
        self.path   = path
        self.color  = color
        self.pos    = path[0].copy().astype(float)
        self.angle  = 0.0
        self.wp_idx = 1
        self.done   = False
        self.steps  = 0

        self.body = plt.Circle(self.pos, ROBOT_RADIUS,
                               color=color, zorder=20, alpha=0.95)
        ax.add_patch(self.body)

        self.arrow, = ax.plot([], [], color='#111', lw=2.5, zorder=21)

        self.wheels = [_make_wheel(ax), _make_wheel(ax)]
        self._repos_wheels()

        self.trail_x = [self.pos[0]]
        self.trail_y = [self.pos[1]]
        self.trail,  = ax.plot(self.trail_x, self.trail_y,
                               color=color, lw=1.8, alpha=0.55,
                               zorder=7, ls='--')

        self.timer_txt = ax.text(
            1, 97, 'T: 0', color=color,
            fontfamily='monospace', fontsize=9, zorder=30,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='#111', edgecolor=color, alpha=0.8))

        self.wp_txt = ax.text(
            1, 91, f'WP 0/{len(path)-1}', color='#aaa',
            fontfamily='monospace', fontsize=8, zorder=30)

    def _repos_wheels(self):
        off = ROBOT_RADIUS * 0.9
        for i, side in enumerate([-1, 1]):
            cx = self.pos[0] - off*np.sin(self.angle)*side
            cy = self.pos[1] + off*np.cos(self.angle)*side
            _update_wheel(self.wheels[i], cx, cy, self.angle)

    def step(self, sim_t):
        if self.done or self.wp_idx >= len(self.path):
            if not self.done:
                self.done = True
                self.body.set_color(C_WIN)
                self.timer_txt.set_color(C_WIN)
                self.timer_txt.set_text(f'DONE {sim_t:.1f}s')
                self.wp_txt.set_text('META!')
                self.wp_txt.set_color(C_WIN)
            return False

        target = self.path[self.wp_idx]
        diff   = target - self.pos
        dist   = np.linalg.norm(diff)

        if dist < WAYPOINT_THRESH:
            self.wp_idx += 1
            return True

        self.angle = np.arctan2(diff[1], diff[0])
        step = min(ROBOT_SPEED, dist)
        self.pos = self.pos + step*diff/dist
        self.steps += 1

        self.body.center = self.pos
        al = ROBOT_RADIUS*1.7
        self.arrow.set_data(
            [self.pos[0], self.pos[0]+al*np.cos(self.angle)],
            [self.pos[1], self.pos[1]+al*np.sin(self.angle)])
        self._repos_wheels()

        self.trail_x.append(self.pos[0])
        self.trail_y.append(self.pos[1])
        self.trail.set_data(self.trail_x, self.trail_y)

        pct = int(100*self.wp_idx/(len(self.path)-1))
        self.timer_txt.set_text(f'T:{sim_t:.1f}  {pct}%')
        self.wp_txt.set_text(f'WP {self.wp_idx}/{len(self.path)-1}')
        return True

# ─────────────────────────────────────────────
#  PANEL COMPARATIVO
# ─────────────────────────────────────────────
def fill_comparison(ax, path_rrt, path_rdp, t_rrt, t_rdp):
    ax.cla()
    ax.set_facecolor('#111')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis('off')

    winner  = 'RRT*' if t_rrt <= t_rdp else 'RDP'
    wc      = C_RRT  if winner == 'RRT*' else C_RDP
    red     = 100*(1 - len(path_rdp)/len(path_rrt))

    lines = [
        ('━━  COMPARATIVA FINAL  ━━', '#ffffff', 11),
        ('', '#fff', 8),
        (f'  Waypoints RRT* : {len(path_rrt):>4}', C_RRT, 9),
        (f'  Waypoints RDP  : {len(path_rdp):>4}   ({red:.0f}% menos)', C_RDP, 9),
        ('', '#fff', 8),
        (f'  Pasos RRT*     : {int(t_rrt):>4}', C_RRT, 9),
        (f'  Pasos RDP      : {int(t_rdp):>4}', C_RDP, 9),
        ('', '#fff', 8),
        (f'  Diferencia     : {abs(int(t_rrt)-int(t_rdp)):>4} pasos', '#aaa', 9),
        ('', '#fff', 8),
        (f'  GANADOR : {winner}', wc, 13),
    ]
    y = 0.94
    for txt, col, sz in lines:
        ax.text(0.05, y, txt, color=col, fontfamily='monospace',
                fontsize=sz, va='top')
        y -= 0.085

    # Barras
    mx = max(t_rrt, t_rdp, 1)
    for y_b, val, col in [(0.09, t_rrt, C_RRT), (0.04, t_rdp, C_RDP)]:
        ax.barh([y_b], [val/mx*0.88], left=0.05, height=0.035,
                color=col, alpha=0.8)


# ─────────────────────────────────────────────
#  CAPTURA DE FRAME
# ─────────────────────────────────────────────
def capture(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100,
                facecolor=fig.get_facecolor(), bbox_inches=None)
    buf.seek(0)
    img = Image.open(buf).copy().convert('RGB')
    buf.close()
    return img

# ─────────────────────────────────────────────
#  MAIN — genera GIF sin plt.show() interactivo
# ─────────────────────────────────────────────
def main():
    np.random.seed(42)
    frames = []

    # ── Figura ──────────────────────────────
    fig = plt.figure(figsize=(14, 9), facecolor=BG, dpi=100)
    gs  = gridspec.GridSpec(2, 2,
                            height_ratios=[3, 1],
                            hspace=0.35, wspace=0.12,
                            left=0.04, right=0.96,
                            top=0.92, bottom=0.04)
    ax_rrt = fig.add_subplot(gs[0, 0])
    ax_rdp = fig.add_subplot(gs[0, 1])
    ax_cmp = fig.add_subplot(gs[1, :])

    draw_environment(ax_rrt, 'RRT*  — ruta original',   C_RRT)
    draw_environment(ax_rdp, 'RDP   — ruta simplificada', C_RDP)
    ax_cmp.set_facecolor('#111'); ax_cmp.axis('off')

    # ── FASE 1: RRT* ────────────────────────
    print("Calculando RRT*...")
    fig.suptitle('Calculando RRT*...', color='#aaa',
                 fontfamily='monospace', fontsize=12)

    path_rrt, _, edges = run_rrt_star(max_iter=4000)
    print(f"  Ruta: {len(path_rrt)} waypoints")

    # Dibujar árbol RRT* completo (una sola vez)
    for p1, p2 in edges:
        ax_rrt.plot([p1[0],p2[0]], [p1[1],p2[1]],
                    color='#00ccff', lw=0.6, alpha=0.2, zorder=1)
    for ax in [ax_rrt, ax_rdp]:
        ax.plot(path_rrt[:,0], path_rrt[:,1],
                color='#ffdd00', lw=1.2, alpha=0.4, zorder=8, ls=':')

    # ── FASE 2: RDP ─────────────────────────
    path_rdp = rdp_simplify(path_rrt, RDP_EPSILON)
    red = 100*(1 - len(path_rdp)/len(path_rrt))
    print(f"  RDP: {len(path_rrt)} → {len(path_rdp)} pts ({red:.0f}% reducción)")

    ax_rrt.plot(path_rrt[:,0], path_rrt[:,1],
                color=C_RRT, lw=2.5, zorder=9)
    ax_rrt.scatter(path_rrt[:,0], path_rrt[:,1],
                   s=12, color='#ff8800', zorder=10)

    ax_rdp.plot(path_rdp[:,0], path_rdp[:,1],
                color=C_RDP, lw=2.5, zorder=9)
    ax_rdp.scatter(path_rdp[:,0], path_rdp[:,1],
                   s=40, color='#ff00ff', zorder=10, marker='D')

    # Frame estático "rutas calculadas"
    fig.suptitle('Rutas calculadas — iniciando carrera',
                 color='#ffffff', fontfamily='monospace', fontsize=12)
    for _ in range(6):      # ~0.3s @ 20fps
        frames.append(capture(fig))

    # ── FASE 3: Countdown ───────────────────
    for i in range(3, 0, -1):
        fig.suptitle(f'Saliendo en  {i} ...',
                     color='#ff4444', fontfamily='monospace', fontsize=16)
        for _ in range(8):
            frames.append(capture(fig))

    # ── FASE 4: Carrera ─────────────────────
    robot_rrt = KheperaRobot(ax_rrt, path_rrt, C_RRT, 'RRT*')
    robot_rdp = KheperaRobot(ax_rdp, path_rdp, C_RDP, 'RDP')

    fig.suptitle('RRT*  vs  RDP  —  EN MARCHA!',
                 color=C_WIN, fontfamily='monospace', fontsize=13)

    sim_t = 0
    t_rrt_done = None
    t_rdp_done = None

    print("Simulando carrera...")
    while not (robot_rrt.done and robot_rdp.done):
        r1 = robot_rrt.step(sim_t)
        r2 = robot_rdp.step(sim_t)
        sim_t += 1

        if robot_rrt.done and t_rrt_done is None:
            t_rrt_done = sim_t
        if robot_rdp.done and t_rdp_done is None:
            t_rdp_done = sim_t

        # Actualizar título
        if robot_rrt.done and not robot_rdp.done:
            fig.suptitle('RRT* llego primero! — RDP en camino...',
                         color=C_RRT, fontfamily='monospace', fontsize=11)
        elif robot_rdp.done and not robot_rrt.done:
            fig.suptitle('RDP llego primero! — RRT* en camino...',
                         color=C_RDP, fontfamily='monospace', fontsize=11)
        elif not robot_rrt.done and not robot_rdp.done:
            fig.suptitle(f'RRT* vs RDP  —  paso {sim_t}',
                         color='#ffffff', fontfamily='monospace', fontsize=12)

        # Capturar 1 de cada 2 pasos para no hacer el GIF gigante
        if sim_t % 2 == 0:
            frames.append(capture(fig))

        if sim_t % 100 == 0:
            print(f"  paso {sim_t}  |  RRT*{'✓' if robot_rrt.done else '…'}  RDP{'✓' if robot_rdp.done else '…'}")

    t_rrt_done = t_rrt_done or sim_t
    t_rdp_done = t_rdp_done or sim_t
    winner  = 'RRT*' if t_rrt_done <= t_rdp_done else 'RDP'
    wc      = C_RRT  if winner == 'RRT*' else C_RDP

    # ── FASE 5: Resultado ───────────────────
    fill_comparison(ax_cmp, path_rrt, path_rdp, t_rrt_done, t_rdp_done)
    fig.suptitle(f'GANADOR : {winner}',
                 color=wc, fontfamily='monospace', fontsize=16)

    for _ in range(30):     # ~1.5s de resultado final
        frames.append(capture(fig))

    plt.close(fig)

    # ── GUARDAR GIF ─────────────────────────
    out = 'carrera_khepera.gif'
    print(f"\nGuardando GIF ({len(frames)} frames)...")

    first = frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    rest  = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
             for f in frames[1:]]

    first.save(
        out,
        save_all=True,
        append_images=rest,
        duration=50,        # 20 fps
        loop=0,
        optimize=False,
        disposal=2
    )

    size_mb = os.path.getsize(out)/1e6
    print(f"GIF guardado: {out}  ({size_mb:.1f} MB, {len(frames)} frames)")


if __name__ == '__main__':
    main()