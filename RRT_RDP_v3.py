"""
RRT* + RDP + Khepera I — Carrera simultánea
Izquierda : Khepera sigue ruta RRT* original
Derecha   : Khepera sigue ruta RDP simplificada
Ambos parten al mismo tiempo — comparativa final de tiempos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
import time

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
    [30,100, 38, 42],
    [60,  80, 33, 38],
    [60, 80,  0,  8],
], dtype=float)

def dilate_obstacles(obs, d):
    return np.clip(obs + np.array([-d, d, -d, d]), 0, None)

OBSTACLES = dilate_obstacles(RAW_OBSTACLES, DILAT)

# ─────────────────────────────────────────────
#  RRT* — funciones
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

def run_rrt_star(ax_tree, max_iter=4000):
    """Corre RRT* dibujando en ax_tree. Devuelve la ruta."""
    V      = [START.copy()]
    parent = [-1]
    cost   = [0.0]
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

        for ni in near_idxs:
            xnear  = np.array(V[ni])
            c_thru = c_min + np.linalg.norm(xnew - xnear)
            if segment_free(xnew, xnear) and c_thru < cost[ni]:
                cost[ni] = c_thru; parent[ni] = new_idx

        xpar = np.array(V[xmin_idx])
        ax_tree.plot([xpar[0], xnew[0]], [xpar[1], xnew[1]],
                     color='#00ccff', lw=1, alpha=0.3, zorder=1)

        if not found and np.linalg.norm(xnew - TARGET) < RADIUS:
            found = True; goal_idx = new_idx

        if it % 40 == 0:
            plt.pause(0.001)
        if found:
            break

    nodes = []
    idx = goal_idx
    while idx != -1:
        nodes.append(V[idx]); idx = parent[idx]
    nodes.reverse()
    return np.array(nodes), cost[goal_idx]

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
#  SETUP VISUAL
# ─────────────────────────────────────────────
def draw_environment(ax, title, title_color):
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, XDIM); ax.set_ylim(0, YDIM)
    ax.set_aspect('equal')
    ax.tick_params(colors='#555')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.set_title(title, color=title_color, fontfamily='monospace', fontsize=10, pad=6)

    for ob in RAW_OBSTACLES:
        w, h = ob[1]-ob[0], ob[3]-ob[2]
        ax.add_patch(patches.Rectangle(
            (ob[0], ob[2]), w, h,
            lw=1, edgecolor='#ff4444', facecolor='#2a0808', alpha=0.9, zorder=2))

    for center, color, label in [(START, '#00ff88', 'INICIO'), (TARGET, '#ff4466', 'META')]:
        ax.add_patch(plt.Circle(center, RADIUS, color=color, alpha=0.4, zorder=5))
        ax.add_patch(plt.Circle(center, RADIUS, color=color, fill=False, lw=2, zorder=6))
        ax.text(center[0]+RADIUS+0.5, center[1]+RADIUS+0.5, label,
                color=color, fontsize=7, zorder=7)

# ─────────────────────────────────────────────
#  HELPERS — ruedas rectangulares rotadas
# ─────────────────────────────────────────────
def _make_wheel(ax, zorder=22):
    """Crea un rectángulo centrado en el origen que luego se transforma."""
    W = ROBOT_RADIUS * 0.55   # largo de la rueda (paralelo al eje del robot)
    H = ROBOT_RADIUS * 0.22   # ancho  de la rueda (grosor)
    rect = patches.Rectangle(
        (-W / 2, -H / 2), W, H,
        linewidth=0.8,
        edgecolor='#555555',
        facecolor='#1a1a1a',
        zorder=zorder
    )
    ax.add_patch(rect)
    return rect


def _update_wheel(rect, cx, cy, angle_rad):
    """
    Mueve y rota 'rect' para que su centro quede en (cx, cy)
    y su eje largo apunte en la dirección 'angle_rad'.
    """
    W = ROBOT_RADIUS * 0.55
    H = ROBOT_RADIUS * 0.22
    # La transformación: trasladar el origen al centro del eje del robot,
    # rotar, luego trasladar al centro de la rueda.
    t = (transforms.Affine2D()
         .translate(cx, cy)
         .rotate_around(cx, cy, angle_rad))
    # El rectángulo tiene su esquina inferior-izquierda en (-W/2, -H/2)
    # así que centrar en (cx, cy) ya está correcto con la traslación anterior.
    rect.set_transform(t + rect.axes.transData)


# ─────────────────────────────────────────────
#  ROBOT — clase para manejar estado
# ─────────────────────────────────────────────
class KheperaRobot:
    def __init__(self, ax, path, color, label):
        self.ax      = ax
        self.path    = path
        self.color   = color
        self.label   = label
        self.pos     = path[0].copy().astype(float)
        self.angle   = 0.0
        self.wp_idx  = 1
        self.done    = False
        self.t_start = None
        self.t_end   = None
        self.steps   = 0

        # ── Cuerpo ──
        self.body = plt.Circle(self.pos, ROBOT_RADIUS, color=color, zorder=20, alpha=0.95)
        ax.add_patch(self.body)

        # ── Flecha de orientación ──
        self.arrow, = ax.plot([], [], color='#111', lw=2.5, zorder=21)

        # ── Ruedas rectangulares (izquierda y derecha) ──
        self.wheels = [_make_wheel(ax, zorder=22), _make_wheel(ax, zorder=22)]
        self._reposition_wheels()   # colocarlas en la posición inicial

        # ── Rastro ──
        self.trail_x = [self.pos[0]]
        self.trail_y = [self.pos[1]]
        self.trail,  = ax.plot(self.trail_x, self.trail_y,
                               color=color, lw=1.8, alpha=0.55, zorder=7, ls='--')

        # ── Textos ──
        self.timer_text = ax.text(
            1, 97, '⏱ 0.00s', color=color,
            fontfamily='monospace', fontsize=9, zorder=30,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#111', edgecolor=color, alpha=0.8))

        self.wp_text = ax.text(
            1, 91, f'WP: 0/{len(path)-1}', color='#aaa',
            fontfamily='monospace', fontsize=8, zorder=30)

    # ── posicionado de ruedas ─────────────────

    def _reposition_wheels(self):
        """Actualiza la posición y rotación de ambas ruedas rectangulares."""
        offset = ROBOT_RADIUS * 0.9   # distancia lateral desde el centro
        for i, side in enumerate([-1, 1]):
            # Centro de cada rueda: perpendicular a la dirección de avance
            cx = self.pos[0] - offset * np.sin(self.angle) * side
            cy = self.pos[1] + offset * np.cos(self.angle) * side
            _update_wheel(self.wheels[i], cx, cy, self.angle)

    # ── API pública ──────────────────────────

    def start(self):
        self.t_start = time.time()

    def step(self):
        """Avanza un paso. Devuelve True si sigue en movimiento."""
        if self.done:
            return False

        if self.wp_idx >= len(self.path):
            self._finish()
            return False

        target = self.path[self.wp_idx]
        diff   = target - self.pos
        dist   = np.linalg.norm(diff)

        if dist < WAYPOINT_THRESH:
            self.wp_idx += 1
            return True

        self.angle = np.arctan2(diff[1], diff[0])
        step = min(ROBOT_SPEED, dist)
        self.pos = self.pos + step * diff / dist
        self.steps += 1

        # ── Actualizar gráficos ──
        self.body.center = self.pos

        al = ROBOT_RADIUS * 1.7
        self.arrow.set_data(
            [self.pos[0], self.pos[0] + al*np.cos(self.angle)],
            [self.pos[1], self.pos[1] + al*np.sin(self.angle)])

        self._reposition_wheels()

        self.trail_x.append(self.pos[0])
        self.trail_y.append(self.pos[1])
        self.trail.set_data(self.trail_x, self.trail_y)

        elapsed  = time.time() - self.t_start
        progress = int(100 * self.wp_idx / (len(self.path)-1))
        self.timer_text.set_text(f'⏱ {elapsed:.2f}s')
        self.wp_text.set_text(f'WP: {self.wp_idx}/{len(self.path)-1}  {progress}%')

        return True

    def _finish(self):
        self.done  = True
        self.t_end = time.time()
        elapsed    = self.t_end - self.t_start
        self.body.set_color('#00ff88')
        self.timer_text.set_text(f'✓ {elapsed:.2f}s')
        self.timer_text.set_color('#00ff88')
        self.wp_text.set_text(f'¡META! {len(self.path)-1} waypoints')
        self.wp_text.set_color('#00ff88')

    def get_time(self):
        if self.t_start and self.t_end:
            return self.t_end - self.t_start
        elif self.t_start:
            return time.time() - self.t_start
        return 0.0


# ─────────────────────────────────────────────
#  PANEL COMPARATIVO FINAL
# ─────────────────────────────────────────────
def show_comparison(fig, ax_cmp, path_rrt, path_rdp, robot_rrt, robot_rdp):
    ax_cmp.set_facecolor('#111')
    ax_cmp.set_xlim(0, 1); ax_cmp.set_ylim(0, 1)
    ax_cmp.axis('off')

    t_rrt = robot_rrt.get_time()
    t_rdp = robot_rdp.get_time()
    winner  = 'RRT*' if t_rrt < t_rdp else 'RDP'
    w_color = '#f0c040' if winner == 'RRT*' else '#ff44ff'
    reduction = 100*(1 - len(path_rdp)/len(path_rrt))

    lines = [
        ('━━━  COMPARATIVA FINAL  ━━━', '#ffffff', 13),
        ('', '#fff', 9),
        (f'  Waypoints RRT*  : {len(path_rrt):>4}', '#f0c040', 10),
        (f'  Waypoints RDP   : {len(path_rdp):>4}   ({reduction:.0f}% menos)', '#ff44ff', 10),
        ('', '#fff', 9),
        (f'  Tiempo RRT*     : {t_rrt:.2f} s', '#f0c040', 10),
        (f'  Tiempo RDP      : {t_rdp:.2f} s', '#ff44ff', 10),
        ('', '#fff', 9),
        (f'  Diferencia      : {abs(t_rrt-t_rdp):.2f} s', '#aaaaaa', 10),
        ('', '#fff', 9),
        (f'  🏆 Más rápido   : {winner}', w_color, 12),
    ]

    y = 0.92
    for text, color, size in lines:
        ax_cmp.text(0.08, y, text, color=color, fontfamily='monospace',
                    fontsize=size, transform=ax_cmp.transAxes, va='top')
        y -= 0.08

    bar_y = 0.12
    ax_cmp.barh([bar_y+0.06], [t_rrt], color='#f0c040', alpha=0.8,
                height=0.04, transform=ax_cmp.transAxes)
    ax_cmp.barh([bar_y],      [t_rdp], color='#ff44ff', alpha=0.8,
                height=0.04, transform=ax_cmp.transAxes)

    plt.pause(0.1)
    print("\n" + "="*52)
    print(f"  COMPARATIVA FINAL")
    print(f"  Waypoints RRT* : {len(path_rrt)}  |  RDP: {len(path_rdp)}  ({reduction:.0f}% menos)")
    print(f"  Tiempo RRT*    : {t_rrt:.2f}s")
    print(f"  Tiempo RDP     : {t_rdp:.2f}s")
    print(f"  🏆 Más rápido  : {winner}  (Δ {abs(t_rrt-t_rdp):.2f}s)")
    print("="*52)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    np.random.seed(42)

    fig = plt.figure(figsize=(18, 11), facecolor='#0a0a0a')
    gs  = gridspec.GridSpec(2, 2, height_ratios=[3, 1],
                            hspace=0.35, wspace=0.12,
                            left=0.04, right=0.96, top=0.93, bottom=0.04)

    ax_rrt = fig.add_subplot(gs[0, 0])
    ax_rdp = fig.add_subplot(gs[0, 1])
    ax_cmp = fig.add_subplot(gs[1, :])

    fig.suptitle('RRT*  vs  RDP  —  Carrera Khepera I',
                 color='#ffffff', fontfamily='monospace', fontsize=14, y=0.97)

    print("="*52)
    print("  FASE 1 — Calculando ruta con RRT*...")
    print("="*52)

    draw_environment(ax_rrt, 'Ruta RRT* original',   '#f0c040')
    draw_environment(ax_rdp, 'Ruta RDP simplificada', '#ff44ff')
    ax_cmp.set_facecolor('#111'); ax_cmp.axis('off')
    ax_cmp.text(0.5, 0.5, 'Esperando carrera...', color='#555',
                fontfamily='monospace', fontsize=12,
                ha='center', va='center', transform=ax_cmp.transAxes)

    plt.ion(); plt.show()

    path_rrt, rrt_cost = run_rrt_star(ax_rrt, max_iter=4000)

    for ax in [ax_rrt, ax_rdp]:
        ax.plot(path_rrt[:,0], path_rrt[:,1],
                color='#ffdd00', lw=1.5, alpha=0.5, zorder=8, ls=':')

    path_rdp  = rdp_simplify(path_rrt, RDP_EPSILON)
    reduction = 100*(1 - len(path_rdp)/len(path_rrt))
    print(f"\n  RDP: {len(path_rrt)} → {len(path_rdp)} puntos ({reduction:.0f}% reducción)")

    ax_rrt.plot(path_rrt[:,0], path_rrt[:,1],
                color='#f0c040', lw=2.5, zorder=9)
    ax_rrt.scatter(path_rrt[:,0], path_rrt[:,1], s=12, color='#ff8800', zorder=10)

    ax_rdp.plot(path_rdp[:,0], path_rdp[:,1],
                color='#ff44ff', lw=2.5, zorder=9)
    ax_rdp.scatter(path_rdp[:,0], path_rdp[:,1],
                   s=40, color='#ff00ff', zorder=10, marker='D')

    plt.pause(0.8)

    print("\n" + "="*52)
    print("  FASE 3 — ¡Carrera simultánea!")
    print("="*52)

    robot_rrt = KheperaRobot(ax_rrt, path_rrt, '#f0c040', 'RRT*')
    robot_rdp = KheperaRobot(ax_rdp, path_rdp, '#ff44ff', 'RDP')

    for i in range(3, 0, -1):
        fig.suptitle(f'¡Saliendo en {i}...',
                     color='#ff4444', fontfamily='monospace', fontsize=18, y=0.97)
        plt.pause(0.6)

    fig.suptitle('RRT*  vs  RDP  —  ¡EN MARCHA!',
                 color='#00ff88', fontfamily='monospace', fontsize=14, y=0.97)

    t_race_start = time.time()
    robot_rrt.start()
    robot_rdp.start()

    while not (robot_rrt.done and robot_rdp.done):
        robot_rrt.step()
        robot_rdp.step()

        elapsed_race = time.time() - t_race_start
        if not robot_rrt.done and not robot_rdp.done:
            fig.suptitle(f'RRT*  vs  RDP  —  ⏱ {elapsed_race:.1f}s',
                         color='#ffffff', fontfamily='monospace', fontsize=13, y=0.97)
        elif robot_rrt.done and not robot_rdp.done:
            fig.suptitle('🏆 RRT* llegó primero — RDP en camino...',
                         color='#f0c040', fontfamily='monospace', fontsize=12, y=0.97)
        elif robot_rdp.done and not robot_rrt.done:
            fig.suptitle('🏆 RDP llegó primero — RRT* en camino...',
                         color='#ff44ff', fontfamily='monospace', fontsize=12, y=0.97)

        plt.pause(0.01)

    plt.pause(0.3)
    show_comparison(fig, ax_cmp, path_rrt, path_rdp, robot_rrt, robot_rdp)

    t_rrt   = robot_rrt.get_time()
    t_rdp   = robot_rdp.get_time()
    winner  = 'RRT*' if t_rrt < t_rdp else 'RDP'
    w_color = '#f0c040' if winner == 'RRT*' else '#ff44ff'
    fig.suptitle(f'🏆  {winner} ganó la carrera  🏆',
                 color=w_color, fontfamily='monospace', fontsize=15, y=0.97)

    plt.ioff()
    plt.savefig('carrera_khepera.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\n  Figura guardada: carrera_khepera.png")
    plt.show()


if __name__ == '__main__':
    main()