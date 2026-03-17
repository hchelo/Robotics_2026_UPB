import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_dilation
from skimage.morphology import ball
from PIL import Image
import io, os

# ════════════════════════════════════════════════════════════════
#  LEER MAPA
# ════════════════════════════════════════════════════════════════
ruta_mapa = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Mapa.txt')
mapa_raw = []
with open(ruta_mapa) as f:
    for linea in f:
        linea = linea.strip()
        if not linea: continue
        mapa_raw.append([float(v) for v in linea.replace('\t',' ').split()])

M_raw = np.array(mapa_raw, dtype=float)
M     = M_raw.T.copy()
FILAS, COLS = M_raw.shape
ANCHO, ALTO = COLS, FILAS

pos_i = np.argwhere(M_raw ==  2)[0]
pos_g = np.argwhere(M_raw == -1)[0]
xi, yi = int(pos_i[1]), int(pos_i[0])
xg, yg = int(pos_g[1]), int(pos_g[0])
M[M == 2] = 0;  M[M == -1] = 0

print(f"Mapa {ANCHO}x{ALTO}  |  Inicio ({xi},{yi})  |  Meta ({xg},{yg})")

# ════════════════════════════════════════════════════════════════
#  OBSTACULOS DILATADOS
# ════════════════════════════════════════════════════════════════
struct  = ball(2)[2]
M_dilat = binary_dilation(M == 1, structure=struct).astype(float)
M_dilat[xi, yi] = 0;  M_dilat[xg, yg] = 0

# ════════════════════════════════════════════════════════════════
#  CAMPOS POTENCIALES
# ════════════════════════════════════════════════════════════════
K  = 3.0
XX, YY = np.meshgrid(np.arange(ANCHO), np.arange(ALTO), indexing='ij')

d_meta = np.sqrt((XX-xg)**2 + (YY-yg)**2).clip(1e-6)
d_ini  = np.sqrt((XX-xi)**2 + (YY-yi)**2).clip(1e-6)

UA = 0.5 * K * d_meta
UR = np.where(M_dilat == 1, 10*K + 1/d_ini, 0.0)
U  = UA + UR

gx_f, gy_f = np.gradient(-U)

# ════════════════════════════════════════════════════════════════
#  DESCENSO DE GRADIENTE
# ════════════════════════════════════════════════════════════════
DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
px, py = [xi], [yi]
cx, cy = xi, yi
for _ in range(50000):
    if cx == xg and cy == yg: break
    bv, nx, ny = U[cx,cy], cx, cy
    for dx, dy in DIRS:
        ni, nj = cx+dx, cy+dy
        if 0<=ni<ANCHO and 0<=nj<ALTO and U[ni,nj]<bv:
            bv, nx, ny = U[ni,nj], ni, nj
    if nx==cx and ny==cy: break
    cx, cy = nx, ny
    px.append(cx); py.append(cy)

path_x, path_y = np.array(px), np.array(py)
N = len(path_x)
print(f"Path: {N} puntos")

# ════════════════════════════════════════════════════════════════
#  WAYPOINTS
# ════════════════════════════════════════════════════════════════
wpts = [0]
for k in range(1, N-1):
    va = np.array([path_x[k]-path_x[k-1], path_y[k]-path_y[k-1]], float)
    vs = np.array([path_x[k+1]-path_x[k], path_y[k+1]-path_y[k]], float)
    na, ns = np.linalg.norm(va), np.linalg.norm(vs)
    if na>1e-9 and ns>1e-9:
        cos = np.clip(np.dot(va,vs)/(na*ns), -1, 1)
        if np.arccos(cos) >= np.deg2rad(20): wpts.append(k)
wpts.append(N-1)
wf = [wpts[0]]
for idx in wpts[1:]:
    if np.hypot(path_x[idx]-path_x[wf[-1]], path_y[idx]-path_y[wf[-1]]) >= 3:
        wf.append(idx)
WP_x, WP_y = path_x[wf], path_y[wf]

# ════════════════════════════════════════════════════════════════
#  COLORMAPS CUSTOM
# ════════════════════════════════════════════════════════════════
cmap_field = LinearSegmentedColormap.from_list('field',
    ['#0a0014','#2d1b69','#11998e','#43cea2','#f8ffae'], N=512)
cmap_ua = LinearSegmentedColormap.from_list('ua',
    ['#0a001a','#1a0050','#4776e6','#8e54e9','#c9ffbf'], N=256)
cmap_ur = LinearSegmentedColormap.from_list('ur',
    ['#0a0000','#3d0000','#c0392b','#e74c3c','#ff9a44'], N=256)

# ════════════════════════════════════════════════════════════════
#  PALETA GITHUB DARK
# ════════════════════════════════════════════════════════════════
BG    = '#0d1117'
PANEL = '#161b22'
BORDER= '#30363d'
ACC1  = '#58a6ff'
ACC2  = '#3fb950'
ACC3  = '#ff7b72'
ACC4  = '#ffa657'
TEXT  = '#e6edf3'
MUTED = '#8b949e'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   PANEL,
    'axes.edgecolor':   BORDER,
    'axes.labelcolor':  MUTED,
    'xtick.color':      MUTED,
    'ytick.color':      MUTED,
    'text.color':       TEXT,
    'font.family':      'monospace',
    'grid.color':       BORDER,
    'grid.linewidth':   0.4,
})

# ════════════════════════════════════════════════════════════════
#  FIGURA
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 7), dpi=100, facecolor=BG)
gs  = GridSpec(3, 3, figure=fig,
               width_ratios=[2.2, 1, 0.85],
               height_ratios=[1, 1, 1],
               hspace=0.08, wspace=0.28,
               left=0.04, right=0.98,
               top=0.91, bottom=0.07)

ax_main = fig.add_subplot(gs[:, 0])
ax_ua   = fig.add_subplot(gs[0, 1])
ax_ur   = fig.add_subplot(gs[1, 1])
ax_tot  = fig.add_subplot(gs[2, 1])
ax_info = fig.add_subplot(gs[:, 2])

for ax in [ax_main, ax_ua, ax_ur, ax_tot, ax_info]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER); sp.set_linewidth(0.8)

ext  = [0, ANCHO, 0, ALTO]
U_c  = np.clip(U,  0, 150)
UA_c = np.clip(UA, 0, UA.max())
UR_c = np.clip(UR, 0, 50)

# ── Panel principal ───────────────────────────────────────────
im_main = ax_main.imshow(U_c.T, origin='lower', extent=ext,
                          cmap=cmap_field, vmin=0, vmax=150,
                          interpolation='bilinear', zorder=0)

obs = np.zeros((*M.T.shape, 4))
obs[M.T == 1] = [0.04, 0.05, 0.07, 1.0]
ax_main.imshow(obs, origin='lower', extent=ext, interpolation='nearest', zorder=1)

dil = np.zeros((*M_dilat.T.shape, 4))
dil[(M_dilat.T == 1) & (M.T == 0)] = [1, 0.27, 0.27, 0.12]
ax_main.imshow(dil, origin='lower', extent=ext, interpolation='nearest', zorder=2)

step_q = max(2, ANCHO//16)
Xs = np.arange(step_q, ANCHO-step_q, step_q)
Ys = np.arange(step_q, ALTO-step_q,  step_q)
Xg_, Yg_ = np.meshgrid(Xs, Ys, indexing='ij')
mask_ok   = M_dilat[Xs[:,None], Ys[None,:]] == 0
ax_main.quiver(
    Xg_[mask_ok], Yg_[mask_ok],
    gx_f[Xs[:,None], Ys[None,:]][mask_ok],
    gy_f[Xs[:,None], Ys[None,:]][mask_ok],
    color='white', alpha=0.12, scale=70, width=0.002, zorder=3)

ax_main.plot(path_x, path_y, color=TEXT, lw=0.5, alpha=0.18, zorder=4)
ax_main.scatter(WP_x, WP_y, s=55, c=ACC4, zorder=6,
                edgecolors=BG, lw=1.2, label='Waypoints')
ax_main.plot(xi, yi, 'o', ms=10, color=ACC2, zorder=8,
             markeredgecolor=BG, markeredgewidth=1.5, label='Inicio')
ax_main.plot(xg, yg, '*', ms=14, color=ACC3, zorder=8,
             markeredgecolor=BG, markeredgewidth=1.0, label='Meta')

cbar = plt.colorbar(im_main, ax=ax_main, fraction=0.025, pad=0.008)
cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
cbar.outline.set_edgecolor(BORDER)
cbar.set_label('U total', color=MUTED, fontsize=8)

ax_main.set_xlim(0, ANCHO); ax_main.set_ylim(0, ALTO)
ax_main.set_aspect('equal')
ax_main.set_xlabel('x [celdas]', fontsize=8)
ax_main.set_ylabel('y [celdas]', fontsize=8)
ax_main.tick_params(labelsize=7)
ax_main.grid(True, alpha=0.3)
ax_main.legend(loc='lower right', fontsize=7,
               facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

traj_line, = ax_main.plot([], [], color=ACC2, lw=2.0, zorder=9,
                           solid_capstyle='round')
attr_line, = ax_main.plot([], [], '--', color=ACC1, lw=1.0,
                           alpha=0.65, zorder=9)
robot_body = plt.Circle((xi, yi), 0.7, color=ACC4, zorder=10,
                         ec=BG, lw=1.5)
ax_main.add_patch(robot_body)
rep_lines  = []
pot_box    = ax_main.text(
    0.02, 0.98, '', transform=ax_main.transAxes,
    fontsize=8, color=TEXT, va='top', ha='left', family='monospace',
    bbox=dict(boxstyle='round,pad=0.4', fc=PANEL, ec=BORDER,
              lw=0.8, alpha=0.92))

# ── Mini paneles ─────────────────────────────────────────────
def setup_mini(ax, data, cmap, title, tcol):
    ax.imshow(data.T, origin='lower', extent=ext,
              cmap=cmap, interpolation='bilinear')
    ax.imshow(obs, origin='lower', extent=ext, interpolation='nearest')
    ax.plot(xi, yi, 'o', ms=5, color=ACC2, zorder=5, markeredgewidth=0)
    ax.plot(xg, yg, '*', ms=7, color=ACC3, zorder=5, markeredgewidth=0)
    ax.set_xlim(0, ANCHO); ax.set_ylim(0, ALTO)
    ax.set_aspect('equal'); ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8, color=tcol, pad=3)
    dot, = ax.plot([], [], 'o', color=ACC4, ms=5, zorder=6,
                   markeredgecolor=BG, markeredgewidth=0.8)
    return dot

dot_ua  = setup_mini(ax_ua,  UA_c, cmap_ua,    'U_attr',  ACC1)
dot_ur  = setup_mini(ax_ur,  UR_c, cmap_ur,    'U_rep',   ACC3)
dot_tot = setup_mini(ax_tot, U_c,  cmap_field, 'U total', ACC4)

rng_circ = plt.Circle((xi, yi), 8, color=ACC3,
                       fill=False, ls='--', lw=0.8, alpha=0.5, zorder=5)
ax_ur.add_patch(rng_circ)

# ── Panel info ────────────────────────────────────────────────
ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
ax_info.axis('off')
ax_info.set_title('Estado', fontsize=9, color=TEXT, pad=6)

theta = np.linspace(0, 2*np.pi, 200)
ax_info.plot(0.5+0.3*np.cos(theta), 0.78+0.14*np.sin(theta),
             color=BORDER, lw=4, solid_capstyle='round')
prog_arc, = ax_info.plot([], [], color=ACC2, lw=4,
                          solid_capstyle='round', zorder=5)
pct_text  = ax_info.text(0.5, 0.78, '0%', ha='center', va='center',
                          fontsize=13, color=TEXT, fontweight='bold')

BAR_Y    = [0.54, 0.43, 0.32]
BAR_COLS = [ACC1, ACC3, ACC4]
BAR_LABS = ['Ua', 'Ur', 'U ']
bar_fills, val_texts = [], []
for y, col, lbl in zip(BAR_Y, BAR_COLS, BAR_LABS):
    ax_info.add_patch(mpatches.FancyBboxPatch(
        (0.05, y), 0.9, 0.07, boxstyle='round,pad=0.01',
        fc=BORDER, ec='none', zorder=2))
    fill = mpatches.FancyBboxPatch(
        (0.055, y+0.005), 0.0, 0.06, boxstyle='square,pad=0.0',
        fc=col, ec='none', zorder=3)
    ax_info.add_patch(fill)
    bar_fills.append(fill)
    ax_info.text(0.05, y+0.035, lbl, ha='left', va='center',
                 fontsize=8, color=col, fontweight='bold', zorder=4)
    val_texts.append(
        ax_info.text(0.95, y+0.035, '0.00', ha='right', va='center',
                     fontsize=8, color=col, family='monospace', zorder=4))

step_text = ax_info.text(0.5, 0.20, 'Paso 0', ha='center',
                          fontsize=9, color=TEXT, fontweight='bold')
pos_text  = ax_info.text(0.5, 0.11, '(0, 0)', ha='center',
                          fontsize=8, color=MUTED, family='monospace')

UA_MAX = max(float(UA[path_x, path_y].max()), 1e-6)
UR_MAX = max(float(UR[path_x, path_y].max()), 1e-6)
U_MAX  = max(float(U [path_x, path_y].max()), 1e-6)

fig.text(0.5, 0.965,
         'Path Planning — Potential Fields  |  Khepera Robot',
         ha='center', fontsize=11, color=TEXT, fontweight='bold')
fig.text(0.5, 0.945,
         'Gradient descent  ·  Obstacle dilation  ·  Waypoint detection',
         ha='center', fontsize=8, color=MUTED)

# ════════════════════════════════════════════════════════════════
#  FUNCION DE ANIMACION
# ════════════════════════════════════════════════════════════════
SKIP = max(1, N // 350)

def update(fi):
    idx = min(fi * SKIP, N - 1)
    rx, ry = path_x[idx], path_y[idx]
    pct = idx / max(N-1, 1)

    traj_line.set_data(path_x[:idx+1], path_y[:idx+1])
    robot_body.set_center((rx, ry))
    attr_line.set_data([rx, xg], [ry, yg])

    for ln in rep_lines: ln.remove()
    rep_lines.clear()
    nearby = sorted(
        [(np.hypot(rx-wx, ry-wy), wx, wy)
         for wx in range(max(0,rx-12), min(ANCHO,rx+13))
         for wy in range(max(0,ry-12), min(ALTO,ry+13))
         if M[wx,wy]==1 and np.hypot(rx-wx,ry-wy)<12]
    )[:5]
    for d, wx, wy in nearby:
        ln, = ax_main.plot([wx,rx],[wy,ry], color=ACC3,
                           lw=0.9, alpha=0.65*(1-d/12), zorder=8)
        rep_lines.append(ln)

    ua_v = UA[rx,ry]; ur_v = UR[rx,ry]; u_v = U[rx,ry]
    pot_box.set_text(
        f' Ua  = {ua_v:6.2f}\n'
        f' Ur  = {ur_v:6.2f}\n'
        f' U   = {u_v:6.2f} ')

    dot_ua.set_data([rx],[ry])
    dot_ur.set_data([rx],[ry])
    dot_tot.set_data([rx],[ry])
    rng_circ.set_center((rx, ry))

    t_end = -np.pi/2 + 2*np.pi*pct
    arc_t = np.linspace(-np.pi/2, t_end, max(2, int(200*pct)+1))
    prog_arc.set_data(0.5+0.3*np.cos(arc_t), 0.78+0.14*np.sin(arc_t))
    pct_text.set_text(f'{int(pct*100)}%')

    vals_raw = [ua_v, ur_v, u_v]
    maxs     = [UA_MAX, UR_MAX, U_MAX]
    for i, (fill, vt) in enumerate(zip(bar_fills, val_texts)):
        fill.set_width(min(0.89 * vals_raw[i]/maxs[i], 0.88))
        vt.set_text(f'{vals_raw[i]:.2f}')

    step_text.set_text(f'Paso {idx} / {N-1}')
    pos_text.set_text(f'({rx}, {ry})')

    return ([traj_line, robot_body, attr_line,
             dot_ua, dot_ur, dot_tot,
             rng_circ, prog_arc, pct_text,
             pot_box, step_text, pos_text]
            + bar_fills + val_texts + rep_lines)

N_FRAMES = (N + SKIP - 1) // SKIP

ani = animation.FuncAnimation(
    fig, update, frames=N_FRAMES,
    interval=40, blit=False, repeat=False)

# ════════════════════════════════════════════════════════════════
#  GUARDAR GIF — savefig → BytesIO → PIL
#  SIN buffer_rgba, 100% portable en Windows
# ════════════════════════════════════════════════════════════════
def guardar_gif(out='path_potencial.gif', fps=18):
    print(f"\nGenerando GIF ({N_FRAMES} frames, {fps} fps)...")
    frames_pil = []

    for fi in range(N_FRAMES):
        update(fi)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100,
                    facecolor=BG, bbox_inches=None)
        buf.seek(0)
        frames_pil.append(Image.open(buf).copy())
        buf.close()
        if (fi+1) % 50 == 0:
            print(f'  {fi+1}/{N_FRAMES} frames...')

    print('Cuantizando y guardando...')
    duration_ms = int(1000 / fps)

    # Convertir RGBA -> RGB (MEDIANCUT no soporta RGBA)
    frames_rgb = [f.convert('RGB') for f in frames_pil]

    # Paleta global de alta calidad
    first = frames_rgb[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    rest  = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
             for f in frames_rgb[1:]]

    first.save(
        out,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2)

    size_mb = os.path.getsize(out) / 1e6
    print(f'GIF guardado: {out}  ({size_mb:.1f} MB, {N_FRAMES} frames)')

plt.show()
guardar_gif()