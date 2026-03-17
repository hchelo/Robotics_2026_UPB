import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import os

# --------------------------------
# PARAMETROS
# --------------------------------

DT        = 0.1
STEPS     = 1000
MAX_RANGE = 30.0

R = np.diag([0.5, np.deg2rad(5.0)])**2
Q_odom = np.diag([0.3, np.deg2rad(4.0)])**2

np.random.seed(7)

N_LM      = 40
landmarks = np.random.uniform(-95, 95, (N_LM, 2))

# --------------------------------
# ESTADO INICIAL
# --------------------------------

xTrue = np.array([[0.0], [0.0], [0.0]])
xOdom = xTrue.copy()
xEst  = xTrue.copy()
PEst  = np.eye(3) * 0.01

lm_map = {}

hxTrue = [xTrue[:2].flatten().copy()]
hxOdom = [xOdom[:2].flatten().copy()]
hxEst  = [xEst[:2].flatten().copy()]

obs_count = np.zeros(N_LM, dtype=int)


# --------------------------------
# TRAYECTORIA POR WAYPOINTS
# --------------------------------

WAYPOINTS = np.array([
    [  0.0,   0.0],
    [ 75.0,  0.0],
    [ 75.0,  75.0],
    [ -75.0, 75.0],
    [ -75.0, -75.0],
    [ -75.0, -75.0],
    [ 75.0, -75.0],
    [ -70.0, 70.0],
])

def get_control_waypoints(x, wp_idx):
    if wp_idx >= len(WAYPOINTS):
        return np.array([0.0, 0.0]), wp_idx

    target = WAYPOINTS[wp_idx]
    dx = target[0] - x[0, 0]
    dy = target[1] - x[1, 0]
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 5.0:
        wp_idx += 1
        if wp_idx >= len(WAYPOINTS):
            return np.array([0.0, 0.0]), wp_idx
        target = WAYPOINTS[wp_idx]
        dx = target[0] - x[0, 0]
        dy = target[1] - x[1, 0]
        dist = np.sqrt(dx**2 + dy**2) + 1e-9

    angle_to_target = np.arctan2(dy, dx)
    angle_err = angle_to_target - x[2, 0]
    angle_err = (angle_err + np.pi) % (2 * np.pi) - np.pi

    v = min(10.0, dist * 0.3)
    w = 2.5 * angle_err
    w = np.clip(w, -1.5, 1.5)

    return np.array([v, w]), wp_idx


# --------------------------------
# MODELO DE MOVIMIENTO
# --------------------------------

def motion_model(x, u, add_noise=False):
    v, w = u
    x = x.copy()
    if add_noise:
        v += np.random.randn() * np.sqrt(Q_odom[0, 0])
        w += np.random.randn() * np.sqrt(Q_odom[1, 1])
    x[0, 0] += v * np.cos(x[2, 0]) * DT
    x[1, 0] += v * np.sin(x[2, 0]) * DT
    x[2, 0] += w * DT
    x[2, 0]  = (x[2, 0] + np.pi) % (2 * np.pi) - np.pi
    return x


# --------------------------------
# SENSOR
# --------------------------------

def observe(x, landmarks):
    z        = []
    detected = []
    indices  = []
    for i, lm in enumerate(landmarks):
        dx = lm[0] - x[0, 0]
        dy = lm[1] - x[1, 0]
        d  = np.sqrt(dx**2 + dy**2)
        if d < MAX_RANGE:
            d_n   = d + np.random.randn() * np.sqrt(R[0, 0])
            ang_n = np.arctan2(dy, dx) - x[2, 0] + np.random.randn() * np.sqrt(R[1, 1])
            ang_n = (ang_n + np.pi) % (2 * np.pi) - np.pi
            z.append([d_n, ang_n, i])
            detected.append(lm)
            indices.append(i)
    return z, detected, indices


# --------------------------------
# ELIPSE DE COVARIANZA
# --------------------------------

def cov_ellipse(ax, mean, cov2x2, n_std=3, **kwargs):
    vals, vecs = np.linalg.eigh(cov2x2)
    vals       = np.maximum(vals, 1e-9)
    angle      = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h       = 2 * n_std * np.sqrt(vals)
    ell        = Ellipse(xy=mean, width=w, height=h, angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell


# --------------------------------
# EKF SLAM
# --------------------------------

def ekf_slam(xEst, PEst, lm_map, u, z):
    n = len(xEst)

    v, w = u
    th   = xEst[2, 0]

    xEst[0:3] = motion_model(xEst[0:3], u, add_noise=False)

    Jx        = np.eye(3)
    Jx[0, 2]  = -v * np.sin(th) * DT
    Jx[1, 2]  =  v * np.cos(th) * DT

    G            = np.eye(n)
    G[0:3, 0:3]  = Jx

    Qfull             = np.zeros((n, n))
    Qfull[0, 0]       = Q_odom[0, 0]
    Qfull[1, 1]       = Q_odom[0, 0]
    Qfull[2, 2]       = Q_odom[1, 1]

    PEst = G @ PEst @ G.T + Qfull

    for obs in z:
        r, b, gid = obs

        if gid not in lm_map:
            j    = len(lm_map)
            lm_map[gid] = j

            lx   = xEst[0, 0] + r * np.cos(b + xEst[2, 0])
            ly   = xEst[1, 0] + r * np.sin(b + xEst[2, 0])

            xEst = np.vstack((xEst, [[lx], [ly]]))

            s    = PEst.shape[0]
            Pnew = np.zeros((s + 2, s + 2))
            Pnew[:s, :s]   = PEst
            Pnew[-2:, -2:] = np.eye(2) * 10.0
            PEst = Pnew

        j    = lm_map[gid]
        row  = 3 + 2 * j

        lx   = xEst[row,     0]
        ly   = xEst[row + 1, 0]

        dx_e = lx - xEst[0, 0]
        dy_e = ly - xEst[1, 0]
        d_e  = np.sqrt(dx_e**2 + dy_e**2) + 1e-9
        b_e  = np.arctan2(dy_e, dx_e) - xEst[2, 0]
        b_e  = (b_e + np.pi) % (2 * np.pi) - np.pi

        inn    = np.array([r - d_e, b - b_e])
        inn[1] = (inn[1] + np.pi) % (2 * np.pi) - np.pi

        H              = np.zeros((2, len(xEst)))
        H[0, 0]        = -dx_e / d_e
        H[0, 1]        = -dy_e / d_e
        H[1, 0]        =  dy_e / d_e**2
        H[1, 1]        = -dx_e / d_e**2
        H[1, 2]        = -1.0
        H[0, row]      =  dx_e / d_e
        H[0, row + 1]  =  dy_e / d_e
        H[1, row]      = -dy_e / d_e**2
        H[1, row + 1]  =  dx_e / d_e**2

        S    = H @ PEst @ H.T + R
        K    = PEst @ H.T @ np.linalg.inv(S)

        xEst = xEst + K @ inn.reshape(-1, 1)
        PEst = (np.eye(len(xEst)) - K @ H) @ PEst
        xEst[2, 0] = (xEst[2, 0] + np.pi) % (2 * np.pi) - np.pi

    return xEst, PEst, lm_map


# --------------------------------
# DIBUJO EN TIEMPO REAL
# --------------------------------

def draw(ax, step, landmarks, xTrue, xOdom, xEst, PEst, lm_map,
         hxTrue_np, hxOdom_np, hxEst_np, detected):

    ax.cla()
    ax.set_facecolor("white")
    lim = 110
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, color="#cccccc", linewidth=0.5, linestyle="--")

    ax.plot(WAYPOINTS[:, 0], WAYPOINTS[:, 1],
            's', color='purple', markersize=9, zorder=10, label='Waypoints')
    ax.plot(WAYPOINTS[:, 0], WAYPOINTS[:, 1],
            '--', color='purple', linewidth=0.8, alpha=0.5, zorder=9)
    for wi, wp in enumerate(WAYPOINTS):
        ax.annotate(f'WP{wi}', xy=wp, xytext=(4, 4),
                    textcoords='offset points', fontsize=7,
                    color='purple', zorder=11)

    ax.plot(landmarks[:, 0], landmarks[:, 1],
            '+k', markersize=7, markeredgewidth=1.2, zorder=3)

    for lm in detected:
        ax.plot([xTrue[0, 0], lm[0]], [xTrue[1, 0], lm[1]],
                color="#aaaaaa", linewidth=0.6, alpha=0.4, zorder=1)

    if len(hxTrue_np) > 1:
        ax.plot(hxTrue_np[:, 0], hxTrue_np[:, 1],
                color="black", linewidth=1.8, zorder=4)
    if len(hxOdom_np) > 1:
        ax.plot(hxOdom_np[:, 0], hxOdom_np[:, 1],
                color="blue", linewidth=1.2, zorder=4)
    if len(hxEst_np) > 1:
        ax.plot(hxEst_np[:, 0], hxEst_np[:, 1],
                color="green", linewidth=1.4, zorder=5)

    ax.plot(xTrue[0, 0], xTrue[1, 0], 'ok', markersize=7, zorder=8)

    for gid, j in lm_map.items():
        row = 3 + 2 * j
        mx, my = xEst[row, 0], xEst[row + 1, 0]
        ax.plot(mx, my, '.r', markersize=7, zorder=6)
        P_lm = PEst[row:row+2, row:row+2]
        cov_ellipse(ax, (mx, my), P_lm, n_std=3,
                    edgecolor="red", facecolor="none",
                    linewidth=1.0, zorder=5)

    nLM = len(lm_map)
    ax.set_title(f"EKF SLAM  |  step {step+1}/{STEPS}  |  landmarks: {nLM}/{N_LM}",
                 fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    handles = [
        mpatches.Patch(color="black", label="True path"),
        mpatches.Patch(color="blue",  label="Odometry"),
        mpatches.Patch(color="green", label="EKF estimate"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)


# --------------------------------
# MAPA FINAL
# --------------------------------

def save_final_map(landmarks, xEst, PEst, lm_map,
                   hxTrue_np, hxOdom_np, hxEst_np,
                   filename="ekf_slam_map.png"):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("white")
    lim = 110
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, color="#eeeeee", linewidth=0.5, linestyle="--")

    ax.plot(landmarks[:, 0], landmarks[:, 1],
            '+k', markersize=10, markeredgewidth=1.8, zorder=3,
            label="Real landmarks")

    for gid, j in lm_map.items():
        row    = 3 + 2 * j
        mx, my = xEst[row, 0], xEst[row + 1, 0]
        ax.plot(mx, my, '.r', markersize=9, zorder=6)
        P_lm = PEst[row:row+2, row:row+2]
        cov_ellipse(ax, (mx, my), P_lm, n_std=3,
                    edgecolor="red", facecolor="none",
                    linewidth=1.3, zorder=5)

    ax.set_title(f"EKF SLAM — Landmark Map  ({len(lm_map)}/{N_LM} detected)",
                 fontsize=13)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    handles = [
        mpatches.Patch(color="black", label="Real landmarks (+)"),
        mpatches.Patch(color="red",   label="Estimated landmarks + 3σ ellipse"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[ok] Mapa guardado en: {out}")
    plt.show(block=False)
    plt.pause(0.1)


# --------------------------------
# MATRIZ DE COVARIANZA (NUEVA)
# --------------------------------

def save_covariance_matrix(xEst, PEst, lm_map, filename="ekf_slam_cov.png"):
    """
    Genera y guarda tres visualizaciones de la matriz de covarianza PEst:
      1. Matriz completa en escala logarítmica con etiquetas
      2. Matriz de correlacion normalizada
      3. Zoom del bloque robot (3x3) con valores numericos
    """
    n = len(xEst)
    nLM = len(lm_map)

    # --- Etiquetas de estados ---
    labels = ['x_r', 'y_r', 'θ_r']
    for j in range(nLM):
        labels += [f'lm{j}x', f'lm{j}y']

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(f"Matriz de Covarianza EKF SLAM  —  estado: {n}×{n}  ({nLM} landmarks)",
                 fontsize=14, fontweight='bold', y=1.01)

    # ================================================================
    # PANEL 1: Matriz completa — escala log
    # ================================================================
    ax1 = fig.add_subplot(1, 3, 1)

    P_log = np.log10(np.abs(PEst) + 1e-12)

    im1 = ax1.imshow(P_log, cmap="plasma", aspect="auto", interpolation="nearest")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("log₁₀|P_ij|", fontsize=9)

    # Línea separadora robot / landmarks
    ax1.axhline(2.5, color='cyan', linewidth=1.5, linestyle='--', alpha=0.8)
    ax1.axvline(2.5, color='cyan', linewidth=1.5, linestyle='--', alpha=0.8)

    # Etiquetas de ejes (solo si el estado no es demasiado grande)
    tick_step = max(1, n // 20)   # mostrar hasta ~20 etiquetas
    ticks = list(range(0, n, tick_step))
    tick_labels = [labels[i] if i < len(labels) else str(i) for i in ticks]

    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(tick_labels, fontsize=6)

    # Recuadros de bloques (robot-robot, robot-LM, LM-LM)
    from matplotlib.patches import Rectangle
    ax1.add_patch(Rectangle((-0.5, -0.5), 3, 3,
                             linewidth=2, edgecolor='cyan', facecolor='none'))
    if nLM > 0:
        lm_size = 2 * nLM
        ax1.add_patch(Rectangle((2.5, 2.5), lm_size, lm_size,
                                 linewidth=1.5, edgecolor='lime', facecolor='none'))
        ax1.add_patch(Rectangle((2.5, -0.5), lm_size, 3,
                                 linewidth=1.5, edgecolor='orange', facecolor='none'))
        ax1.add_patch(Rectangle((-0.5, 2.5), 3, lm_size,
                                 linewidth=1.5, edgecolor='orange', facecolor='none'))

    ax1.set_title("Covarianza completa\n(escala log₁₀)", fontsize=10)
    ax1.set_xlabel("Estado j", fontsize=9)
    ax1.set_ylabel("Estado i", fontsize=9)

    # Leyenda de bloques
    legend_patches = [
        mpatches.Patch(edgecolor='cyan',   facecolor='none', label='P_rr  (robot-robot)', linewidth=2),
        mpatches.Patch(edgecolor='lime',   facecolor='none', label='P_ll  (LM-LM)',       linewidth=1.5),
        mpatches.Patch(edgecolor='orange', facecolor='none', label='P_rl  (robot-LM)',    linewidth=1.5),
    ]
    ax1.legend(handles=legend_patches, loc='lower right', fontsize=7,
               facecolor='#222222', labelcolor='white', framealpha=0.85)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[ok] Covarianza guardada en: {out}")
    plt.show(block=False)
    plt.pause(0.1)


# --------------------------------
# SIMULACION PRINCIPAL
# --------------------------------

wp_idx = 0

fig, ax = plt.subplots(figsize=(9, 9))

for step in range(STEPS):

    u, wp_idx = get_control_waypoints(xTrue, wp_idx)

    xTrue = motion_model(xTrue, u, add_noise=True)
    xOdom = motion_model(xOdom, u, add_noise=True)

    z, detected, det_idx = observe(xTrue, landmarks)
    for idx in det_idx:
        obs_count[idx] += 1

    xEst, PEst, lm_map = ekf_slam(xEst, PEst, lm_map, u, z)

    hxTrue.append(xTrue[:2].flatten().copy())
    hxOdom.append(xOdom[:2].flatten().copy())
    hxEst.append(xEst[:2].flatten().copy())

    hxTrue_np = np.array(hxTrue)
    hxOdom_np = np.array(hxOdom)
    hxEst_np  = np.array(hxEst)

    draw(ax, step, landmarks, xTrue, xOdom, xEst, PEst, lm_map,
         hxTrue_np, hxOdom_np, hxEst_np, detected)

    plt.pause(0.01)

plt.close()

# --- Mapa final ---
save_final_map(landmarks, xEst, PEst, lm_map,
               np.array(hxTrue), np.array(hxOdom), np.array(hxEst),
               filename="ekf_slam_map.png")

# --- Matriz de covarianza ---
save_covariance_matrix(xEst, PEst, lm_map,
                       filename="ekf_slam_cov.png")