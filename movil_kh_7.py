import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# ================================================================
#  PARÁMETROS REALES DEL ROBOT KHEPERA
# ================================================================
r        = 0.008     # radio de rueda [m]
l        = 0.054     # distancia entre ruedas [m]
R_ROBOT  = 0.0275    # radio robot [m] → diámetro 55 mm
R_RUEDA  = 0.005     # radio visual rueda [m]  (grosor)
L_RUEDA  = 0.010     # largo visual rueda [m]  (no excede R_ROBOT=0.0275)
dt       = 0.05      # paso de tiempo [s]

PULSOS_REV   = 600
MM_POR_PULSO = (2 * np.pi * r) / PULSOS_REV

# Velocidad más lenta para animación más realista
V_LIN  = 0.03    # m/s avance  (más lento = más realista)
V_ROT  = 0.25    # rad/s giro

# ================================================================
#  CONVERSIONES
# ================================================================
def metros_a_pulsos(metros):
    return abs(metros) / MM_POR_PULSO

def angulo_a_pulsos(angulo_rad):
    arco = (l / 2) * abs(angulo_rad)
    return arco / MM_POR_PULSO

# ================================================================
#  GEOMETRÍA  (todo derivado de a = 4 cm)
# ================================================================
a = 0.04
c = a / 2
b = 2 * a

OBS_W = c
OBS_H = b

X_Pi      = 0.0
X_obs_izq = X_Pi + a
X_obs_der = X_obs_izq + OBS_W
X_Pf      = X_obs_der + a
OBS_CX    = X_obs_izq + OBS_W / 2
OBS_CY    = 0.0

obs = dict(
    xmin = X_obs_izq,
    xmax = X_obs_der,
    ymin = OBS_CY - OBS_H / 2,
    ymax = OBS_CY + OBS_H / 2,
)

M      = R_ROBOT + 0.005
Y_alto = obs["ymax"] + M

S  = np.array([X_Pi,  0.0,     np.pi    ])
P1 = np.array([X_Pi,  Y_alto,  np.pi/2  ])
P2 = np.array([X_Pf,  Y_alto,  0.0      ])
P3 = np.array([X_Pf,  0.0,    -np.pi/2  ])
P4 = np.array([X_Pf,  0.0,     0.0      ])

waypoints = [S, P1, P2, P3, P4]

# ================================================================
#  ENCODERS + REGISTRO DE PULSOS POR OPERACIÓN
# ================================================================
enc_L = enc_R = 0.0
registro_pulsos = []   # lista de dicts con info de cada operación

def actualizar_encoders(dL, dR):
    global enc_L, enc_R
    enc_L += dL
    enc_R += dR

# ================================================================
#  CINEMÁTICA DIRECTA
# ================================================================
def cinematica_directa(x, vL, vR):
    v  = r * (vR + vL) / 2.0
    w  = r * (vR - vL) / l
    xn = np.array([
        x[0] + v * np.cos(x[2]) * dt,
        x[1] + v * np.sin(x[2]) * dt,
        x[2] + w * dt
    ])
    xn[2] = np.arctan2(np.sin(xn[2]), np.cos(xn[2]))
    return xn

# ================================================================
#  PRIMITIVA 1: GIRAR — por pulsos exactos
# ================================================================
def girar_pulsos(x, theta_destino, etiqueta=""):
    global enc_L, enc_R
    traj = [x.copy()]
    x    = x.copy()

    delta = np.arctan2(np.sin(theta_destino - x[2]),
                       np.cos(theta_destino - x[2]))
    if abs(delta) < np.deg2rad(0.1):
        x[2] = theta_destino
        return np.array(traj), x

    pulsos_total = angulo_a_pulsos(delta)
    pulsos_ejec  = 0.0
    signo        = np.sign(delta)
    w_rueda      = V_ROT / r
    pulsos_paso  = (w_rueda * r * dt) / MM_POR_PULSO

    enc_L_antes = enc_L;  enc_R_antes = enc_R

    while pulsos_ejec + pulsos_paso <= pulsos_total:
        vR =  signo * w_rueda
        vL = -signo * w_rueda
        actualizar_encoders(-signo * pulsos_paso, signo * pulsos_paso)
        pulsos_ejec += pulsos_paso
        x = cinematica_directa(x, vL, vR)
        traj.append(x.copy())

    residuo = pulsos_total - pulsos_ejec
    if residuo > 1e-6:
        fraccion = residuo / pulsos_paso
        vR =  signo * w_rueda * fraccion
        vL = -signo * w_rueda * fraccion
        actualizar_encoders(-signo * residuo, signo * residuo)
        pulsos_ejec += residuo
        x = cinematica_directa(x, vL, vR)
        traj.append(x.copy())

    x[2] = theta_destino
    traj.append(x.copy())

    registro_pulsos.append({
        "op":       f"GIRO {etiqueta}",
        "delta_deg": np.rad2deg(delta),
        "pulsos_L": abs(enc_L - enc_L_antes),
        "pulsos_R": abs(enc_R - enc_R_antes),
        "pulsos_teoricos": pulsos_total,
    })
    return np.array(traj), x

# ================================================================
#  PRIMITIVA 2: AVANZAR — por pulsos exactos
# ================================================================
def avanzar_pulsos(x, distancia, etiqueta=""):
    global enc_L, enc_R
    traj   = [x.copy()]
    x      = x.copy()
    th     = x[2]
    x0, y0 = x[0], x[1]

    pulsos_total = metros_a_pulsos(distancia)
    pulsos_ejec  = 0.0
    w_rueda      = V_LIN / r
    pulsos_paso  = (w_rueda * r * dt) / MM_POR_PULSO

    enc_L_antes = enc_L;  enc_R_antes = enc_R

    while pulsos_ejec + pulsos_paso <= pulsos_total:
        actualizar_encoders(pulsos_paso, pulsos_paso)
        pulsos_ejec += pulsos_paso
        x  = cinematica_directa(x, w_rueda, w_rueda)
        x[2] = th
        traj.append(x.copy())

    residuo = pulsos_total - pulsos_ejec
    if residuo > 1e-6:
        fraccion = residuo / pulsos_paso
        actualizar_encoders(residuo, residuo)
        pulsos_ejec += residuo
        x = cinematica_directa(x, w_rueda * fraccion, w_rueda * fraccion)
        x[2] = th
        traj.append(x.copy())

    x[0] = x0 + distancia * np.cos(th)
    x[1] = y0 + distancia * np.sin(th)
    x[2] = th
    traj.append(x.copy())

    registro_pulsos.append({
        "op":       f"AVANCE {etiqueta}",
        "dist_cm":  distancia * 100,
        "pulsos_L": abs(enc_L - enc_L_antes),
        "pulsos_R": abs(enc_R - enc_R_antes),
        "pulsos_teoricos": pulsos_total,
    })
    return np.array(traj), x

# ================================================================
#  MOVER A WAYPOINT: GIRAR → AVANZAR → GIRAR
# ================================================================
def ir_a_waypoint(x, wp, idx_wp):
    gp  = wp[:2];  thf = wp[2]
    dx  = gp[0] - x[0];  dy = gp[1] - x[1]
    dist     = np.sqrt(dx**2 + dy**2)
    th_hacia = np.arctan2(dy, dx)

    t1, x = girar_pulsos (x, th_hacia, etiqueta=f"WP{idx_wp} orient")
    t2, x = avanzar_pulsos(x, dist,    etiqueta=f"WP{idx_wp} ({dist*100:.1f}cm)")
    x[0], x[1] = gp
    t3, x = girar_pulsos (x, thf,     etiqueta=f"WP{idx_wp} reorient")
    return np.vstack([t1, t2[1:], t3[1:]]), x

# ================================================================
#  SIMULAR
# ================================================================
print("Calculando trayectoria...")
trayectorias = []
x_curr = S.copy()
for i in range(len(waypoints) - 1):
    tramo, x_curr = ir_a_waypoint(x_curr, waypoints[i+1], i+1)
    trayectorias.append(tramo)

traj_total = np.vstack(trayectorias)

# ================================================================
#  REPORTE DE PULSOS
# ================================================================
print("\n" + "=" * 65)
print("   REPORTE DE PULSOS POR OPERACIÓN")
print("=" * 65)
print(f"  {'Operación':<30} {'Δθ/dist':>10} {'Pulsos L':>10} {'Pulsos R':>10}")
print("-" * 65)

total_pulsos_giro = 0.0
total_pulsos_avance = 0.0

for reg in registro_pulsos:
    if "GIRO" in reg["op"]:
        detalle = f"{reg['delta_deg']:+.1f}°"
        total_pulsos_giro += reg["pulsos_teoricos"]
        print(f"  {reg['op']:<30} {detalle:>10} "
              f"{reg['pulsos_L']:>10.1f} {reg['pulsos_R']:>10.1f}")
    else:
        detalle = f"{reg['dist_cm']:.2f} cm"
        total_pulsos_avance += reg["pulsos_teoricos"]
        print(f"  {reg['op']:<30} {detalle:>10} "
              f"{reg['pulsos_L']:>10.1f} {reg['pulsos_R']:>10.1f}")

print("=" * 65)
print(f"  {'TOTAL GIROS':<30} {'':>10} {total_pulsos_giro:>10.1f} {total_pulsos_giro:>10.1f}")
print(f"  {'TOTAL AVANCES':<30} {'':>10} {total_pulsos_avance:>10.1f} {total_pulsos_avance:>10.1f}")
print(f"  {'TOTAL GENERAL':<30} {'':>10} {enc_L:>10.1f} {enc_R:>10.1f}")
print("=" * 65)
print(f"\n  Referencia:")
print(f"  Giro 90°  = {angulo_a_pulsos(np.pi/2):.1f} pulsos/rueda")
print(f"  1 cm      = {metros_a_pulsos(0.01):.1f} pulsos/rueda")
print(f"  Tiempo total animación ≈ {len(traj_total)*dt:.1f} s")

# ================================================================
#  COLORES POR FASE
# ================================================================
COL_GIRO = '#FFD700'
COL_AVZ  = '#00E676'

fase_colors = []
x_tmp = S.copy()
for wp in waypoints[1:]:
    gp = wp[:2]; thf = wp[2]
    dx = gp[0]-x_tmp[0]; dy = gp[1]-x_tmp[1]
    dist     = np.sqrt(dx**2+dy**2)
    th_hacia = np.arctan2(dy, dx)

    def sim_giro(xc, th_d):
        delta = np.arctan2(np.sin(th_d-xc[2]), np.cos(th_d-xc[2]))
        if abs(delta) < np.deg2rad(0.1):
            xc[2] = th_d; return 1, xc
        pt = angulo_a_pulsos(delta); pe = 0.0
        pp = (V_ROT/r * r * dt) / MM_POR_PULSO
        signo = np.sign(delta); n = 1; xc = xc.copy()
        while pe + pp <= pt:
            pe += pp; n += 1
            xc = cinematica_directa(xc, -signo*V_ROT/r, signo*V_ROT/r)
        residuo = pt - pe
        if residuo > 1e-6:
            f = residuo/pp; n += 1
            xc = cinematica_directa(xc, -signo*V_ROT/r*f, signo*V_ROT/r*f)
        xc[2] = th_d; return n+1, xc

    def sim_avz(xc, dist2):
        pt = metros_a_pulsos(dist2); pe = 0.0; th = xc[2]
        pp = (V_LIN/r * r * dt) / MM_POR_PULSO
        n = 1; x0, y0 = xc[0], xc[1]; xc = xc.copy()
        while pe + pp <= pt:
            pe += pp; n += 1
            xc = cinematica_directa(xc, V_LIN/r, V_LIN/r); xc[2] = th
        residuo = pt - pe
        if residuo > 1e-6:
            f = residuo/pp; n += 1
            xc = cinematica_directa(xc, V_LIN/r*f, V_LIN/r*f); xc[2] = th
        xc[0] = x0+dist2*np.cos(th); xc[1] = y0+dist2*np.sin(th)
        return n+1, xc

    n1, x1 = sim_giro(x_tmp, th_hacia)
    n2, x2 = sim_avz (x1, dist)
    x2[0], x2[1] = gp
    n3, x3 = sim_giro(x2, thf)
    fase_colors += [COL_GIRO]*n1 + [COL_AVZ]*n2 + [COL_GIRO]*n3
    x_tmp = x3

fase_colors = (fase_colors + [COL_AVZ]*len(traj_total))[:len(traj_total)]

# ================================================================
#  FUNCIÓN: DIBUJAR ROBOT CON RUEDITAS
# ================================================================
def draw_robot_patch(ax, x, color_body):
    """
    Dibuja el cuerpo circular del robot + 2 ruedas rectangulares
    posicionadas a izquierda y derecha según la orientación θ.
    Devuelve lista de patches para poder eliminarlos después.
    """
    th  = x[2]
    cx, cy = x[0], x[1]

    added = []

    # ── Cuerpo ──────────────────────────────────────────────────
    body = plt.Circle((cx, cy), R_ROBOT, color=color_body,
                       zorder=5, alpha=0.92)
    body._khepera = True
    ax.add_patch(body)
    added.append(body)

    # ── Dirección (línea blanca) ─────────────────────────────────
    tip = (cx + R_ROBOT*1.7*np.cos(th), cy + R_ROBOT*1.7*np.sin(th))
    line, = ax.plot([cx, tip[0]], [cy, tip[1]],
                    color='white', lw=2.2, zorder=6)
    line._khepera = True
    added.append(line)

    # ── Ruedas (2 rectángulos, perpendiculares a la dirección) ───
    # Vector perpendicular a la dirección de movimiento
    perp = np.array([-np.sin(th), np.cos(th)])
    fwd  = np.array([ np.cos(th), np.sin(th)])

    # Posición de cada rueda: a ±l/2 del centro
    for lado in [+1, -1]:
        # Centro de la rueda
        wc = np.array([cx, cy]) + lado * (l/2) * perp

        # Las 4 esquinas del rectángulo de la rueda
        # largo a lo largo de fwd, ancho a lo largo de perp
        hw = R_RUEDA   # "ancho" de la rueda (radio)
        hl = L_RUEDA   # "largo" de la rueda

        corners = np.array([
            wc + hl*fwd + hw*perp,
            wc - hl*fwd + hw*perp,
            wc - hl*fwd - hw*perp,
            wc + hl*fwd - hw*perp,
        ])

        wheel = plt.Polygon(corners, color='#333333',
                            ec='#aaaaaa', lw=0.8, zorder=6)
        wheel._khepera = True
        ax.add_patch(wheel)
        added.append(wheel)

    return added

# ================================================================
#  FIGURA
# ================================================================
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    f"Robot Khepera  a={a*100:.0f}cm | Obstáculo {OBS_W*100:.0f}×{OBS_H*100:.0f}cm | "
    f"Control por PULSOS  |  🟡 Giro    🟢 Avance recto",
    fontsize=10, color='white'
)

ax_map = axes[0]
ax_enc = axes[1]

# ── Panel encoders ───────────────────────────────────────────────
eL_hist = [0.0]; eR_hist = [0.0]
_eL = _eR = 0.0
for k in range(len(traj_total)-1):
    dxy  = np.sqrt((traj_total[k+1,0]-traj_total[k,0])**2 +
                   (traj_total[k+1,1]-traj_total[k,1])**2)
    dth  = np.arctan2(np.sin(traj_total[k+1,2]-traj_total[k,2]),
                      np.cos(traj_total[k+1,2]-traj_total[k,2]))
    _eL += (dxy - dth*l/2) / MM_POR_PULSO
    _eR += (dxy + dth*l/2) / MM_POR_PULSO
    eL_hist.append(_eL); eR_hist.append(_eR)

tvec = np.arange(len(traj_total))*dt
ax_enc.plot(tvec, eL_hist, color='deepskyblue', lw=1.5, label='Enc. Izq (L)')
ax_enc.plot(tvec, eR_hist, color='tomato',      lw=1.5, label='Enc. Der (R)')

t_acc = 0
for i, tramo in enumerate(trayectorias):
    t_acc += len(tramo)
    idx = min(t_acc-1, len(tvec)-1)
    ax_enc.axvline(tvec[idx], color='white', lw=0.7, ls='--', alpha=0.4)
    ax_enc.text(tvec[idx]+0.1, 5, f'WP{i+1}', color='white', fontsize=7)

ax_enc.set_xlabel("Tiempo [s]")
ax_enc.set_ylabel("Pulsos acumulados")
ax_enc.set_title(f"Encoders L / R  |  Giro 90°={angulo_a_pulsos(np.pi/2):.0f}p  "
                 f"|  1cm={metros_a_pulsos(0.01):.0f}p")
ax_enc.legend(fontsize=9); ax_enc.grid(True, alpha=0.3)
enc_vline = ax_enc.axvline(0, color='yellow', lw=1.2, ls=':')

# ── Panel mapa ───────────────────────────────────────────────────
def draw_env(ax):
    ax.set_facecolor("#080808")
    ob = patches.Rectangle(
        (obs["xmin"], obs["ymin"]), OBS_W, OBS_H,
        lw=1.5, edgecolor='cyan', facecolor='#1a5f8a', zorder=2
    )
    ax.add_patch(ob)

    y_top = obs["ymax"] + 0.004
    y_bot = obs["ymin"] - 0.013

    ax.annotate("", xy=(obs["xmax"], y_top+0.009),
                xytext=(obs["xmin"], y_top+0.009),
                arrowprops=dict(arrowstyle="<->", color='cyan', lw=1))
    ax.text(OBS_CX, y_top+0.014, f"c={c*100:.0f}cm",
            color='cyan', fontsize=7, ha='center')

    ax.annotate("", xy=(obs["xmin"], y_bot),
                xytext=(X_Pi, y_bot),
                arrowprops=dict(arrowstyle="<->", color='lime', lw=1))
    ax.text((X_Pi+obs["xmin"])/2, y_bot-0.009,
            f"a={a*100:.0f}cm", color='lime', fontsize=7, ha='center')

    ax.annotate("", xy=(X_Pf, y_bot),
                xytext=(obs["xmax"], y_bot),
                arrowprops=dict(arrowstyle="<->", color='yellow', lw=1))
    ax.text((obs["xmax"]+X_Pf)/2, y_bot-0.009,
            f"a={a*100:.0f}cm", color='yellow', fontsize=7, ha='center')

    x_r = obs["xmax"]+0.007
    ax.annotate("", xy=(x_r, obs["ymax"]),
                xytext=(x_r, obs["ymin"]),
                arrowprops=dict(arrowstyle="<->", color='#ff9999', lw=1))
    ax.text(x_r+0.004, OBS_CY, f"2a={OBS_H*100:.0f}cm",
            color='#ff9999', fontsize=7, va='center')

    ax.axhline(Y_alto, color='#1e2e1e', lw=0.8, ls='--')
    ax.axhline(0.0,    color='#333333', lw=0.5, ls=':')

    ax.plot(X_Pi, 0, 'o', color='lime',   ms=10, zorder=8)
    ax.annotate("", xy=(X_Pi-0.013,0), xytext=(X_Pi+0.003,0),
                arrowprops=dict(arrowstyle="->", color='lime', lw=2), zorder=9)
    ax.text(X_Pi, -0.024, '$P_i$', color='lime', fontsize=10, ha='center')

    ax.plot(X_Pf, 0, '*', color='yellow', ms=13, zorder=8)
    ax.annotate("", xy=(X_Pf+0.013,0), xytext=(X_Pf-0.003,0),
                arrowprops=dict(arrowstyle="->", color='yellow', lw=2), zorder=9)
    ax.text(X_Pf, -0.024, '$P_f$', color='yellow', fontsize=10, ha='center')

    labels = ['S','P1','P2','P3','P4']
    for i, wp in enumerate(waypoints[1:-1], start=1):
        ax.plot(*wp[:2], 's', color='#444444', ms=4, zorder=3)
        ax.text(wp[0]+0.002, wp[1]+0.003, labels[i],
                color='#777777', fontsize=7, zorder=9)

draw_env(ax_map)
pad = 0.03
ax_map.set_xlim(X_Pi - pad*2, X_Pf + pad*2)
ax_map.set_ylim(obs["ymin"] - 0.045, Y_alto + pad)
ax_map.set_aspect('equal')
ax_map.set_xlabel("x [m]"); ax_map.set_ylabel("y [m]")
ax_map.grid(True, alpha=0.15)

traj_scat = ax_map.scatter([], [], s=5, zorder=4)
robot_parts = []   # lista de patches/lines del robot actual

# Subsampling: mostrar todos los frames (animación más lenta)
step_anim  = max(1, len(traj_total) // 800)
frames_idx = list(range(0, len(traj_total), step_anim)) + [len(traj_total)-1]

FASE = {COL_GIRO: '🟡 Girando', COL_AVZ: '🟢 Avanzando recto'}

def animate(fi):
    global robot_parts

    idx = frames_idx[fi]
    x   = traj_total[idx]
    col = fase_colors[idx]

    # Eliminar robot anterior
    for p in robot_parts:
        try:
            p.remove()
        except Exception:
            pass
    robot_parts = []

    # Dibujar robot con rueditas
    robot_parts = draw_robot_patch(ax_map, x, col)

    # Trayectoria coloreada
    if idx > 0:
        traj_scat.set_offsets(np.c_[traj_total[:idx,0],
                                     traj_total[:idx,1]])
        traj_scat.set_color(fase_colors[:idx])

    enc_vline.set_xdata([idx*dt, idx*dt])

    eL_now = eL_hist[min(idx, len(eL_hist)-1)]
    eR_now = eR_hist[min(idx, len(eR_hist)-1)]

    ax_map.set_title(
        f"{FASE.get(col,'?')}  |  t={idx*dt:.1f}s  |  "
        f"({x[0]*100:.1f},{x[1]*100:.1f})cm  θ={np.rad2deg(x[2]):.0f}°  |  "
        f"EncL={eL_now:.0f}p  EncR={eR_now:.0f}p",
        fontsize=8
    )
    return robot_parts + [traj_scat, enc_vline]

ani = animation.FuncAnimation(
    fig, animate,
    frames=len(frames_idx),
    interval=40,       # 40ms ≈ 25fps, más lento y realista
    blit=False,
    repeat=False
)

plt.subplots_adjust(left=0.07, right=0.97, bottom=0.09, top=0.91, wspace=0.28)
plt.show()