import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from scipy.ndimage import binary_dilation
from skimage.morphology import ball

# ================================================================
#  MAPA REAL  50×50  (cargado del archivo del usuario)
#  Convención:
#    0  = libre
#    1  = obstáculo
#    2  = inicio del robot
#   -1  = meta
# ================================================================
import os

# ================================================================
#  LEER MAPA DESDE ARCHIVO  mapa.txt
#  Formato: filas separadas por \n, valores separados por \t o espacios
# ================================================================
ruta_mapa = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Mapa.txt')

mapa_raw = []
with open(ruta_mapa, 'r') as f:
    for linea in f:
        linea = linea.strip()
        if not linea:
            continue
        # Soporta separador tabulador o espacio
        valores = linea.replace('\t', ' ').split()
        mapa_raw.append([float(v) for v in valores])

print(f"  Archivo leído: {ruta_mapa}")
print(f"  Filas={len(mapa_raw)}  Cols={len(mapa_raw[0])}")

# ================================================================
#  PARSEAR MAPA: extraer inicio, meta y obstáculos
#  El mapa está en [fila, col] → transponer para que sea [x, y]
#  Escalar de 50×50 a 100×100 cm (cada celda = 2 cm)
# ================================================================
M_raw = np.array(mapa_raw, dtype=float)   # shape (50, 50)  [fila=y, col=x]

# El mapa raw está indexado [fila][col] → M_raw[row, col]
# Para campos potenciales usamos [col, row] = [x, y]
M = M_raw.T.copy()    # shape (50, 50)  M[x, y]

FILAS, COLS = M_raw.shape    # 50, 50
ANCHO = COLS   # eje X = columnas
ALTO  = FILAS  # eje Y = filas

# Buscar inicio (2) y meta (-1)
pos_inicio = np.argwhere(M_raw == 2)
pos_meta   = np.argwhere(M_raw == -1)

if len(pos_inicio) == 0:
    raise ValueError("No se encontró el inicio (valor 2) en el mapa")
if len(pos_meta) == 0:
    raise ValueError("No se encontró la meta (valor -1) en el mapa")

# [row, col] → xi=col, yi=row  (para indexar M[x,y])
xi = int(pos_inicio[0, 1])   # columna = x
yi = int(pos_inicio[0, 0])   # fila    = y
xg = int(pos_meta[0, 1])
yg = int(pos_meta[0, 0])

# Limpiar el mapa (quitar 2 y -1, dejar solo 0/1)
M[M ==  2] = 0
M[M == -1] = 0

print("=" * 55)
print("   MAPA CARGADO")
print("=" * 55)
print(f"  Tamaño mapa : {ANCHO} × {ALTO} celdas (≡ 100×100 cm, 2cm/celda)")
print(f"  Inicio (xi,yi) : ({xi}, {yi})")
print(f"  Meta   (xg,yg) : ({xg}, {yg})")
print(f"  Obstáculos     : {int(np.sum(M==1))} celdas")
print("=" * 55)

# ================================================================
#  DILATAR OBSTÁCULOS  (margen de seguridad para el robot)
#  Radio = 2 celdas ≈ 4 cm  (similar al strel sphere r=3 de MATLAB)
# ================================================================
RADIO_DILATE = 2

obs_bin  = (M == 1)
struct   = ball(RADIO_DILATE)[RADIO_DILATE]
M_dilat  = binary_dilation(obs_bin, structure=struct).astype(float)

# Proteger inicio y meta de la dilatación
M_dilat[xi, yi] = 0
M_dilat[xg, yg] = 0

# ================================================================
#  CAMPOS POTENCIALES
#  UA(i,j) = 1/2 * K * dist_a_meta
#  UR(i,j) = 10*K + 1/dist_al_inicio   (solo en obstáculos)
# ================================================================
K = 3.0

XX, YY = np.meshgrid(np.arange(ANCHO), np.arange(ALTO), indexing='ij')

# Distancias
dist_meta   = np.sqrt((XX - xg)**2 + (YY - yg)**2)
dist_meta[dist_meta == 0] = 1e-6

dist_inicio = np.sqrt((XX - xi)**2 + (YY - yi)**2)
dist_inicio[dist_inicio == 0] = 1e-6

# Potenciales
UA = 0.5 * K * dist_meta
UR = np.zeros_like(UA)
UR[M_dilat == 1] = 10*K + 1.0 / dist_inicio[M_dilat == 1]

U = UA + UR

# ================================================================
#  DESCENSO DE GRADIENTE  (8-vecinos, igual que MATLAB)
# ================================================================
path_x = [xi]
path_y = [yi]
cx, cy  = xi, yi
MAX_IT  = 50000

VECINOS = [(-1,-1),(-1,0),(-1,1),
           ( 0,-1),       ( 0,1),
           ( 1,-1),( 1,0),( 1,1)]

for _ in range(MAX_IT):
    if cx == xg and cy == yg:
        break
    mejor_val = np.inf
    nx, ny    = cx, cy
    for dx, dy in VECINOS:
        ni, nj = cx+dx, cy+dy
        if 0 <= ni < ANCHO and 0 <= nj < ALTO:
            if U[ni, nj] < mejor_val:
                mejor_val = U[ni, nj]
                nx, ny = ni, nj
    if nx == cx and ny == cy:
        print(f"⚠️  Mínimo local en ({cx},{cy})")
        break
    cx, cy = nx, ny
    path_x.append(cx)
    path_y.append(cy)

path_x = np.array(path_x)
path_y = np.array(path_y)
print(f"\n✅ Path encontrado: {len(path_x)} puntos")

# ================================================================
#  DETECTAR ESQUINAS (waypoints)
#  Cambio de dirección ≥ 20° → esquina
# ================================================================
UMBRAL_ANGULO = np.deg2rad(20)
waypoints_idx = [0]

for k in range(1, len(path_x)-1):
    v_ant = np.array([path_x[k]  -path_x[k-1], path_y[k]  -path_y[k-1]], dtype=float)
    v_sig = np.array([path_x[k+1]-path_x[k],   path_y[k+1]-path_y[k]],   dtype=float)
    na = np.linalg.norm(v_ant); ns = np.linalg.norm(v_sig)
    if na < 1e-9 or ns < 1e-9: continue
    cos_a  = np.clip(np.dot(v_ant, v_sig)/(na*ns), -1, 1)
    if np.arccos(cos_a) >= UMBRAL_ANGULO:
        waypoints_idx.append(k)

waypoints_idx.append(len(path_x)-1)

# Filtrar waypoints muy cercanos (dist mínima 3 celdas)
wp_f = [waypoints_idx[0]]
for idx in waypoints_idx[1:]:
    px, py = path_x[wp_f[-1]], path_y[wp_f[-1]]
    qx, qy = path_x[idx],      path_y[idx]
    if np.sqrt((qx-px)**2+(qy-py)**2) >= 3:
        wp_f.append(idx)

waypoints_idx = wp_f
WP_x = path_x[waypoints_idx]
WP_y = path_y[waypoints_idx]

# ================================================================
#  REPORTE EN CONSOLA
# ================================================================
print(f"📍 Waypoints detectados: {len(WP_x)}\n")
print(f"  {'#':>2}  {'X(cel)':>6}  {'Y(cel)':>6}  {'X(cm)':>6}  {'Y(cm)':>6}  {'θ':>7}")
print("  " + "─"*45)
for k in range(len(WP_x)):
    wx, wy = WP_x[k], WP_y[k]
    if k < len(WP_x)-1:
        dx_ = WP_x[k+1]-wx; dy_ = WP_y[k+1]-wy
        th_ = np.rad2deg(np.arctan2(dy_, dx_))
    else:
        th_ = np.rad2deg(np.arctan2(WP_y[-1]-WP_y[-2], WP_x[-1]-WP_x[-2]))
    label = " ← INICIO" if k==0 else (" ← META" if k==len(WP_x)-1 else "")
    print(f"  {k:>2}  {wx:>6}  {wy:>6}  {wx*2:>6.0f}  {wy*2:>6.0f}  {th_:>+6.1f}°{label}")

# ================================================================
#  VISUALIZACIÓN  (6 paneles igual que antes)
# ================================================================
plt.style.use("dark_background")
fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.patch.set_facecolor('#0d0d0d')
for ax in axes.flat:
    ax.set_facecolor('#0d0d0d')

fig.suptitle("Path Planning — Campos Potenciales  |  Mapa 100×100 cm  (2 cm/celda)",
             fontsize=13, color='white')

ext = [0, ANCHO, 0, ALTO]

# 1. Mapa original
ax = axes[0,0]
ax.imshow(M.T, origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=1.5)
ax.plot(xi, yi, 'go', ms=10, label='Inicio')
ax.plot(xg, yg, 'r*', ms=12, label='Meta')
ax.set_title("Mapa Original"); ax.legend(fontsize=8)
ax.set_xlabel("x [celdas]"); ax.set_ylabel("y [celdas]")

# 2. Mapa dilatado
ax = axes[0,1]
ax.imshow(M_dilat.T, origin='lower', extent=ext, cmap='Oranges', vmin=0, vmax=1.5)
ax.plot(xi, yi, 'go', ms=10); ax.plot(xg, yg, 'r*', ms=12)
ax.set_title(f"Obstáculos Dilatados (r={RADIO_DILATE} celdas)")
ax.set_xlabel("x [celdas]"); ax.set_ylabel("y [celdas]")

# 3. Potencial de atracción
ax = axes[0,2]
im3 = ax.imshow(UA.T, origin='lower', extent=ext, cmap='viridis')
plt.colorbar(im3, ax=ax, fraction=0.046)
ax.plot(xi, yi, 'go', ms=10); ax.plot(xg, yg, 'r*', ms=12)
ax.set_title("Potencial Atracción $U_A$")
ax.set_xlabel("x [celdas]"); ax.set_ylabel("y [celdas]")

# 4. Potencial de repulsión
ax = axes[1,0]
UR_vis = np.copy(UR); UR_vis[UR_vis > 50] = 50
im4 = ax.imshow(UR_vis.T, origin='lower', extent=ext, cmap='hot')
plt.colorbar(im4, ax=ax, fraction=0.046)
ax.plot(xi, yi, 'go', ms=10); ax.plot(xg, yg, 'r*', ms=12)
ax.set_title("Potencial Repulsión $U_R$")
ax.set_xlabel("x [celdas]"); ax.set_ylabel("y [celdas]")

# 5. Potencial total + gradiente
ax = axes[1,1]
U_vis = np.copy(U); U_vis[U_vis > 150] = 150
im5 = ax.imshow(U_vis.T, origin='lower', extent=ext, cmap='plasma')
plt.colorbar(im5, ax=ax, fraction=0.046)
step = 4
Xs = np.arange(0, ANCHO, step); Ys = np.arange(0, ALTO, step)
Xg_, Yg_ = np.meshgrid(Xs, Ys, indexing='ij')
gx, gy   = np.gradient(-U)
ax.quiver(Xg_, Yg_, gx[::step,::step], gy[::step,::step],
          color='white', alpha=0.4, scale=400, width=0.003)
ax.set_title("Potencial Total + Gradiente")
ax.set_xlabel("x [celdas]"); ax.set_ylabel("y [celdas]")

# 6. Ruta + waypoints
ax = axes[1,2]
ax.imshow(M_dilat.T, origin='lower', extent=ext,
          cmap='Greys', vmin=0, vmax=2, alpha=0.4)
ax.imshow(np.where(M==1, 1, np.nan).T, origin='lower', extent=ext,
          cmap='Blues', vmin=0, vmax=1, alpha=0.8)

ax.plot(path_x, path_y, color='lime', lw=1.5, alpha=0.8, label='Ruta')
ax.scatter(WP_x, WP_y, s=90, c='yellow', zorder=6,
           edgecolors='white', lw=0.8, label=f'{len(WP_x)} Waypoints')

for k in range(len(WP_x)):
    wx, wy = WP_x[k], WP_y[k]
    ax.text(wx+0.5, wy+0.5, str(k), color='yellow', fontsize=7, zorder=7)
    if k < len(WP_x)-1:
        dx_ = WP_x[k+1]-wx; dy_ = WP_y[k+1]-wy
        ax.annotate("",
            xy=(wx+dx_*0.6, wy+dy_*0.6),
            xytext=(wx+dx_*0.4, wy+dy_*0.4),
            arrowprops=dict(arrowstyle="->", color='orange', lw=1.5))

ax.plot(xi, yi, 'go', ms=11, zorder=8, label='Inicio')
ax.plot(xg, yg, 'r*', ms=13, zorder=8, label='Meta')
ax.set_xlim(0, ANCHO); ax.set_ylim(0, ALTO)
ax.set_title(f"Ruta + Waypoints (Esquinas)")
ax.set_xlabel("x [celdas]"); ax.set_ylabel("y [celdas]")
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.15)

# Tabla de waypoints dentro del panel
tabla = "Waypoints Khepera:\n\n"
tabla += f"  {'#':>2}  {'X':>4}  {'Y':>4}  {'Xcm':>5}  {'Ycm':>5}  {'θ':>7}\n"
tabla += "  " + "─"*35 + "\n"
for k in range(len(WP_x)):
    wx, wy = WP_x[k], WP_y[k]
    if k < len(WP_x)-1:
        dx_ = WP_x[k+1]-wx; dy_ = WP_y[k+1]-wy
        th_ = np.rad2deg(np.arctan2(dy_, dx_))
    else:
        th_ = np.rad2deg(np.arctan2(WP_y[-1]-WP_y[-2], WP_x[-1]-WP_x[-2]))
    tabla += f"  {k:>2}  {wx:>4}  {wy:>4}  {wx*2:>5.0f}  {wy*2:>5.0f}  {th_:>+6.1f}°\n"

ax.text(1.04, 0.99, tabla, transform=ax.transAxes,
        fontsize=7, color='white', va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#1a1a2e',
                  edgecolor='#4444aa', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ================================================================
# ================================================================
#  FIGURA 2 — ROBOT KHEPERA NAVEGANDO POR LOS WAYPOINTS
#  Cinemática inversa por pulsos:
#    1. ORIENTAR  → girar en sitio hasta apuntar al waypoint
#    2. DESPLAZAR → avanzar recto contando pulsos
#    3. REORIENTAR→ girar hasta orientación final del waypoint
# ================================================================
# ================================================================

# ── Parámetros Khepera ──────────────────────────────────────────
r_k       = 0.008    # radio rueda [m]
l_k       = 0.054    # distancia entre ruedas [m]
R_ROBOT_K = 0.0275   # radio robot [m]  ∅55mm
R_RUEDA_K = 0.005    # grosor visual rueda
L_RUEDA_K = 0.010    # largo visual rueda
dt_k      = 0.05     # paso de tiempo [s]

PULSOS_REV_K   = 600
MM_POR_PULSO_K = (2 * np.pi * r_k) / PULSOS_REV_K

V_LIN_K = 0.03   # m/s  (lento y visible)
V_ROT_K = 0.25   # rad/s

# Escala mapa→metros: cada celda = 2 cm = 0.02 m
CEL_A_M = 0.02

# ── Conversiones encoder ────────────────────────────────────────
def m2p(metros):
    return abs(metros) / MM_POR_PULSO_K

def a2p(rad):
    return abs((l_k / 2) * rad) / MM_POR_PULSO_K

# ── Cinemática directa ──────────────────────────────────────────
def cinem_dir(x, vL, vR):
    v  = r_k * (vR + vL) / 2.0
    w  = r_k * (vR - vL) / l_k
    xn = np.array([
        x[0] + v * np.cos(x[2]) * dt_k,
        x[1] + v * np.sin(x[2]) * dt_k,
        x[2] + w * dt_k
    ])
    xn[2] = np.arctan2(np.sin(xn[2]), np.cos(xn[2]))
    return xn

# ── Encoders globales ───────────────────────────────────────────
enc_L_k = enc_R_k = 0.0
reg_pulsos = []

def upd_enc(dL, dR):
    global enc_L_k, enc_R_k
    enc_L_k += dL; enc_R_k += dR

# ── PASO 1: ORIENTAR ────────────────────────────────────────────
def girar_k(x, theta_dest, tag=""):
    global enc_L_k, enc_R_k
    traj = [x.copy()]; x = x.copy()
    delta = np.arctan2(np.sin(theta_dest-x[2]), np.cos(theta_dest-x[2]))
    if abs(delta) < np.deg2rad(0.1):
        x[2] = theta_dest; return np.array(traj), x, 0.0
    pt    = a2p(delta); pe = 0.0; signo = np.sign(delta)
    wr    = V_ROT_K / r_k
    pp    = (wr * r_k * dt_k) / MM_POR_PULSO_K
    eL0   = enc_L_k; eR0 = enc_R_k
    while pe + pp <= pt:
        upd_enc(-signo*pp, signo*pp); pe += pp
        x = cinem_dir(x, -signo*wr, signo*wr); traj.append(x.copy())
    res = pt - pe
    if res > 1e-6:
        f = res/pp; upd_enc(-signo*res, signo*res); pe += res
        x = cinem_dir(x, -signo*wr*f, signo*wr*f); traj.append(x.copy())
    x[2] = theta_dest; traj.append(x.copy())
    reg_pulsos.append({"op": f"GIRO {tag}", "delta": np.rad2deg(delta), "p": pt})
    return np.array(traj), x, pt

# ── PASO 2: DESPLAZAR ───────────────────────────────────────────
def avanzar_k(x, dist, tag=""):
    traj = [x.copy()]; x = x.copy()
    th = x[2]; x0, y0 = x[0], x[1]
    pt = m2p(dist); pe = 0.0
    wr = V_LIN_K / r_k
    pp = (wr * r_k * dt_k) / MM_POR_PULSO_K
    while pe + pp <= pt:
        upd_enc(pp, pp); pe += pp
        x = cinem_dir(x, wr, wr); x[2] = th; traj.append(x.copy())
    res = pt - pe
    if res > 1e-6:
        f = res/pp; upd_enc(res, res); pe += res
        x = cinem_dir(x, wr*f, wr*f); x[2] = th; traj.append(x.copy())
    x[0] = x0 + dist*np.cos(th)
    x[1] = y0 + dist*np.sin(th)
    x[2] = th; traj.append(x.copy())
    reg_pulsos.append({"op": f"AVANCE {tag}", "dist_m": dist, "p": pt})
    return np.array(traj), x, pt

# ── Secuencia completa por waypoint ─────────────────────────────
def ir_wp_k(x, wp_m, idx):
    gp  = wp_m[:2]; thf = wp_m[2]
    dx_ = gp[0]-x[0]; dy_ = gp[1]-x[1]
    dist     = np.sqrt(dx_**2+dy_**2)
    th_hacia = np.arctan2(dy_, dx_)
    t1, x, p1 = girar_k  (x, th_hacia, tag=f"WP{idx} orient")
    t2, x, p2 = avanzar_k(x, dist,     tag=f"WP{idx} {dist*100:.1f}cm")
    x[0], x[1] = gp
    t3, x, p3 = girar_k  (x, thf,      tag=f"WP{idx} reorient")
    return np.vstack([t1, t2[1:], t3[1:]]), x

# ── Construir waypoints en metros ───────────────────────────────
# La orientación final de cada WP = dirección hacia el siguiente
wp_metros = []
for k in range(len(WP_x)):
    wx_m = WP_x[k] * CEL_A_M
    wy_m = WP_y[k] * CEL_A_M
    if k < len(WP_x)-1:
        dx_ = (WP_x[k+1]-WP_x[k])*CEL_A_M
        dy_ = (WP_y[k+1]-WP_y[k])*CEL_A_M
        th_ = np.arctan2(dy_, dx_)
    else:
        dx_ = (WP_x[-1]-WP_x[-2])*CEL_A_M
        dy_ = (WP_y[-1]-WP_y[-2])*CEL_A_M
        th_ = np.arctan2(dy_, dx_)
    wp_metros.append(np.array([wx_m, wy_m, th_]))

# ── Simular trayectoria completa ────────────────────────────────
print("\nSimulando Khepera en waypoints...")
trajs_k = []
x_k = wp_metros[0].copy()
for i in range(len(wp_metros)-1):
    tramo, x_k = ir_wp_k(x_k, wp_metros[i+1], i+1)
    trajs_k.append(tramo)

traj_k = np.vstack(trajs_k)

# ── Colores por fase ────────────────────────────────────────────
COL_G = '#FFD700'   # amarillo = girando
COL_A = '#00E676'   # verde    = avanzando

fase_k = []
xc = wp_metros[0].copy()
for wp in wp_metros[1:]:
    gp = wp[:2]; thf = wp[2]
    dx_ = gp[0]-xc[0]; dy_ = gp[1]-xc[1]
    dist     = np.sqrt(dx_**2+dy_**2)
    th_hacia = np.arctan2(dy_, dx_)

    def ng(x, th_d):
        delta = np.arctan2(np.sin(th_d-x[2]), np.cos(th_d-x[2]))
        if abs(delta) < np.deg2rad(0.1): return 1, x.copy()
        pt = a2p(delta); pe = 0.0; s = np.sign(delta)
        wr = V_ROT_K/r_k; pp = (wr*r_k*dt_k)/MM_POR_PULSO_K
        n = 1; xc2 = x.copy()
        while pe+pp <= pt:
            pe += pp; n += 1
            xc2 = cinem_dir(xc2, -s*wr, s*wr)
        res = pt-pe
        if res>1e-6:
            f=res/pp; n+=1; xc2=cinem_dir(xc2,-s*wr*f,s*wr*f)
        xc2[2]=th_d; return n+1, xc2

    def na(x, d):
        pt = m2p(d); pe = 0.0; th = x[2]
        wr = V_LIN_K/r_k; pp=(wr*r_k*dt_k)/MM_POR_PULSO_K
        n=1; x0,y0=x[0],x[1]; xc2=x.copy()
        while pe+pp<=pt:
            pe+=pp; n+=1; xc2=cinem_dir(xc2,wr,wr); xc2[2]=th
        res=pt-pe
        if res>1e-6:
            f=res/pp; n+=1; xc2=cinem_dir(xc2,wr*f,wr*f); xc2[2]=th
        xc2[0]=x0+d*np.cos(th); xc2[1]=y0+d*np.sin(th)
        return n+1, xc2

    n1,x1 = ng(xc, th_hacia)
    n2,x2 = na(x1, dist)
    x2[0],x2[1] = gp
    n3,x3 = ng(x2, thf)
    fase_k += [COL_G]*n1 + [COL_A]*n2 + [COL_G]*n3
    xc = x3

fase_k = (fase_k + [COL_A]*len(traj_k))[:len(traj_k)]

# ── Reporte pulsos ───────────────────────────────────────────────
print("\n" + "="*60)
print("   REPORTE DE PULSOS — KHEPERA")
print("="*60)
print(f"  {'Operación':<28} {'Δ':>10} {'Pulsos':>10}")
print("  " + "─"*50)
tot_g = tot_a = 0.0
for reg in reg_pulsos:
    if "GIRO" in reg["op"]:
        print(f"  {reg['op']:<28} {reg['delta']:>+9.1f}°  {reg['p']:>9.1f}")
        tot_g += reg["p"]
    else:
        print(f"  {reg['op']:<28} {reg['dist_m']*100:>9.2f}cm  {reg['p']:>9.1f}")
        tot_a += reg["p"]
print("="*60)
print(f"  {'TOTAL GIROS':<28} {'':>10}  {tot_g:>9.1f}")
print(f"  {'TOTAL AVANCES':<28} {'':>10}  {tot_a:>9.1f}")
print(f"  {'TOTAL GENERAL L':<28} {'':>10}  {enc_L_k:>9.1f}")
print(f"  {'TOTAL GENERAL R':<28} {'':>10}  {enc_R_k:>9.1f}")
print("="*60)

# ── Figura 2: animación robot ────────────────────────────────────
plt.style.use("dark_background")
fig2, ax2 = plt.subplots(figsize=(9, 9))
fig2.patch.set_facecolor('#0d0d0d')
ax2.set_facecolor('#080808')
fig2.suptitle("Robot Khepera — Cinemática Inversa por Pulsos\n"
              "🟡 Orientar   🟢 Desplazar   (3 pasos por waypoint)",
              fontsize=11, color='white')

# Dibujar mapa de fondo (escalado a metros)
ext_m = [0, ANCHO*CEL_A_M, 0, ALTO*CEL_A_M]
ax2.imshow(M_dilat.T, origin='lower', extent=ext_m,
           cmap='Greys', vmin=0, vmax=2, alpha=0.35)
ax2.imshow(np.where(M==1, 1, np.nan).T, origin='lower', extent=ext_m,
           cmap='Blues', vmin=0, vmax=1, alpha=0.7)

# Ruta del path planning (en metros)
ax2.plot(path_x*CEL_A_M, path_y*CEL_A_M,
         color='#00ff0033', lw=1, alpha=0.5, label='Ruta potencial')

# Waypoints
for k, wp in enumerate(wp_metros):
    ax2.plot(wp[0], wp[1], 's', color='yellow', ms=7, zorder=6)
    ax2.text(wp[0]+0.002, wp[1]+0.002, str(k),
             color='yellow', fontsize=8, zorder=7)

# Inicio y meta
ax2.plot(wp_metros[0][0],  wp_metros[0][1],  'go', ms=12, zorder=8, label='Inicio')
ax2.plot(wp_metros[-1][0], wp_metros[-1][1],  'r*', ms=14, zorder=8, label='Meta')

ax2.set_xlim(0, ANCHO*CEL_A_M)
ax2.set_ylim(0, ALTO*CEL_A_M)
ax2.set_aspect('equal')
ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.15)

traj_scat2 = ax2.scatter([], [], s=5, zorder=4)
robot_parts2 = []

def draw_robot2(ax, x, color):
    added = []
    cx, cy, th = x[0], x[1], x[2]
    # Cuerpo circular
    body = plt.Circle((cx, cy), R_ROBOT_K, color=color, zorder=5, alpha=0.92)
    body._kh = True; ax.add_patch(body); added.append(body)
    # Flecha dirección
    ln, = ax.plot([cx, cx+R_ROBOT_K*1.7*np.cos(th)],
                  [cy, cy+R_ROBOT_K*1.7*np.sin(th)],
                  color='white', lw=2, zorder=6)
    ln._kh = True; added.append(ln)
    # Ruedas
    perp = np.array([-np.sin(th),  np.cos(th)])
    fwd  = np.array([ np.cos(th),  np.sin(th)])
    for lado in [+1, -1]:
        wc = np.array([cx, cy]) + lado*(l_k/2)*perp
        corners = np.array([
            wc + L_RUEDA_K*fwd + R_RUEDA_K*perp,
            wc - L_RUEDA_K*fwd + R_RUEDA_K*perp,
            wc - L_RUEDA_K*fwd - R_RUEDA_K*perp,
            wc + L_RUEDA_K*fwd - R_RUEDA_K*perp,
        ])
        wh = plt.Polygon(corners, color='#333333', ec='#aaaaaa',
                         lw=0.8, zorder=6)
        wh._kh = True; ax.add_patch(wh); added.append(wh)
    return added

step2      = max(1, len(traj_k)//600)
frames2    = list(range(0, len(traj_k), step2)) + [len(traj_k)-1]
FASE2      = {COL_G: '🟡 Orientando', COL_A: '🟢 Desplazando'}

def animate2(fi):
    global robot_parts2
    idx = frames2[fi]; x = traj_k[idx]; col = fase_k[idx]
    for p in robot_parts2:
        try: p.remove()
        except: pass
    robot_parts2 = draw_robot2(ax2, x, col)
    if idx > 0:
        traj_scat2.set_offsets(np.c_[traj_k[:idx,0], traj_k[:idx,1]])
        traj_scat2.set_color(fase_k[:idx])
    ax2.set_title(
        f"{FASE2.get(col,'?')}  |  t={idx*dt_k:.1f}s  |  "
        f"pos=({x[0]*100:.1f},{x[1]*100:.1f})cm  θ={np.rad2deg(x[2]):.0f}°  |  "
        f"EncL={enc_L_k:.0f}p  EncR={enc_R_k:.0f}p",
        fontsize=8, color='white'
    )
    return robot_parts2 + [traj_scat2]

ani2 = animation.FuncAnimation(
    fig2, animate2,
    frames=len(frames2),
    interval=30,
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()