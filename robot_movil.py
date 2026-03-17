"""
Simulación de Robot Móvil Diferencial
======================================
3 Fases:
  1. Orientación inicial
  2. Desplazamiento
  3. Re-orientación final

✔ Movimiento en tiempo real
✘ No guarda GIF
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, Circle

# ─────────────────────────────────────────────
# Parámetros físicos
# ─────────────────────────────────────────────
L   = 54.0
R   =  8.0
PPR = 600

MM_PER_PULSE = (2 * np.pi * R) / PPR
print(f"mm por pulso: {MM_PER_PULSE:.6f} mm")

# ─────────────────────────────────────────────
# Puntos de misión
# ─────────────────────────────────────────────
START = np.array([0.0,   0.0])
END   = np.array([100.0, -90.0])
FINAL_ANGLE_DEG = 0.0

TURN_SPEED = 2.0
MOVE_SPEED = 60.0
DT         = 0.01

def normalize(a):
    while a >  np.pi: a -= 2*np.pi
    while a < -np.pi: a += 2*np.pi
    return a

# ─────────────────────────────────────────────
# Simulación offline completa
# ─────────────────────────────────────────────
def simulate():
    states = []
    pulses = []
    phases = []

    x, y, th = START[0], START[1], 0.0
    pL, pR = 0.0, 0.0

    target_angle1 = np.arctan2(END[1]-START[1], END[0]-START[0])
    target_angle2 = np.deg2rad(FINAL_ANGLE_DEG)

    # Fase 1
    for _ in range(20000):
        diff = normalize(target_angle1 - th)
        if abs(diff) < 0.005:
            break
        step = np.sign(diff) * TURN_SPEED * DT
        if abs(step) > abs(diff):
            step = diff
        th += step

        arc = abs(step) * L / 2
        if diff > 0:
            pL -= arc / MM_PER_PULSE
            pR += arc / MM_PER_PULSE
        else:
            pL += arc / MM_PER_PULSE
            pR -= arc / MM_PER_PULSE

        states.append((x, y, th))
        pulses.append((pL, pR))
        phases.append(1)

    # Fase 2
    for _ in range(20000):
        dx, dy = END[0]-x, END[1]-y
        dist = np.hypot(dx, dy)
        if dist < 0.5:
            break
        d = min(MOVE_SPEED * DT, dist)
        x += np.cos(th) * d
        y += np.sin(th) * d
        pL += d / MM_PER_PULSE
        pR += d / MM_PER_PULSE

        states.append((x, y, th))
        pulses.append((pL, pR))
        phases.append(2)

    # Fase 3
    for _ in range(20000):
        diff = normalize(target_angle2 - th)
        if abs(diff) < 0.005:
            break
        step = np.sign(diff) * TURN_SPEED * DT
        if abs(step) > abs(diff):
            step = diff
        th += step

        arc = abs(step) * L / 2
        if diff > 0:
            pL -= arc / MM_PER_PULSE
            pR += arc / MM_PER_PULSE
        else:
            pL += arc / MM_PER_PULSE
            pR -= arc / MM_PER_PULSE

        states.append((x, y, th))
        pulses.append((pL, pR))
        phases.append(3)

    return np.array(states), np.array(pulses), np.array(phases)

print("Calculando trayectoria...")
STATES, PULSES, PHASES = simulate()
print("Trayectoria calculada.")

# Submuestreo para suavidad
STRIDE = max(1, len(STATES)//500)
S  = STATES[::STRIDE]
P  = PULSES[::STRIDE]
PH = PHASES[::STRIDE]
N  = len(S)

# ─────────────────────────────────────────────
# Configuración figura
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[2,1])

ax_main  = fig.add_subplot(gs[:,0])
ax_theta = fig.add_subplot(gs[0,1])
ax_pulse = fig.add_subplot(gs[1,1])

# Mapa
ax_main.set_xlim(-30, 140)
ax_main.set_ylim(-130, 40)
ax_main.set_aspect('equal')
ax_main.grid(True)
ax_main.set_title("Plano XY")

ax_main.plot(*START, 'go')
ax_main.plot(*END, 'r+')

trail, = ax_main.plot([], [], 'b-', linewidth=2)

# Robot
BODY_W = 18
BODY_H = 14

robot = FancyBboxPatch((-BODY_W/2, -BODY_H/2),
                       BODY_W, BODY_H,
                       boxstyle="round,pad=1",
                       facecolor='cyan')

ax_main.add_patch(robot)

def move_robot(x, y, th):
    t = (mtransforms.Affine2D()
         .rotate(th)
         .translate(x, y)
         + ax_main.transData)
    robot.set_transform(t)

# Gráfica ángulo
ax_theta.set_xlim(0, N)
ax_theta.set_ylim(-200, 200)
ax_theta.set_title("Ángulo θ (°)")
theta_line, = ax_theta.plot([], [])

# Gráfica pulsos
ax_pulse.set_xlim(0, N)
ymax = max(abs(P[-1,0]), abs(P[-1,1])) * 1.2
ax_pulse.set_ylim(-ymax*0.1, ymax)
ax_pulse.set_title("Pulsos encoder")
pulseL, = ax_pulse.plot([], [], label="L")
pulseR, = ax_pulse.plot([], [], label="R")
ax_pulse.legend()

frames = np.arange(N)
theta_deg = np.degrees(S[:,2])

# ─────────────────────────────────────────────
# Función de actualización
# ─────────────────────────────────────────────
def update(i):

    # Trayectoria
    trail.set_data(S[:i+1,0], S[:i+1,1])

    # Robot
    x, y, th = S[i]
    move_robot(x, y, th)

    # Ángulo
    theta_line.set_data(frames[:i+1], theta_deg[:i+1])

    # Pulsos
    pulseL.set_data(frames[:i+1], P[:i+1,0])
    pulseR.set_data(frames[:i+1], P[:i+1,1])

    return trail, robot, theta_line, pulseL, pulseR

# Posición inicial
move_robot(START[0], START[1], 0)

# ─────────────────────────────────────────────
# Animación EN PANTALLA (sin guardar GIF)
# ─────────────────────────────────────────────
ani = FuncAnimation(fig, update, frames=N,
                    interval=20, blit=False, repeat=False)

plt.show()