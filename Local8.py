import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Ellipse
import os
from PIL import Image
import tempfile

# ==================== CONFIG ====================
np.random.seed(42)

KHEPERA_L = 1.05
dt = 0.1
maxRange = 4.0

Q = np.diag([0.05**2, 0.05**2, (2*np.pi/180)**2])
R = np.diag([0.1**2, (5*np.pi/180)**2])
CHI2_THRESHOLD = 5.991

landmarks = np.array([
    [10,13],[12,11],[14,10],[16,8],
    [18,13],[10,7],[15,14],[13,6],
])

# ==================== FUNCIONES ====================

def normalize_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

def motion(x, vl, vr):
    v = (vr+vl)/2
    w = (vr-vl)/KHEPERA_L
    return np.array([
        x[0]+v*np.cos(x[2])*dt,
        x[1]+v*np.sin(x[2])*dt,
        normalize_angle(x[2]+w*dt)
    ])

def jacobian_F(x, vl, vr):
    v=(vr+vl)/2
    th=x[2]
    return np.array([
        [1,0,-v*np.sin(th)*dt],
        [0,1, v*np.cos(th)*dt],
        [0,0,1]
    ])

def measure(x,lm):
    dx=lm[0]-x[0]; dy=lm[1]-x[1]
    r=np.sqrt(dx**2+dy**2)
    phi=normalize_angle(np.arctan2(dy,dx)-x[2])
    return np.array([r,phi])

def jacobian_H(x,lm):
    dx=lm[0]-x[0]; dy=lm[1]-x[1]
    q=dx**2+dy**2; r=np.sqrt(q)
    return np.array([
        [-dx/r, -dy/r, 0],
        [ dy/q, -dx/q, -1]
    ])

def control(x):
    v = 0.5
    yref = 10 + 2*np.sin(0.3*x[0])
    theta_ref = np.arctan2(yref - x[1], 0.5)
    w = np.clip(2*(theta_ref - x[2]), -1.2, 1.2)
    vr = v + w*KHEPERA_L/2
    vl = v - w*KHEPERA_L/2
    return vl, vr

def run_simulation(n_frames=500):
    """Corre la simulación desde cero y devuelve lista de figuras."""
    x_real = np.array([0.0, 8.0, 0.0])
    x_est  = x_real.copy()
    P = np.eye(3)*0.3

    traj_real = []
    traj_est  = []
    errors    = []
    frames    = []

    fig, ax = plt.subplots(figsize=(7,7))

    for frame_idx in range(n_frames):
        vl, vr = control(x_real)

        # Real
        noise = np.array([0.02*np.random.randn(),
                          0.02*np.random.randn(),
                          0.01*np.random.randn()])
        x_real = motion(x_real, vl, vr) + noise

        # Predicción EKF
        F = jacobian_F(x_est, vl, vr)
        x_pred = motion(x_est, vl, vr)
        P = F @ P @ F.T + Q

        # Update EKF — registrar cuáles beacons fueron detectados
        detected_lms = []
        for i, lm in enumerate(landmarks):
            if np.linalg.norm(lm - x_real[:2]) < maxRange:
                z = measure(x_real, lm) + np.array([
                    np.random.randn()*np.sqrt(R[0,0]),
                    np.random.randn()*np.sqrt(R[1,1])
                ])
                H = jacobian_H(x_pred, lm)
                z_pred = measure(x_pred, lm)
                y = z - z_pred
                y[1] = normalize_angle(y[1])
                S = H @ P @ H.T + R
                if y @ np.linalg.solve(S, y) > CHI2_THRESHOLD:
                    continue
                K = P @ H.T @ np.linalg.inv(S)
                x_pred = x_pred + K @ y
                P = (np.eye(3) - K @ H) @ P
                detected_lms.append(i)  # beacon aceptado por gating

        x_est = x_pred
        traj_real.append(x_real.copy())
        traj_est.append(x_est.copy())
        errors.append(np.linalg.norm(x_real[:2] - x_est[:2]))

        # ---- Dibujar frame ----
        ax.cla()

        # Landmarks: negro normal, resaltados en naranja si detectados este frame
        for i, lm in enumerate(landmarks):
            if i in detected_lms:
                # Beacon detectado: círculo de rango + línea al robot + triángulo naranja
                ax.add_patch(Circle(lm, maxRange,
                                    color='orange', alpha=0.10, zorder=0))
                ax.plot([x_real[0], lm[0]], [x_real[1], lm[1]],
                        color='orange', lw=1.2, alpha=0.7, zorder=1)
                ax.scatter(lm[0], lm[1], c='orange', marker='^', s=120,
                           zorder=3, edgecolors='black', linewidths=0.8)
            else:
                ax.scatter(lm[0], lm[1], c='black', marker='^', s=60, zorder=3)

        tr = np.array(traj_real)
        te = np.array(traj_est)
        if len(tr) > 1:
            ax.plot(tr[:,0], tr[:,1], 'g',   label="Real")
            ax.plot(te[:,0], te[:,1], 'r--', label="EKF")

        ax.add_patch(Circle((x_real[0], x_real[1]), 0.3, color='green',  zorder=4))
        ax.add_patch(Circle((x_est[0],  x_est[1]),  0.3, color='red', alpha=0.6, zorder=4))

        vals, vecs = np.linalg.eigh(P[:2,:2])
        angle = np.degrees(np.arctan2(vecs[1,1], vecs[0,1]))
        ax.add_patch(Ellipse(
            (x_est[0], x_est[1]),
            2*np.sqrt(vals[0])*2,
            2*np.sqrt(vals[1])*2,
            angle=angle,
            edgecolor='red', fill=False, linestyle='--', zorder=5
        ))

        # Leyenda manual con proxy artists
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Line2D([0],[0], color='green', label='Real'),
            Line2D([0],[0], color='red', linestyle='--', label='EKF'),
            ax.scatter([],[],  c='black',  marker='^', label='Landmark'),
            ax.scatter([],[],  c='orange', marker='^', label='Detectado'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

        n_det = len(detected_lms)
        det_str = f" | Beacons: {n_det}" if n_det > 0 else ""
        ax.set_title(
            f"Frame {frame_idx+1:03d}/{n_frames}  |  EKF Localización  |  "
            f"Error={errors[-1]:.3f} m{det_str}"
        )
        # Centrar vista en el robot real
        half = 6.0  # mitad del ancho de ventana en metros
        cx, cy = x_real[0], x_real[1]
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.grid(True)

        # Capturar frame como imagen PIL
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
        frames.append(img)

        if (frame_idx+1) % 50 == 0:
            print(f"  {frame_idx+1}/{n_frames} frames procesados...")

    plt.close(fig)
    return frames

# ==================== GUARDAR GIF ====================

def guardar_gif(out="ekf_presentacion.gif", n_frames=500):
    print(f"Generando GIF ({n_frames} frames)...")
    frames = run_simulation(n_frames)
    print("Convirtiendo frames a paleta...")

    # Convertir a paleta para optimizar tamaño
    frames_p = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]

    frames_p[0].save(
        out,
        save_all=True,
        append_images=frames_p[1:],
        duration=80,
        loop=0,
        optimize=True
    )
    print(f"GIF guardado en: {out}")

guardar_gif()