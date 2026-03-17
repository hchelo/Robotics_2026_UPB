# Clase Robótica – Ing. de Sistemas UPB 2026

Se implementan algoritmos fundamentales de movimiento de un robot Diferencial:

- 📍 Localización con Filtro de Kalman Extendido (EKF)
- 🗺️ SLAM (Simultaneous Localization and Mapping)
- 🧭 Planificación de rutas (RRT y RDP)
- 🚗 Control de robot tipo Diferencial con encoders

## 📌 Tecnologías utilizadas

- Python
- NumPy
- Matplotlib
- Simulación cinemática
- Filtro de Kalman Extendido (EKF)


## 📍 Localización con EKF

El robot sigue una trayectoria senoidal mientras el EKF estima su posición en presencia de ruido.

![EKF](ekf_presentacion.gif)

---

## 🗺️ SLAM (Mapeo + Localización)

El robot construye un mapa del entorno mientras se localiza simultáneamente.

![SLAM](ekf_slam.gif)

---

## 🧭 Planificación de rutas

- RRT (Rapidly-exploring Random Trees)
![RRT](carrera_khepera.gif)
- RDP (Simplificación de trayectorias)
- Campos Potenciales
![PTH](path_potencial.gif)
Permite generar rutas eficientes en entornos con obstáculos.

## 🚀 Autor

Marcelo Saavedra Alcoba
