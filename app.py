import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp

# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="Free Kick Simulator",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Free Kick Simulator")
st.caption(
    "Interactive browser version of your Python free-kick trajectory simulator. "
    "Tune launch, drag, and spin values to see the ball path and whether it scores."
)

# ==============================
# Physics Constants
# ==============================
g = 9.81        # Gravity (m/s^2)
rho = 1.2       # Air density (kg/m^3)
R = 0.11        # Ball radius (m)
A = np.pi * R**2  # Cross-sectional area (m^2)
m = 0.43        # Ball mass (kg)
mu = 1.81e-5    # Dynamic viscosity of air (Pa·s)

goal_width = 7.32
goal_height = 2.44
net_depth = 1.5

# Force model options
FORCE_MODE_NONE = "Initial velocity only"
FORCE_MODE_GRAVITY = "Gravity only"
FORCE_MODE_DRAG = "Gravity + air resistance"
FORCE_MODE_FULL = "Gravity + drag + Magnus"
FORCE_MODES = [
    FORCE_MODE_NONE,
    FORCE_MODE_GRAVITY,
    FORCE_MODE_DRAG,
    FORCE_MODE_FULL,
]

PRESETS = {
    "Custom": None,
    "Ronaldo": {
        "force_mode": FORCE_MODE_FULL,
        "vx": 30.0,
        "vy": 0.0,
        "vz": 8.0,
        "spin_side": 0.0,
        "spin_top": 2.0,
        "cd": 0.35,
        "cl": 0.02,
        "goal_x": 25.0,
    },
    "Beckham": {
        "force_mode": FORCE_MODE_FULL,
        "vx": 24.0,
        "vy": 0.0,
        "vz": 11.0,
        "spin_side": 0.0,
        "spin_top": 35.0,
        "cd": 0.25,
        "cl": 0.30,
        "goal_x": 30.0,
    },
    "Carlos": {
        "force_mode": FORCE_MODE_FULL,
        "vx": 40.0,
        "vy": -10.0,
        "vz": 6.5,
        "spin_side": 0.0,
        "spin_top": 60.0,
        "cd": 0.3,
        "cl": 0.25,
        "goal_x": 29.0,
    },
    "Messi": {
        "force_mode": FORCE_MODE_FULL,
        "vx": 22.0,
        "vy": 0.0,
        "vz": 12.0,
        "spin_side": 0.0,
        "spin_top": 40.0,
        "cd": 0.25,
        "cl": 0.32,
        "goal_x": 22.0,
    },
}

DEFAULTS = {
    "force_mode": FORCE_MODE_FULL,
    "x0": 0.0,
    "y0": 0.0,
    "z0": 0.01,
    "vx": 24.0,
    "vy": 0.0,
    "vz": 10.0,
    "spin_side": 0.0,
    "spin_top": 20.0,
    "cd": 0.25,
    "cl": 0.25,
    "goal_x": 20.0,
}


def calculate_dimensionless_numbers(U0, omega, CD, CL, force_mode):
    if force_mode in [FORCE_MODE_NONE, FORCE_MODE_GRAVITY]:
        effective_CD = 0.0
        effective_CL = 0.0
        effective_omega = 0.0
    elif force_mode == FORCE_MODE_DRAG:
        effective_CD = CD
        effective_CL = 0.0
        effective_omega = 0.0
    else:
        effective_CD = CD
        effective_CL = CL
        effective_omega = omega

    F_D = 0.5 * rho * effective_CD * np.pi * R**2 * U0**2
    F_G = m * g
    D_r = F_D / F_G if F_G > 0 else 0.0

    SP = (4 * effective_CL * R * effective_omega) / (effective_CD * U0) if (effective_CD * U0) > 0 else 0.0

    D = 2 * R
    Re = (rho * U0 * D) / mu
    return D_r, SP, Re



def classify_trajectory(D_r, SP, force_mode):
    if force_mode == FORCE_MODE_NONE:
        return "INERTIAL STRAIGHT LINE", "#795548"
    if force_mode == FORCE_MODE_GRAVITY:
        return "GRAVITATIONAL PARABOLA", "#4CAF50"
    if force_mode == FORCE_MODE_DRAG:
        if D_r > 1.0:
            return "STRAIGHT / DRAG-DOMINATED", "#FF9800"
        return "TRUNCATED PARABOLA", "#2196F3"

    if D_r < 0.1 and SP < 0.1:
        return "GRAVITATIONAL PARABOLA", "#4CAF50"
    if D_r > 1.0 and SP < 0.1:
        return "STRAIGHT/KNUCKLEBALL", "#FF9800"
    if abs(D_r - 1.0) < 0.5 and SP < 0.1:
        return "TRUNCATED PARABOLA", "#2196F3"
    if D_r > 0.5 and SP > 1.0:
        return "SPIRAL (Magnus)", "#E91E63"
    if 0.1 < SP < 1.0:
        return "CURVED TRAJECTORY", "#9C27B0"
    return "MIXED REGIME", "#607D8B"



def football_ode(t, u, CD, CL, omega_x, omega_y, omega_z, force_mode):
    x, y, z, vx, vy, vz = u
    v = np.array([vx, vy, vz], dtype=float)
    v_mag = np.linalg.norm(v)

    if force_mode == FORCE_MODE_NONE:
        return [vx, vy, vz, 0.0, 0.0, 0.0]

    if force_mode == FORCE_MODE_GRAVITY:
        return [vx, vy, vz, 0.0, 0.0, -g]

    if v_mag < 0.1:
        drag_acc = np.zeros(3)
    else:
        drag_acc = -(0.5 * rho * A / m) * CD * v_mag * v

    if force_mode == FORCE_MODE_DRAG:
        return [vx, vy, vz, drag_acc[0], drag_acc[1], drag_acc[2] - g]

    omega = np.array([omega_x, omega_y, omega_z], dtype=float)
    omega_mag = np.linalg.norm(omega)

    if v_mag < 0.1 or omega_mag < 0.1:
        magnus_acc = np.zeros(3)
    else:
        omega_cross_v = np.cross(omega, v)
        magnus_acc = (0.5 * rho * A / m) * CL * v_mag * omega_cross_v / omega_mag

    return [
        vx,
        vy,
        vz,
        drag_acc[0] + magnus_acc[0],
        drag_acc[1] + magnus_acc[1],
        drag_acc[2] + magnus_acc[2] - g,
    ]



def hit_ground(t, u, CD, CL, omega_x, omega_y, omega_z, force_mode):
    if t < 0.01:
        return 1.0
    return u[2]


hit_ground.terminal = True
hit_ground.direction = -1


@st.cache_data(show_spinner=False)
def calculate_trajectory(x0, y0, z0, vx, vy, vz, CD, CL, omega_x, omega_y, omega_z, goal_x, force_mode):
    u0 = [x0, y0, z0, vx, vy, vz]

    sol = solve_ivp(
        lambda t, u: football_ode(t, u, CD, CL, omega_x, omega_y, omega_z, force_mode),
        (0, 10),
        u0,
        events=lambda t, u: hit_ground(t, u, CD, CL, omega_x, omega_y, omega_z, force_mode),
        dense_output=True,
        max_step=0.01,
    )

    if sol.t_events[0].size > 0:
        t_ground = sol.t_events[0][0]
        t_final = np.linspace(0, t_ground, 300)
    else:
        t_end = sol.t[-1] if len(sol.t) > 0 else 10
        t_final = np.linspace(0, t_end, 300)

    if len(t_final) > 0:
        trajectory = sol.sol(t_final)
        x = trajectory[0]
        y = trajectory[1]
        z = trajectory[2]
        if force_mode != FORCE_MODE_NONE:
            z = np.maximum(z, 0)
    else:
        x = np.array([x0])
        y = np.array([y0])
        z = np.array([z0])

    goal_tolerance = 0.5
    ball_in_goal = False
    goal_point = None

    for i in range(len(x)):
        if (
            goal_x - goal_tolerance <= x[i] <= goal_x + goal_tolerance
            and -goal_width / 2 <= y[i] <= goal_width / 2
            and 0 <= z[i] <= goal_height
        ):
            ball_in_goal = True
            goal_point = (x[i], y[i], z[i])
            break

    return x, y, z, ball_in_goal, goal_point



def build_phase_figure(D_r, SP, force_mode):
    fig = go.Figure()

    regions = [
        (0.01, 0.1, 0.01, 0.1, "Gravitational Parabola", "rgba(76,175,80,0.25)"),
        (1.0, 10.0, 0.01, 0.1, "Straight / Knuckleball", "rgba(255,152,0,0.25)"),
        (0.5, 1.5, 0.01, 0.15, "Truncated Parabola", "rgba(33,150,243,0.2)"),
        (1.0, 10.0, 1.0, 10.0, "Spiral (Magnus)", "rgba(233,30,99,0.25)"),
        (0.5, 10.0, 0.1, 1.0, "Curved", "rgba(156,39,176,0.18)"),
    ]
    for x0, x1, y0, y1, name, color in regions:
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor=color,
                hoverinfo="skip",
                showlegend=False,
                name=name,
            )
        )

    if force_mode == FORCE_MODE_NONE:
        px, py = 0.011, 0.011
    elif force_mode in [FORCE_MODE_GRAVITY, FORCE_MODE_DRAG] and D_r > 0.01 and D_r < 10:
        px, py = max(D_r, 0.011), 0.011
    elif D_r > 0.01 and SP > 0.01 and D_r < 10 and SP < 10:
        px, py = D_r, SP
    else:
        px, py = None, None

    if px is not None and py is not None:
        fig.add_trace(
            go.Scatter(
                x=[px],
                y=[py],
                mode="markers",
                marker=dict(size=14, color="red", line=dict(width=2, color="black")),
                name="Current Kick",
            )
        )

    for val in [0.1, 1.0]:
        fig.add_vline(x=val, line_dash="dash", line_color="gray", opacity=0.6)
        fig.add_hline(y=val, line_dash="dash", line_color="gray", opacity=0.6)

    fig.update_layout(
        title="Phase Diagram",
        xaxis_title="Drag Number (D_r)",
        yaxis_title="Spin Number (SP)",
        xaxis_type="log",
        yaxis_type="log",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig



def _goal_segments(goal_x):
    return [
        ([goal_x, goal_x], [-goal_width / 2, -goal_width / 2], [0, goal_height]),
        ([goal_x, goal_x], [goal_width / 2, goal_width / 2], [0, goal_height]),
        ([goal_x, goal_x], [-goal_width / 2, goal_width / 2], [goal_height, goal_height]),
        ([goal_x, goal_x, goal_x, goal_x, goal_x],
         [-goal_width / 2, goal_width / 2, goal_width / 2, -goal_width / 2, -goal_width / 2],
         [0, 0, goal_height, goal_height, 0]),
    ]



def build_trajectory_figure(x, y, z, x0, y0, z0, goal_x, traj_color, goal_point):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color=traj_color, width=8),
            name="Ball Path",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[x0], y=[y0], z=[z0],
            mode="markers",
            marker=dict(size=6, color="black"),
            name="Start",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode="markers",
            marker=dict(size=5, color="white", line=dict(width=2, color="black")),
            name="Ball",
        )
    )

    if goal_point is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[goal_point[0]], y=[goal_point[1]], z=[goal_point[2]],
                mode="markers",
                marker=dict(size=8, color="gold", symbol="diamond"),
                name="Goal Point",
            )
        )

    for xs, ys, zs in _goal_segments(goal_x):
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color="#d62728", width=7),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    x_max = max(35.0, float(np.max(x)) + 5.0, goal_x + 5.0)
    x_min = min(-5.0, float(np.min(x)) - 2.0, x0 - 2.0)
    z_max = max(15.0, float(np.max(z)) + 1.0)
    y_range = max(15.0, abs(float(np.min(y))) + 2.0, abs(float(np.max(y))) + 2.0)

    fig.update_layout(
        title="3D Free Kick Trajectory",
        height=650,
        scene=dict(
            xaxis_title="Distance (m)",
            yaxis_title="Lateral (m)",
            zaxis_title="Height (m)",
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[-y_range, y_range]),
            zaxis=dict(range=[0, z_max]),
            aspectmode="manual",
            aspectratio=dict(x=1.7, y=1.1, z=0.8),
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


# ==============================
# Sidebar controls
# ==============================
with st.sidebar:
    st.header("Controls")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]

    current = DEFAULTS.copy()
    if preset:
        current.update(preset)

    force_mode = st.radio("Force model", FORCE_MODES, index=FORCE_MODES.index(current["force_mode"]))

    st.subheader("Start position")
    x0 = st.slider("X", -5.0, 5.0, float(current["x0"]), 0.5)
    y0 = st.slider("Y", -10.0, 10.0, float(current["y0"]), 0.5)
    z0 = st.slider("Z", 0.0, 3.0, float(current["z0"]), 0.1)

    st.subheader("Velocity")
    vx = st.slider("Forward velocity", 10.0, 40.0, float(current["vx"]), 0.5)
    vy = st.slider("Lateral velocity", -10.0, 10.0, float(current["vy"]), 0.5)
    vz = st.slider("Upward velocity", 0.0, 20.0, float(current["vz"]), 0.5)

    st.subheader("Spin")
    spin_side = st.slider(
        "Side spin (RPM)",
        -600.0,
        600.0,
        float(current["spin_side"]),
        10.0,
        disabled=force_mode != FORCE_MODE_FULL,
    )
    spin_top = st.slider(
        "Top spin (RPM)",
        -600.0,
        600.0,
        float(current["spin_top"]),
        10.0,
        disabled=force_mode != FORCE_MODE_FULL,
    )

    st.subheader("Environment")
    cd = st.slider(
        "Drag coefficient (CD)",
        0.1,
        0.6,
        float(current["cd"]),
        0.05,
        disabled=force_mode in [FORCE_MODE_NONE, FORCE_MODE_GRAVITY],
    )
    cl = st.slider(
        "Lift coefficient (CL)",
        0.0,
        0.6,
        float(current["cl"]),
        0.05,
        disabled=force_mode != FORCE_MODE_FULL,
    )
    goal_x = st.slider("Goal distance (m)", 10.0, 40.0, float(current["goal_x"]), 1.0)

# ==============================
# Calculation
# ==============================
omega_x = spin_side * (2 * np.pi / 60)
omega_z = spin_top * (2 * np.pi / 60)
omega_y = 0.0

if force_mode != FORCE_MODE_FULL:
    omega_x = omega_y = omega_z = 0.0

if force_mode in [FORCE_MODE_NONE, FORCE_MODE_GRAVITY]:
    cd_effective = 0.0
    cl_effective = 0.0
elif force_mode == FORCE_MODE_DRAG:
    cd_effective = cd
    cl_effective = 0.0
else:
    cd_effective = cd
    cl_effective = cl

U0 = float(np.sqrt(vx**2 + vy**2 + vz**2))
omega_total = float(np.sqrt(omega_x**2 + omega_y**2 + omega_z**2))
D_r, SP, Re = calculate_dimensionless_numbers(U0, omega_total, cd, cl, force_mode)
trajectory_type, traj_color = classify_trajectory(D_r, SP, force_mode)

x, y, z, is_goal, goal_point = calculate_trajectory(
    x0, y0, z0, vx, vy, vz, cd_effective, cl_effective, omega_x, omega_y, omega_z, goal_x, force_mode
)

if len(x) > 1 and np.min(x) <= goal_x <= np.max(x):
    y_at_goal = float(np.interp(goal_x, x, y))
    z_at_goal = float(np.interp(goal_x, x, z))
    status_text = f"MISS — y={y_at_goal:.2f} m, z={z_at_goal:.2f} m at goal plane"
elif is_goal:
    status_text = "GOAL!"
else:
    status_text = "MISS — Ball does not reach goal plane"

if is_goal:
    status_text = "GOAL!"

angle = float(np.degrees(np.arctan2(vz, vx)))
distance = float(np.max(x) - x0) if len(x) > 0 else 0.0
max_height = float(np.max(z)) if len(z) > 0 else 0.0

# ==============================
# Layout
# ==============================
metric_cols = st.columns(6)
metric_cols[0].metric("Outcome", "Goal" if is_goal else "Miss")
metric_cols[1].metric("Launch speed", f"{U0:.1f} m/s")
metric_cols[2].metric("Launch angle", f"{angle:.1f}°")
metric_cols[3].metric("Spin", f"{omega_total * 60 / (2 * np.pi):.0f} RPM")
metric_cols[4].metric("Distance", f"{distance:.1f} m")
metric_cols[5].metric("Max height", f"{max_height:.1f} m")

left, right = st.columns([1, 2])
with left:
    st.plotly_chart(build_phase_figure(D_r, SP, force_mode), use_container_width=True)
    st.markdown("### Classification")
    st.write(f"**Force model:** {force_mode}")
    st.write(f"**Trajectory type:** {trajectory_type}")
    st.write(f"**Status:** {status_text}")
    st.write(f"**Drag number:** {D_r:.3f}")
    st.write(f"**Spin number:** {SP:.3f}")
    st.write(f"**Reynolds number:** {Re:.0f}")
    st.write(f"**CD used:** {cd_effective:.2f}")
    st.write(f"**CL used:** {cl_effective:.2f}")

with right:
    st.plotly_chart(
        build_trajectory_figure(x, y, z, x0, y0, z0, goal_x, traj_color, goal_point),
        use_container_width=True,
    )

with st.expander("About this web app"):
    st.write(
        "This interactive web app simulates the trajectory of a football during a free kick using real physical models. Users can adjust parameters such as initial velocity, spin, air resistance, and starting position to see how they affect the ball’s path through the air."
        "The simulator models several forces acting on the ball, including gravity, aerodynamic drag, and the Magnus effect caused by spin. These forces are solved numerically to calculate the ball’s motion in three dimensions, allowing realistic curved and dipping shots to be reproduced."
        "The app also classifies each shot using dimensionless parameters such as the drag number and spin number, showing where the kick lies on a phase diagram of different trajectory regimes (parabolic, knuckleball, curved, spiral, etc.)."
        "A 3D visualization displays the ball’s flight path toward the goal, along with indicators showing whether the shot results in a goal or miss. Preset configurations based on famous free kicks allow users to quickly explore different styles of shots."
        "Overall, the simulator demonstrates how physics principles like aerodynamics and rotational forces influence football trajectories, providing both a learning tool and an interactive exploration of free kick dynamics."
    )
