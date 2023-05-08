import os
import taichi as ti
from omegaconf import OmegaConf

from phyfu.utils.path_utils import ModulePath


module_path_utils = ModulePath("taichi", "two_balls")
yaml_cfg = OmegaConf.load(module_path_utils.mutate_config_path)

cfg = yaml_cfg

world = OmegaConf.load(os.path.join(module_path_utils.mutate_config_path)).model_def

arch = ti.cuda if cfg.use_gpu else ti.cpu

real = ti.f32
# ti.init(arch=arch, default_fp=real, flatten_if=True)
# real = ti.f64
ti.init(arch=ti.cpu, debug=True, default_fp=real, flatten_if=False,
        advanced_optimization=False, cpu_max_num_threads=1)

max_num_steps = max(cfg.num_steps, cfg.seed_getter.max_steps)
mut_steps = cfg.mut_steps
# epsilon = cfg.epsilon
dt = cfg.dt
# alpha = 0.00000
learning_rate = cfg.lr

vis_interval = 8
output_vis_interval = 8

vis_resolution = 1024

state_size = len(world.init_pos[0])
num_objs = len(world.init_pos)

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(state_size, dtype=real)

loss = scalar()
x = vec()
x_inc = vec()  # for TOI
v = vec()
impulse = vec()
ctrls = vec()
mut_dev = vec()
seed_to_mut = vec()
ref_x = vec()
ref_v = vec()

radius = world.radius
elasticity = world.elasticity

ti.root.dense(ti.i, max_num_steps + 1).dense(ti.j, num_objs).place(x, v, x_inc, impulse)
ti.root.dense(ti.i, max_num_steps).dense(ti.j, num_objs).place(ctrls)
ti.root.dense(ti.i, num_objs).place(mut_dev)
ti.root.dense(ti.i, mut_steps).dense(ti.j, num_objs).place(seed_to_mut)
ti.root.dense(ti.i, num_objs).place(ref_x)
ti.root.dense(ti.i, num_objs).place(ref_v)
ti.root.place(loss)
ti.root.lazy_grad()


num_planes = len(world.plane_pos)
plane_pos = ti.Vector.field(state_size, dtype=real, shape=(num_planes,))
plane_pos[0] = ti.Vector(list(world.plane_pos[0]))
plane_pos[1] = ti.Vector(list(world.plane_pos[1]))

zero_state = ti.Vector([0 for _ in range(state_size)])


@ti.func
def collide_with_plane(t, ball_id, plane_id):
    dist = (x[t, ball_id] + dt * v[t, ball_id]) - plane_pos[plane_id]
    dist_norm = dist.norm()
    if dist_norm < radius:
        dir = ti.Vector.normalized(dist, 1e-6)
        projected_v = dir.dot(v[t, ball_id])

        if projected_v < 0:
            imp = -(1 + world.elasticity) * projected_v * dir
            toi = (dist_norm - world.radius) / ti.min(
                -1e-3, projected_v)  # Time of impact
            x_inc_contrib = ti.min(toi - dt, 0) * imp
            x_inc[t + 1, ball_id] += x_inc_contrib
            impulse[t + 1, ball_id] += imp


@ti.func
def collide_ball_pairs(t, s_id, o_id):
    dist = (x[t, s_id] + dt * v[t, s_id]) - (x[t, o_id] + dt * v[t, o_id])
    dist_norm = dist.norm()
    rela_v = v[t, s_id] - v[t, o_id]
    if dist_norm < 2 * radius:
        dir = ti.Vector.normalized(dist, 1e-6)
        projected_v = dir.dot(rela_v)

        if projected_v < 0:
            imp = -(1 + world.elasticity) * 0.5 * projected_v * dir
            toi = (dist_norm - 2 * world.radius) / ti.min(
                -1e-3, projected_v)  # Time of impact
            x_inc_contrib = ti.min(toi - dt, 0) * imp
            x_inc[t + 1, s_id] += x_inc_contrib
            impulse[t + 1, s_id] += imp

@ti.kernel
def collide(t: ti.i32):
    for i in ti.static(range(num_objs)):
        for j in range(0, i):
            collide_ball_pairs(t, i, j)
    for i in ti.static(range(num_objs)):
        for j in range(i + 1, num_objs):
            collide_ball_pairs(t, i, j)
    for i in ti.static(range(num_objs)):
        for j in range(num_planes):
            collide_with_plane(t, i, j)


@ti.kernel
def advance_w_toi(t: ti.i32):
    for i in ti.static(range(num_objs)):
        v[t, i] = v[t - 1, i] + impulse[t, i] + ctrls[t - 1, i] * dt
        x[t, i] = x[t - 1, i] + dt * v[t, i] + x_inc[t, i]


@ti.kernel
def compute_loss(num_steps: ti.int32):
    for i in range(num_objs):
        x_diff = ti.abs(x[num_steps, i] - ref_x[i])
        for k in ti.static(range(state_size)):
            loss[None] += x_diff[k]
    for i in range(num_objs):
        v_diff = ti.abs(v[num_steps, i] - ref_v[i])
        for k in ti.static(range(state_size)):
            loss[None] += v_diff[k]


def init_states(init_pos, init_vel):
    for k in ti.static(range(num_objs)):
        x[0, k] = ti.Vector(init_pos[k])
        v[0, k] = ti.Vector(init_vel[k])


def fit_to_canvas(p):
    return (p + 2.) / 2.


def forward(num_steps: int):
    interval = vis_interval
    pixel_radius = int(radius * 1024) + 1

    for t in range(1, num_steps + 1):
        collide(t - 1)
        advance_w_toi(t)  # from t - 1 to t


@ti.kernel
def reset_before_forward():
    """
    Reset the simulation states so that the forward simulation can be correct
    :return:
    """
    loss[None] = 0.0
    for t, i in ti.ndrange(impulse.shape[0], impulse.shape[1]):
        impulse[t, i] = zero_state
        x_inc[t, i] = zero_state


@ti.kernel
def mutate_seed_ctrl():
    for i, j in ti.ndrange(mut_steps, num_objs):
        ctrls[i, j] = seed_to_mut[i, j] * (1 + mut_dev[j])


def forward_and_get_grad(num_steps):
    with ti.ad.Tape(loss):
        mutate_seed_ctrl()
        forward(num_steps)
        compute_loss(num_steps)
    return loss[None], mut_dev.grad.to_numpy()
