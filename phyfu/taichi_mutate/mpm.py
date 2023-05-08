import taichi as ti
import argparse
import os
import math
import numpy as np

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
mu = E
la = E
max_steps = 2500
steps = 1024
gravity = 3.8

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
ref_x, ref_v = vec(), vec()
n_invalid = ti.field(dtype=ti.i32)
default_x = None
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
seed_weights = scalar()
seed_bias = scalar()
mut_weights = scalar()
mut_bias = scalar()

x_avg = vec()

actuation = scalar()
actuation_omega = 20
act_strength = 4


def allocate_fields():
    # ti.root.dense(ti.j, n_actuators).dense(ti.j, n_sin_waves)\
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves))\
        .place(weights, seed_weights, mut_weights)
    ti.root.dense(ti.i, n_actuators).place(bias, seed_bias, mut_bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)

    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.i, n_particles).place(ref_x, ref_v)

    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)
    ti.root.place(n_invalid)

    ti.root.lazy_grad()


@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


def reset_n_invalid():
    n_invalid[None] = 0


def encountered_invalid():
    return n_invalid[None] > 0


def is_state_valid(x_init_to_check):
    reset_n_invalid()
    check_state_validity(x_init_to_check)
    flag = True
    if n_invalid[None] > 0:
        flag = False
    reset_n_invalid()
    return flag


@ti.kernel
def check_state_validity(x_init_to_check: ti.types.ndarray()):
    for i in range(n_particles * dim):
        t = x_init_to_check[i]
        base = ti.cast(t * inv_dx - 0.5, ti.i32)
        if base < 0 or base + 3 >= n_grid:
            n_invalid[None] += 1


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        if base[0] < 0 or base[0] + 2 >= n_grid or base[1] < 0 or base[1] + 2 >= n_grid:
            n_invalid[None] += 1
            base = ti.Vector([0, 0], dt=ti.int32)
            fx = ti.Vector([0, 0], dt=ti.int32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        if base[0] < 0 or base[0] + 2 >= n_grid or base[1] < 0 or base[1] + 2 >= n_grid:
            n_invalid[None] += 1
            base = ti.Vector([0, 0], dt=ti.int32)
            fx = ti.Vector([0, 0], dt=ti.int32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                          2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)


@ti.kernel
def add_mutation(s: ti.i32, m: ti.i32):
    for i in range(n_actuators):
        act, w, b = 0.0, 0.0, 0.0
        for j in ti.static(range(n_sin_waves)):
            if s < m:
                w = seed_weights[i, j] * (1. + mut_weights[i, j])
            else:
                w = seed_weights[i, j]
            act += w * ti.sin(actuation_omega * s * dt + 2 * math.pi / n_sin_waves * j)
        if s < m:
            b = seed_bias[i] * (1. + mut_bias[i])
        else:
            b = seed_bias[i]
        act += b
        actuation[s, i] = ti.tanh(act)


@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])


@ti.kernel
def compute_loss():
    dist = x_avg[None][0]
    loss[None] = -dist


@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)


@ti.ad.grad_replaced
def mut_advance(s, mut_steps):
    clear_grid()
    add_mutation(s, mut_steps)
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(mut_advance)
def mut_advance_grad(s, mut_steps):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    add_mutation.grad(s, mut_steps)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()


@ti.kernel
def compute_mut_loss(num_steps: ti.i32):
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        # contrib = 1.0
        x_diff = ti.abs(x[num_steps, i] - ref_x[i])
        v_diff = ti.abs(v[num_steps, i] - ref_v[i])
        ti.atomic_add(loss[None], contrib * (x_diff[0] + x_diff[1]))
        ti.atomic_add(loss[None], contrib * (v_diff[0] + v_diff[1]))


def loss_and_grads(num_steps, mut_steps):
    loss[None] = 0.0
    with ti.ad.Tape(loss):
        for s in range(num_steps):
            mut_advance(s, mut_steps)
            if encountered_invalid():
                break
        compute_mut_loss(num_steps)
    if encountered_invalid():
        raise RuntimeError("Encountered invalid state")
    grads = []
    for i in range(n_actuators):
        for j in range(n_sin_waves):
            grads.append(mut_weights.grad[i, j])
    for i in range(n_actuators):
        grads.append(mut_bias.grad[i])
    return loss[None], np.array(grads)


@ti.kernel
def copy_init_state(init_x: ti.types.ndarray(), init_v: ti.types.ndarray()):
    for i in range(n_particles):
        for k in ti.static(range(dim)):
            x[0, i][k] = init_x[i, k]
            v[0, i][k] = init_v[i, k]


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles, default_x
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        default_x = np.array(self.x, dtype=np.float32)
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act


def robot(scene):
    scene.set_offset(0.1, 0.03)
    scene.add_rect(0.0, 0.1, 0.3, 0.1, -1)
    scene.add_rect(0.0, 0.0, 0.05, 0.1, 0)
    scene.add_rect(0.05, 0.0, 0.05, 0.1, 1)
    scene.add_rect(0.2, 0.0, 0.05, 0.1, 2)
    scene.add_rect(0.25, 0.0, 0.05, 0.1, 3)
    scene.set_n_actuators(4)


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF, show_gui=False)


def visualize(s, folder, save_name: str = None):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    if save_name is None:
        gui.show(f'{folder}/{s:04d}.png')
    else:
        gui.show(os.path.join(folder, f"{save_name}.png"))


def init_scene():
    # initialization
    scene = Scene()
    robot(scene)
    scene.finalize()
    allocate_fields()

    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]


def main():
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    options = parser.parse_args()

    init_scene()
    for s in range(max_steps):
        for i in range(n_actuators):
            for j in range(n_sin_waves):
                weights[i, j] = np.random.randn() * 0.01

    losses = []
    for iter in range(options.iters):
        with ti.ad.Tape(loss):
            forward()
        l = loss[None]
        losses.append(l)
        print('i=', iter, 'loss=', l)
        learning_rate = 0.1

        for i in range(n_actuators):
            for j in range(n_sin_waves):
                # print(weights.grad[i, j])
                weights[i, j] -= learning_rate * weights.grad[i, j]
            bias[i] -= learning_rate * bias.grad[i]

        # if iter % 100 == 0:
        if iter == 99:
            # visualize
            forward(1500)
            for s in range(15, 1500, 16):
                visualize(s, 'diffmpm/iter{:03d}/'.format(iter))

    # ti.profiler_print()
    plt.title("Optimization of Initial Velocity")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    plt.savefig("grad_mpm.png")
    # plt.show()


if __name__ == '__main__':
    main()
