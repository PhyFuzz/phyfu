import warp as wp
import warp.sim
from warp.sim import State


@wp.kernel
def integrate_particles(x: wp.array(dtype=wp.vec3),
                        v: wp.array(dtype=wp.vec3),
                        f: wp.array(dtype=wp.vec3),
                        w: wp.array(dtype=float),
                        gravity: wp.vec3,
                        dt: float,
                        x_new: wp.array(dtype=wp.vec3),
                        v_new: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]
    f0 = f[tid]

    inv_mass = w[tid]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    # if tid == 0:
    #     print("v0:")
    #     print(v0)
    #     print("f0:")
    #     print(f0)
    v1 = v0 + (f0 * inv_mass + gravity * wp.step(0.0 - inv_mass)) * dt
    x1 = x0 + v1 * dt

    x_new[tid] = x1
    v_new[tid] = v1


@wp.kernel
def collide_with_wall(particle_q: wp.array(dtype=wp.vec3), radius: float, k_n: float,
                      wall_p: wp.array(dtype=wp.float32), wall_n: wp.array(dtype=wp.float32),
                      particle_f: wp.array(dtype=wp.vec3)):
    pid, wid = wp.tid()
    q = particle_q[pid]
    w_p = wall_p[wid]
    w_n = wall_n[wid]

    d = (q[0] - w_p) * w_n
    err = d - radius
    if err > 0:  # no contact
        return

    # perfect elastic collision, no damping, normal force magnitude
    f_n = - err * 2. * k_n
    # print("q:")
    # print(q)
    # print("wall p:")
    # print(wall_p)
    # print("wall n:")
    # print(wall_n)
    # print("d:")
    # print(d)
    # print("err:")
    # print(err)

    # print("pid:")
    # print(pid)
    # print("wid:")
    # print(wid)
    # print("fn:")
    # print(f_n)
    wp.atomic_add(particle_f, pid, wp.vec3(f_n * w_n, 0., 0.))


@wp.kernel
def particle_particle_collision(
        particle_x: wp.array(dtype=wp.vec3),
        radius: float,
        k_n: float,
        # outputs
        particle_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    s_id = tid  # self id
    o_id = 1 - tid  # other id

    x_s = particle_x[s_id]
    x_o = particle_x[o_id]
    d = wp.length(x_s - x_o)
    n = (x_s - x_o) / d
    err = d - radius * 2.0
    if err > 0:  # no contact
        return

    # perfect elastic collision, no damping, normal force magnitude
    f_n = - err * k_n
    wp.atomic_add(particle_f, s_id, n * f_n)


@wp.kernel
def set_particle_f(particle_f: wp.array(dtype=wp.vec3), act: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    particle_f[tid] = wp.vec3(act[tid], 0., 0.)


class TwoBallsEuler:
    @staticmethod
    def simulate(model, act, state_in: State, state_out: State, dt):
        wp.launch(
            kernel=set_particle_f,
            dim=2,
            inputs=[state_in.particle_f, act],
            device=model.device
        )
        wp.launch(
            kernel=particle_particle_collision,
            dim=2,
            inputs=[
                state_in.particle_q,
                model.particle_radius,
                model.customized_kn,
            ],
            outputs=[state_in.particle_f],
            device=model.device,
        )

        wp.launch(kernel=collide_with_wall, dim=(2, 2),
                  inputs=[state_in.particle_q, model.particle_radius,
                          model.customized_kn, model.wall_p, model.wall_n,
                          state_in.particle_f])

        wp.launch(
            kernel=integrate_particles,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                model.particle_inv_mass,
                model.gravity,
                dt
            ],
            outputs=[
                state_out.particle_q,
                state_out.particle_qd
            ],
            device=model.device)

        return state_out
