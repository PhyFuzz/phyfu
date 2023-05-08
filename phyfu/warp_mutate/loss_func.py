import warp as wp
from warp.sim import State

from phyfu.common.loss_func import LossFunction


class TwoBallsLossFunc(LossFunction):
    def __init__(self, loss_name):
        super().__init__(loss_name)
        self.loss = wp.empty(1, dtype=wp.float32, requires_grad=True)

    @staticmethod
    @wp.kernel
    def square_reduce(q1: wp.array(dtype=wp.vec3), q2: wp.array(dtype=wp.vec3),
                      qd1: wp.array(dtype=wp.vec3),
                      qd2: wp.array(dtype=wp.vec3),
                      loss: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        q_diff = q1[tid] - q2[tid]
        qd_diff = qd1[tid] - qd2[tid]
        wp.atomic_add(loss, 0, wp.dot(q_diff, q_diff) + wp.dot(qd_diff, qd_diff))

    @staticmethod
    @wp.kernel
    def linear_reduce(q1: wp.array(dtype=wp.vec3), q2: wp.array(dtype=wp.vec3),
                      qd1: wp.array(dtype=wp.vec3),
                      qd2: wp.array(dtype=wp.vec3),
                      loss: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        ones = wp.vec3(1.0, 0.0, 0.0)
        q_diff = q1[tid] - q2[tid]
        qd_diff = qd1[tid] - qd2[tid]
        diff = wp.abs(wp.dot(q_diff, ones)) + wp.abs(wp.dot(qd_diff, ones))
        wp.atomic_add(loss, 0, diff)

    def linear_loss(self, output: State, label: State):
        self.loss.zero_()
        wp.launch(self.linear_reduce, dim=len(output.particle_q),
                  inputs=[output.particle_q, label.particle_q,
                          output.particle_qd, label.particle_qd],
                  outputs=[self.loss])
        return self.loss

    def square_loss(self, output: State, label: State):
        self.loss.zero_()
        wp.launch(self.square_reduce, dim=len(output.particle_q),
                  inputs=[output.particle_q, label.particle_q,
                          output.particle_qd, label.particle_qd],
                  outputs=[self.loss])
        return self.loss

class SnakeLossFunc(LossFunction):
    def __init__(self, loss_name):
        super().__init__(loss_name)
        self.loss = wp.empty(1, dtype=wp.float32, requires_grad=True)

    @staticmethod
    @wp.kernel
    def square_reduce(q1: wp.array(dtype=wp.transform), q2: wp.array(dtype=wp.transform),
                      qd1: wp.array(dtype=wp.spatial_vector),
                      qd2: wp.array(dtype=wp.spatial_vector),
                      loss: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        p1 = wp.transform_get_translation(q1[tid])
        p2 = wp.transform_get_translation(q2[tid])
        r1 = wp.transform_get_rotation(q1[tid])
        r2 = wp.transform_get_rotation(q2[tid])
        p_diff = wp.dot(p1 - p2, p1 - p2)
        r_diff = wp.dot(r1 - r2, r1 - r2)

        w1 = wp.spatial_top(qd1[tid])
        w2 = wp.spatial_top(qd2[tid])
        v1 = wp.spatial_bottom(qd1[tid])
        v2 = wp.spatial_bottom(qd2[tid])

        w_diff = wp.dot(w1 - w2, w1 - w2)
        v_diff = wp.dot(v1 - v2, v1 - v2)

        wp.atomic_add(loss, 0, p_diff + r_diff + w_diff + v_diff)
        # loss[0] = p_diff + r_diff + w_diff + v_diff

    @staticmethod
    @wp.func
    def abs_sum_vec3(v: wp.vec3):
        s = 0.0
        for i in range(3):
            s += wp.abs(v[i])
        return s

    @staticmethod
    @wp.func
    def abs_sum_quat(w: wp.quatf):
        s = 0.0
        for i in range(4):
            s += wp.abs(w[i])
        return s

    @staticmethod
    @wp.kernel
    def linear_reduce(q1: wp.array(dtype=wp.transform), q2: wp.array(dtype=wp.transform),
                      qd1: wp.array(dtype=wp.spatial_vector),
                      qd2: wp.array(dtype=wp.spatial_vector),
                      loss: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        p1 = wp.transform_get_translation(q1[tid])
        p2 = wp.transform_get_translation(q2[tid])
        r1 = wp.transform_get_rotation(q1[tid])
        r2 = wp.transform_get_rotation(q2[tid])
        p_diff = SnakeLossFunc.abs_sum_vec3(p1 - p2)
        r_diff = SnakeLossFunc.abs_sum_quat(r1 - r2)

        w1 = wp.spatial_top(qd1[tid])
        w2 = wp.spatial_top(qd2[tid])
        v1 = wp.spatial_bottom(qd1[tid])
        v2 = wp.spatial_bottom(qd2[tid])

        w_diff = SnakeLossFunc.abs_sum_vec3(w1 - w2)
        v_diff = SnakeLossFunc.abs_sum_vec3(v1 - v2)

        wp.atomic_add(loss, 0, p_diff + r_diff + w_diff + v_diff)
        # loss[0] = p_diff + r_diff + w_diff + v_diff

    def linear_loss(self, output: State, label: State):
        self.loss.zero_()
        wp.launch(self.linear_reduce, dim=len(output.body_q),
                  inputs=[output.body_q, label.body_q, output.body_qd, label.body_qd],
                  outputs=[self.loss])
        return self.loss

    def square_loss(self, output: State, label: State):
        self.loss.zero_()
        wp.launch(self.square_reduce, dim=len(output.body_q),
                  inputs=[output.body_q, label.body_q, output.body_qd, label.body_qd],
                  outputs=[self.loss])
        return self.loss
