import warp as wp
import warp.sim
from warp.sim.integrator_euler import integrate_bodies, eval_body_joints, \
    eval_body_contacts


def compute_forces(model, state, joint_act, body_f):
    if model.contact_count > 0 and model.ground:
        wp.launch(kernel=eval_body_contacts,
                  dim=model.contact_count,
                  inputs=[
                      state.body_q,
                      state.body_qd,
                      model.body_com,
                      model.contact_body0,
                      model.contact_point0,
                      model.contact_dist,
                      model.contact_material,
                      model.shape_materials
                  ],
                  outputs=[
                      body_f
                  ],
                  device=model.device)

    wp.launch(kernel=eval_body_joints,
              dim=model.body_count,
              inputs=[
                  state.body_q,
                  state.body_qd,
                  model.body_com,
                  model.joint_q_start,
                  model.joint_qd_start,
                  model.joint_type,
                  model.joint_parent,
                  model.joint_X_p,
                  model.joint_X_c,
                  model.joint_axis,
                  model.joint_target,
                  joint_act,
                  model.joint_target_ke,
                  model.joint_target_kd,
                  model.joint_limit_lower,
                  model.joint_limit_upper,
                  model.joint_limit_ke,
                  model.joint_limit_kd,
                  model.joint_attach_ke,
                  model.joint_attach_kd,
              ],
              outputs=[
                  body_f
              ],
              device=model.device)


class SnakeEuler:
    @staticmethod
    def simulate(model, joint_act, state_in, state_out, dt):
        body_f = state_in.body_f

        compute_forces(model, state_in, joint_act, body_f)

        wp.launch(
            kernel=integrate_bodies,
            dim=model.body_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_in.body_f,
                model.body_com,
                model.body_mass,
                model.body_inertia,
                model.body_inv_mass,
                model.body_inv_inertia,
                model.gravity,
                dt,
            ],
            outputs=[
                state_out.body_q,
                state_out.body_qd
            ],
            device=model.device)

        return state_out
