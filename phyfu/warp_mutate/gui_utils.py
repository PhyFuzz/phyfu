import warp.sim.render


def render(model, num_steps, save_path):
    renderer = warp.sim.render.SimRenderer(model.model, save_path)
    for i in range(0, model.substeps * num_steps, model.substeps):
        render_time = i * model.dt
        renderer.begin_frame(render_time)
        renderer.render(model.states[i])
        renderer.end_frame()

    renderer.save()
