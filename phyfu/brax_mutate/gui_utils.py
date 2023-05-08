from IPython.display import display, HTML
from brax.io import html
import brax

from phyfu.brax_mutate import model_loader


def viz_history(brax_model: model_loader.BraxModel, history):
    trace = [brax.QP(pos=p, vel=v, rot=r, ang=a) for p, v, r, a in
             zip(history.pos, history.vel, history.rot, history.ang)]
    display(HTML(html.render(brax_model.world, trace)))
