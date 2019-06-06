import annotation.render.render
from .render_registry import select_render

def select_render(name):
	return render_registry.select_render(name)




