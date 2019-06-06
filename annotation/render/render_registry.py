RENDER_REGISTRY = {}

def register_render(render_name):
	def inner_func(func):
		RENDER_REGISTRY[render_name] = func
		return func
	return inner_func

def select_render(render_name):
	return RENDER_REGISTRY.get(render_name, None)