from .builder import build_hook


class HookPool:

    def __init__(self, hooks, modes, logger):
        self.hooks = []
        self.modes = modes
        self.logger = logger
        self.register_hooks(hooks)

    def register_hook(self, hook_cfg):
        hook = build_hook(hook_cfg)
        if set(hook.modes) & set(self.modes):
            self.hooks.insert(-1, hook)
        else:
            self.logger.warning(
                f'{hook.__class__.__name__} is not in modes {self.modes}')

    def register_hooks(self, hook_cfgs):
        for hook_cfg in hook_cfgs:
            self.register_hook(hook_cfg)

    def fire(self, hook_type, looper):
        for hook in self.hooks:
            getattr(hook, hook_type)(looper)
