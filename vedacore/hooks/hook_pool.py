from .builder import build_hook


class HookPool:
    def __init__(self, hooks):
        self.hooks = []
        self.register_hooks(hooks)

    def register_hook(self, hook_cfg):
        hook = build_hook(hook_cfg)
        self.hooks.insert(-1, hook)

    def register_hooks(self, hook_cfgs):
        for hook_cfg in hook_cfgs:
            self.register_hook(hook_cfg)

    def fire(self, hook_type, looper):
        for hook in self.hooks:
            getattr(hook, hook_type)(looper)
