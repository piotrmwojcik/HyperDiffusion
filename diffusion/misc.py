class module_requires_grad:
    def __init__(self, module, requires_grad=True):
        self.module = module
        self.requires_grad = requires_grad
        self.prev = []

    def __enter__(self):
        for p in self.module.parameters():
            self.prev.append(p.requires_grad)
            p.requires_grad = self.requires_grad

    def __exit__(self, exc_type, exc_value, traceback):
        for p, r in zip(self.module.parameters(), self.prev):
            p.requires_grad = r
