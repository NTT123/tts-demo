def os_cache(x):
    return x


def create(*args, **kwargs):
    class T:
        def load_registry(self, *args, **kwargs):
            return None

    return T()
