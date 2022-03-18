
__all__ = ['no_op']

class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

