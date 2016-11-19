# A dummy event for neutering profiling code
class DummyEvent(object):
    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass

    def begin(self):
        pass

    def end(self):
        pass
