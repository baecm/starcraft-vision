class Event:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def run(self):
        # TODO: implement event preprocessor
        raise NotImplementedError
        