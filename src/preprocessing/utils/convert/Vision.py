class Vision:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def run(self):
        # vision data decoded and converted already in Preprocessor
        pass
        