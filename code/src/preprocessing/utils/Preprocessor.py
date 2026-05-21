from .preprocess import viewport_parallel_utils

from .preprocess import State, Vision, Event, Viewport


class Preprocessor:
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def run(self):
        with State(self.data) as state:
            state.run()
        
        with Vision(self.data) as vision:
            vision.run()

        # # TODO: implement event preprocessor 
        # with Event(self.data) as event:
        #     event.run()
