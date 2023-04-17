from Commands import Commands

class google(Commands):
    def __init__(self):
        self.commands = {
            "Google Search": self.google_search
        }

    def google_search(self, input: str):
        # Just a test
        pass
