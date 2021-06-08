class Model:
    def __init__(self, log_gamma=None, proposal=None, tag=0):
        self.left = None
        self.right = None
        self.log_gamma = log_gamma
        self.proposal = proposal
        self.depth = 0
        self.tag = tag
        
    def build_left(self, log_gamma, proposal, tag):
        self.left = Model(log_gamma, proposal, tag)
        self.depth += 1

    def build_right(self, log_gamma, proposal, tag):
        self.right = Model(log_gamma, proposal, tag)
        self.depth += 1
        
    def __str__(self):
        return "depth:" + str(self.depth)
