# class Model:
#     def __init__(self, log_gamma=None, proposal=None, tag=0, update=True):
#         self.left = None
#         self.right = None
#         self.log_gamma = log_gamma
#         self.proposal = proposal
#         self.depth = 0
#         self.tag = tag
#         self.update = update
        
#     def build_left(self, log_gamma, proposal, tag, update=True):
#         self.left = Model(log_gamma, proposal, tag, update)
#         self.depth += 1

#     def build_right(self, log_gamma, proposal, tag, update=True):
#         self.right = Model(log_gamma, proposal, tag, update)
#         self.depth += 1
        
#     def __str__(self):
#         return "depth:" + str(self.depth)


class Model:
    def __init__(self, log_gamma=None, proposal=None, tag=0, update=True):
        self.children = []
        self.log_gamma = log_gamma
        self.proposal = proposal
        self.seq = [tag]
        self.tag = tag
        self.update = update
        
    def build_child(self, log_gamma, proposal, tag, update=True):
        self.children.append(Model(log_gamma, proposal, tag, update))
        self.seq.append(tag)
        
    def __str__(self):
        return "nodes:" + str(self.seq)
