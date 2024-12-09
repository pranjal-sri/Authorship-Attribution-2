class Tree:
    def __init__(self):
        self.start, self.end = -1, -1 # store the start and end indices of the text span
        self.text = None # store the text of the node
        self.left, self.right = None, None # store the left and right children
        self.parent = None # store the parent node
        self.children = set()