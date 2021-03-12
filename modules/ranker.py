class Ranker(object):
    def __init__(self):
        pass

    def rank(self, graph):
        raise NotImplementedError

    def get_proximities(self, query_node):
        raise NotImplementedError


class DiffusionRanker(Ranker):
    def __init__(self):
        pass
