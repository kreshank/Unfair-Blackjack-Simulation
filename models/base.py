class Model():
    def bet_strategy(self, state, min_bet, max_bet):
        raise NotImplementedError

    def decision_strategy(self, state):
        raise NotImplementedError

    def count_strategy(self, card):
        raise NotImplementedError
    
    def save(self, path):
        raise NotImplementedError 

    def load(path):
        raise NotImplementedError