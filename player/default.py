class DefaultPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def bet(self, state, min_bet):
        raise NotImplemented
    
    # 0 = stand, 1 = hit, 2 = double down, 3 = split
    def decide(self, state):
        raise NotImplemented
    
    def count(self, card):
        if card in [2, 3, 4, 5, 6]:
            self.count += 1
        elif card in [1, 10]:
            self.count -= 1
        return self.count