class Player:
    def __init__(self, name):
        self.name = name
        self.hands = ()
        self.balance = 1000 
        self.count = 0

    # Bet of 0 means sit out, bet must be greater than or equal to min_bet
    # Bet < 1 means leave table
    def bet(self, state, min_bet):
        raise NotImplemented
    
    # 0 = stand, 1 = hit, 2 = double down, 3 = split
    def decide(self, state):
        raise NotImplemented
    
    # Returns count adjustment for given card
    def count(self, card):
        raise NotImplemented
    
    def value_hand(self):
        value = 0
        aces = 0
        for card in self.hand:
            if card >= 10:
                value += 10
            elif card == 1:
                aces += 1
                value += 11
            else:
                value += card
        while value > 21 and aces:
            value -= 10
            aces -= 1
        return value, aces