class Player:
    def __init__(self, name):
        self.name = name
        self.hand = ()
        self.balance = 1000 
        self.count = 0

    # Bet of 0 means sit out, bet must be greater than or equal to min_bet
    # Bet < 0 means leave table
    # metadata is a dict of any info passed to decision making
    # returns bet, metadata
    def bet(self, state, min_bet, max_bet):
        raise NotImplementedError
    
    # 0 = stand, 1 = hit, 2 = double down, 3 = split
    def decide(self, state):
        raise NotImplementedError
    
    # Returns count adjustment for given card
    def update_count(self, card):
        raise NotImplementedError
    
    def value_hand(self):
        value = 0
        As = 0
        for card in self.hand:
            if card >= 10:
                value += 10
            elif card == 1:
                As += 1
                value += 11
            else:
                value += card
        aces = As
        while value > 21 and As:
            value -= 10
            As -= 1
        return value, aces