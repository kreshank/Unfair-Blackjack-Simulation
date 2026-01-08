class Player:
    def __init__(self, 
                 name=None, 
                 balance=1000,
                 max_balance=5000):
        self.name = name or self.__class__.__name__
        self.hand = ()
        self.max_balance = 5 * balance # Guarantee cashout time
        self.balance = balance
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