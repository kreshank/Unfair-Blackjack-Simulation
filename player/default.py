from player.base import Player

class DefaultPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        self.pair_splitting = [ # (split card, dealer upcard)
            [False]*14, # placehold
            [True]*14, # A
            [False]*4 + [True]*4 + [False]*6, # 2
            [False]*4 + [True]*4 + [False]*6, # 3
            [False]*14, # 4
            [False]*14, # 5
            [False]*3 + [True]*4 + [False]*7, # 6
            [False]*2 + [True]*6 + [False]*6, # 7
            [True]*14, # 8
            [False]*2 + [True]*5 + [False] + [True]*2 + [False]*4, # 9
            [False]*14, # 10
            [False]*14, # 11
            [False]*14, # 12
            [False]*14, # 13
        ]
        self.soft_totals = [ # (non ace card, dealer upcard)
            [0]*14, # placehold
            [0]*14, # A placehold
            [1]*5 + [2]*2 + [1]*7, # 2
            [1]*5 + [2]*2 + [1]*7, # 3
            [1]*4 + [2]*3 + [1]*7, # 4
            [1]*4 + [2]*3 + [1]*7, # 5
            [1]*3 + [2]*4 + [1]*7, # 6
            [1]*2 + [2]*5 + [0]*2 + [1]*5, # 7
            [0]*6 + [2] + [0]*7, # 8
            [0]*14, # 9
        ]
        self.hard_totals = [ # (hand total, dealer upcard)
            [1]*14, # placehold
            [1]*14, # placehold
            [1]*14, # 2
            [1]*14, # 3
            [1]*14, # 4
            [1]*14, # 5
            [1]*14, # 6
            [1]*14, # 7
            [1]*14, # 8
            [1]*3 + [2]*4 + [1]*7, # 9
            [0]*2 + [2]*8 + [1]*4, # 10
            [2]*14, # 11
            [1]*4 + [0]*3 + [1]*7, # 12
            [1]*2 + [0]*5 + [1]*7, # 13
            [1]*2 + [0]*5 + [1]*7, # 14
            [1]*2 + [0]*5 + [1]*7, # 15
            [1]*2 + [0]*5 + [1]*7, # 16
            [0]*14, # 17
            [0]*14, # 18
            [0]*14, # 19
            [0]*14, # 20
        ]

    def bet(self, state, min_bet):
        return min_bet
    
    # 0 = stand, 1 = hit, 2 = double down, 3 = split
    def decide(self, state):
        dealer_upcard = state['dealer_upcard']
        if len(self.hand) == 2 and self.hand[0] == self.hand[1]:
            if self.pair_splitting[self.hand[0]][dealer_upcard]:
                return 3
        value, aces = self.value_hand()
        if aces == 1:
            return self.soft_totals[value - 11][dealer_upcard]
        return self.hard_totals[value][dealer_upcard]
    
    def count(self, card):
        if card in [2, 3, 4, 5, 6]:
            self.count += 1
        elif card in [1, 10]:
            self.count -= 1
        return self.count