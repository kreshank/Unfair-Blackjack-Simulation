from player.base import Player

class TrainedPlayer(Player):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def bet(self, state, min_bet, max_bet):
        if self.balance >= self.max_balance:
            return -1, None
        return self.model.bet_strategy(state, min_bet, max_bet)
    
    def decide(self, state):
        return self.model.decision_strategy(state)
    
    def update_count(self, card):
        change = self.model.count_strategy(card)
        self.count += change 
        return change