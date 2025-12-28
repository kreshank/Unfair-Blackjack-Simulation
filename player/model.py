from player.base import Player

class TrainedPlayer(Player):
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def bet(self, state, min_bet, max_bet):
        return self.model.bet_strategy(state, min_bet, max_bet)
    
    def decide(self, state):
        return self.model.decision_strategy(state)
    
    def update_count(self, card):
        change = self.model.count_strategy(card)
        self.count += change 
        return change