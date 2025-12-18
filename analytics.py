class GameAnalytics:
    def __init__(self):
        self.games_played = 0
        self.player_wins = 0
        self.dealer_wins = 0
        
        self.busts = 0
        self.house_loss_total = 0
        self.cards_since_last_shuffle = 0

        # Frequency and history (as known by players)
        self.card_frequency = {}
        self.card_history = []