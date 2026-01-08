from blackjack.base import Blackjack
from numpy import random

class FairBlackjack(Blackjack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def should_shuffle(self):
        if len(self.deck) <= self.deck_count * 52 * 0.25:
            return True
        return False

    def shuffle_deck(self):
        self.deck = []
        for _ in range(self.deck_count):
            for card_value in range(1, 14):
                for _ in range(4):
                    self.deck.append(card_value)
        random.shuffle(self.deck)
        self.analytics.cards_since_last_shuffle = 0
    
    def should_switch_dealers(self):
        return False