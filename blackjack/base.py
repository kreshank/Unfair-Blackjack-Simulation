from analytics import GameAnalytics
from dataclasses import dataclass
from typing import Tuple
from player.base import Player

@dataclass
class PlayerHand:
    player: Player
    playing: bool
    at_table: bool
    bet: int
    hand: Tuple
    true_count: int
    unresolved_hands: List[Tuple] # [(value, bet), ...]

class Blackjack:
    def __init__(self, analytics=None, deck_count=6, min_bet=10, payout_ratio=1.5):
        self.analytics = analytics or GameAnalytics()

        self.deck_count = deck_count
        self.player_count = 0
        self.min_bet = min_bet
        self.payout_ratio = payout_ratio

        self.deck = []
        self.players = []
        self.dealer_hand = () # Second is always face down
    
    # Returns True if game is ready to start new round, False if dealer has Blackjack
    def start_round(self):
        if not self.players:
            raise Exception("No players in the game")
        if self.should_shuffle():
            self.shuffle_deck()
            self.burn(cards=1)
        elif self.should_switch_dealers():
            self.burn(cards=1)

        self.deal()

        if self.value_hand(self.dealer_hand)[0] == 21:
            for ph in self.players:
                ph.true_count += ph.player.count(self.dealer_hand[1])
                if ph.playing and self.value_hand(ph.hand)[0] == 21:
                    ph.player.balance += ph.bet
            return False
        return True

    def play_hand(self, ph, split_hand=False):
        if not split_hand and self.value_hand(ph.hand)[0] == 21:
            ph.player.balance += ph.bet * (self.payout_ratio + 1)
        while True:
            state = {}
            decision = ph.player.decide(state)
            if decision == 0:  # Stand
                ph.unresolved_hands.append((self.value_hand(ph.hand)[0], ph.bet))
                break
            elif decision == 1:  # Hit
                ph.hand += (self.draw(),)
                if self.value_hand(ph.hand)[0] > 21:
                    break
            elif decision == 2:  # Double
                if len(ph.hand) == 2 and ph.bet <= ph.player.balance:
                    ph.balance -= ph.bet
                    ph.bet *= 2
                    ph.hand += (self.draw(),)
                    if self.value_hand(ph.hand)[0] > 21:
                        break
                else:
                    raise Exception("Player not permitted to double down.")
            elif decision == 3:  # Split
                if len(ph.hand) == 2 and ph.hand[0] == ph.hand[1]:
                    bet = ph.bet
                    next_hand = (ph.hand[0], self.draw())
                    ph.hand[1] = self.draw()
                    self.play_hand(ph, split_hand=True)
                    ph.bet = bet
                    ph.hand = next_hand 
                    self.play_hand(ph)
            else:
                raise Exception("Invalid decision")

    def play_game(self):
        for ph in self.players:
            if ph.playing and ph.at_table:
                self.play_hand(ph)

        for ph in self.players:
            ph.true_count += ph.player.count(self.dealer_hand[1])
        
        dealer_value = 0
        while dealer_value := self.value_hand(self.dealer_hand)[0] < 17:
            self.dealer_hand += (self.draw(),)

        for ph in self.players:
            if ph.playing:
                for value, bet in ph.unresolved_hands:
                    if value > dealer_value:
                        ph.player.balance += bet * 2
                        
                    elif value == dealer_value:
                        ph.player.balance += bet


    def add_player(self, player):
        self.players.append(PlayerHand(player=player, playing=False, at_table=True, bet=0, hand=()))
    
    def remove_player(self, player_name):
        for ph in self.players:
            if ph.player.name == player_name:
                ph.at_table = False
                ph.playing = False
                return
        raise Exception(f"Player {player_name} not in the game")

    def deal(self):
        state = {}
        for ph in self.players:
            ph.hand = ()
            ph.player.bet(state, self.min_bet)
            if ph.at_table and ph.bet >= self.min_bet:
                ph.hand += (self.draw(),)
                ph.playing = True
                ph.balance -= ph.bet
        self.dealer_hand = (self.draw(),)
        for ph in self.players:
            ph.hand += (self.draw(),)
        self.dealer_hand += (self.draw(hidden=True),)

    def draw(self, hidden=False):
        if self.deck:
            card = self.deck.pop()
            self.analytics.cards_since_last_shuffle += 1
            self.analytics.card_history.append(card)
            self.analytics.card_frequency[card] = self.analytics.card_frequency.get(card, 0) + 1
            for ph in self.players:
                if not hidden:
                    ph.true_count += ph.player.count(card)
            return card
        else:
            raise Exception("The deck is empty")

    def burn(self, cards=1):
        for _ in range(cards):
            if self.deck:
                card = self.deck.pop()
                self.analytics.cards_since_last_shuffle += 1
                for ph in self.players:
                    ph.true_count += ph.player.count(card)

    def value_hand(self, hand):
        value = 0
        aces = 0
        for card in hand:
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

    def should_shuffle(self):
        raise NotImplementedError("Subclasses must implement should_shuffle method")

    def shuffle_deck(self):
        raise NotImplementedError("Subclasses must implement initialize_deck method")
    
    def should_switch_dealers(self):
        raise NotImplementedError("Subclasses must implement should_switch_dealers method")