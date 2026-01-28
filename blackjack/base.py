from analytics import GameAnalytics
from dataclasses import dataclass
from typing import Tuple, List, Dict
from player.base import Player
from utils.helpers import value_hand

@dataclass
class PlayerHand:
    player: Player
    playing: bool
    at_table: bool
    bet: int
    hand: Tuple
    metadata: Dict[str, any]
    true_count: int
    unresolved_hands: List[Tuple] # [(value, bet), ...]

class Blackjack:
    def __init__(self, 
                 analytics=None, 
                 deck_count=6, 
                 min_bet=10, 
                 max_bet=1000, 
                 payout_ratio=1.5, 
                 debug_level=0):
        self.analytics = analytics or GameAnalytics()

        self.deck_count = deck_count
        self.player_count = 0
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.payout_ratio = payout_ratio

        self.deck = []
        self.players = []
        self.dealer_hand = () # Second is always face down
        self.debug_level = debug_level
    
    def start_round(self):
        if not self.players:
            raise Exception("No players in the game")
        if self.should_shuffle():
            self.shuffle_deck()
            self.burn(cards=1)
            return "shuffled"
        elif self.should_switch_dealers():
            self.burn(cards=1)
            return "switched_dealer"
        return "continue"

    # Returns True if game should continue, False if dealer has blackjack
    def check_deal(self):
        self.deal()
        if self.debug_level >= 2:
            for ph in self.players:
                if ph.playing:
                    print (f"\tPlayer {ph.player.name} hand:\t", ph.hand)
            print ("Dealer shows:", self.dealer_hand[0])

        if value_hand(self.dealer_hand)[0] == 21:
            if self.debug_level >= 2:
                print ("\tDealer has Blackjack!")
            for ph in self.players:
                ph.true_count += ph.player.update_count(self.dealer_hand[1])
                if ph.playing and value_hand(ph.hand)[0] == 21:
                    if self.debug_level >= 2:
                        print (f"\t\tPlayer {ph.player.name} pushes {ph.bet}")
                    ph.player.balance += ph.bet
                elif self.debug_level >= 2:
                    print (f"\t\tPlayer {ph.player.name} loses {ph.bet}")

            return False
        return True

    def play_hand(self, ph, split_hand=False):
        if self.debug_level >= 2:
            print (f"\tPlayer \"{ph.player.name}\" making decisions...")
        if not split_hand and value_hand(ph.hand)[0] == 21:
            ph.player.balance += ph.bet * (self.payout_ratio + 1)
        while True:
            state = {
                'dealer_upcard': self.dealer_hand[0],
                'hand': ph.hand,
                'bet': ph.bet,
                'metadata': ph.metadata,
                'cards_played' : self.analytics.cards_since_last_shuffle,
                'split_hand': split_hand,
                'can_split': len(ph.hand) == 2 and ph.hand[0] == ph.hand[1],
                'can_double': len(ph.hand) == 2 and ph.bet <= ph.player.balance,
                'count': ph.player.count,
                'balance': ph.player.balance,
                'max_balance': ph.player.max_balance,
                'decks': self.deck_count,
            }
            decision = ph.player.decide(state)
            if self.debug_level >= 2:
                print (f"\t\thand: {ph.hand}, decision: {self.decision_to_str(decision)}")
            if decision == 0:  # Stand
                ph.unresolved_hands.append((value_hand(ph.hand)[0], ph.bet))
                break
            elif decision == 1:  # Hit
                ph.hand += (self.draw(),)
            elif decision == 2:  # Double
                if len(ph.hand) == 2 and ph.bet <= ph.player.balance:
                    ph.player.balance -= ph.bet
                    ph.bet *= 2
                    ph.hand += (self.draw(),)
                    ph.unresolved_hands.append((value_hand(ph.hand)[0], ph.bet))
                    break
                else:
                    raise Exception(f"Player {ph.player.name} not permitted to double down.")
            elif decision == 3:  # Split
                if len(ph.hand) == 2 and ph.hand[0] == ph.hand[1] and ph.bet <= ph.player.balance:
                    bet = ph.bet
                    ph.player.balance -= ph.bet
                    next_hand = (ph.hand[0], self.draw())
                    ph.hand = (ph.hand[1], self.draw())
                    self.play_hand(ph, split_hand=True)
                    ph.bet = bet
                    ph.hand = next_hand 
                    self.play_hand(ph)
            else:
                raise Exception("Invalid decision")
            
            if value_hand(ph.hand)[0] > 21:
                break

    def play_game(self):
        for ph in self.players:
            if ph.playing and ph.at_table:
                self.play_hand(ph)

        for ph in self.players:
            ph.true_count += ph.player.update_count(self.dealer_hand[1])
        
        dealer_value = 0
        while (dealer_value := value_hand(self.dealer_hand)[0]) < 17:
            if self.debug_level >= 2:
                print (f"Dealer hand: {self.dealer_hand}, value: {dealer_value}")
            card = self.draw()
            self.dealer_hand += (card,)
        
        if dealer_value > 21:
            if self.debug_level >= 2:
                print ("\t\tDealer busts!")
            for ph in self.players:
                if ph.playing:
                    for value, bet in ph.unresolved_hands:
                        if self.debug_level >= 2:
                            print (f"\tPlayer {ph.player.name} wins {bet}")
                        ph.player.balance += bet * 2
            return

        if self.debug_level >= 2:
            print (f"Dealer final hand: {self.dealer_hand} value: {dealer_value}")
        for ph in self.players:
            if ph.playing:
                for value, bet in ph.unresolved_hands:
                    if value > dealer_value:
                        if self.debug_level >= 2:
                            print (f"\tPlayer \"{ph.player.name}\" wins {bet}")
                        ph.player.balance += bet * 2
                    elif value == dealer_value:
                        if self.debug_level >= 2:
                            print (f"\tPlayer \"{ph.player.name}\" pushes")
                        ph.player.balance += bet
                    else:
                        if self.debug_level >= 2:
                            print (f"\tPlayer \"{ph.player.name}\" loses {bet}")



    def add_player(self, player):
        self.players.append(
            PlayerHand(
                player=player, 
                playing=False, 
                at_table=True, 
                bet=0, 
                hand=(),
                true_count=0,
                unresolved_hands=[],
                metadata={},
            )
        )
    
    def remove_player(self, player_name):
        for ph in self.players:
            if ph.player.name == player_name:
                ph.at_table = False
                ph.playing = False
                return
        raise Exception(f"Player {player_name} not in the game")

    def deal(self):
        for i, ph in enumerate(self.players):
            state = {
                'table_pos': i,
                'players_table': len([p for p in self.players if p.at_table]),
                'players_play': len([p for p in self.players if p.playing]),
                'decks': self.deck_count,
                'count': ph.player.count,
                'balance': ph.player.balance,
                'max_balance': ph.player.max_balance,
                'payout_ratio': self.payout_ratio,
            }
            ph.hand = ()
            ph.unresolved_hands = []
            ph.bet, ph.metadata = ph.player.bet(state, self.min_bet, self.max_bet)
            if ph.at_table and ph.bet >= self.min_bet:
                if self.debug_level >= 1:
                    print (f"Player \"{ph.player.name}\" ({ph.player.balance}) bets {ph.bet}")
                ph.hand += (self.draw(),)
                ph.playing = True
                ph.player.balance -= ph.bet
            elif ph.bet < 0:
                if self.debug_level >= 1:
                    print (f"Player {ph.player.name} cashes out with {ph.player.balance}")
                ph.at_table = False
                ph.playing = False
                self.remove_player(ph.player.name)
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
                    ph.true_count += ph.player.update_count(card)
            return card
        else:
            raise Exception("The deck is empty")

    def burn(self, cards=1):
        for _ in range(cards):
            if self.deck:
                card = self.deck.pop()
                self.analytics.cards_since_last_shuffle += 1
                for ph in self.players:
                    ph.true_count += ph.player.update_count(card)
    
    def decision_to_str(self, decision):
        if decision == 0:
            return "stand"
        elif decision == 1:
            return "hit"
        elif decision == 2:
            return "double down"
        elif decision == 3:
            return "split"
        else:
            return "Unknown Decision"
        
    def get_state(self, ph):
        state = {
            'deck_count': self.deck_count,
            'min_bet': self.min_bet,
            'payout_ratio': self.payout_ratio,
            'players': [],
            'dealer_hand': self.dealer_hand,
            'deck_size': len(self.deck),
        }
        for ph in self.players:
            state['players'].append({
                'name': ph.player.name,
                'balance': ph.player.balance,
                'hand': ph.hand,
                'playing': ph.playing,
                'at_table': ph.at_table,
                'bet': ph.bet,
                'true_count': ph.true_count,
            })
        return state
    
    def in_game(self, player_name):
        for ph in self.players:
            if ph.player.name == player_name and ph.at_table:
                return True
        return False

    def should_shuffle(self):
        raise NotImplementedError("Subclasses must implement should_shuffle method")

    def shuffle_deck(self):
        raise NotImplementedError("Subclasses must implement initialize_deck method")
    
    def should_switch_dealers(self):
        raise NotImplementedError("Subclasses must implement should_switch_dealers method")