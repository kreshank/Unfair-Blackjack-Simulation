from player.base import Player
from blackjack.base import Blackjack

from player.default import DefaultPlayer
from blackjack.fair import FairBlackjack

def main():
    game = FairBlackjack(debug=True)
    player = DefaultPlayer("1")
    game.add_player(player)

    hands = 100
    for _ in range(hands):
        print (f"--- Starting hand {_ + 1} of {hands} ---")
        if game.start_round():
            game.play_game()

if __name__ == "__main__":
    main()