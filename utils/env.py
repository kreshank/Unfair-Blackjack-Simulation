import random
import torch
from player.default import DefaultPlayer

def reset_environment(player, GameClass, env_range):
    game = GameClass(analytics=None,
                     deck_count=random.choice(env_range['deck_count']),
                     min_bet=random.choice(env_range['min_bet']),
                     max_bet=random.choice(env_range['max_bet']),
                     payout_ratio=random.choice(env_range['payout_ratio']),
                     debug_level=env_range['debug_level'],
                     )
    player_count = torch.round(
        torch.clamp(
            torch.normal(env_range['players_mean'],
                         env_range['players_std']),
            1,
            6)
    ).int().item()
    pos_id = random.randint(1, player_count)
    player.balance = random.choice(env_range['balance'])
    for i in range(player_count):
        if i == pos_id - 1:
            game.add_player(player)
        else:
            game.add_player(DefaultPlayer(name=i))
    return game