import torch
from models.network import Network
from blackjack.fair import FairBlackjack
from player.default import DefaultPlayer
from player.model import TrainedPlayer
import tqdm
import random
import numpy as np

def train_network(player: TrainedPlayer, 
                  GameClass, 
                  epochs: int,
                  episodes_per_epoch: int, 
                  horizon: int,
                  alpha: float = 0.001,
                  gamma: float = 0.99,
                  seed = 42,
                  episode_test_debug = 50,
                  num_tests = 50,
                  env_range = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(player.model.parameters(), lr=alpha)

    for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs"):
        for episode in range(episodes_per_epoch):
            player.model.train()
            print ("new episode")
            game = reset_environment(player, GameClass, env_range)
            done = False 
            hands = 0
            hand_log_probs = []
            while not done and hands < horizon:
                game.start_round()
                balance_in = player.balance
                if game.check_deal():
                    game.play_game()
                
                hands += 1
                done = (player.balance >= player.max_balance 
                        or player.balance <= game.min_bet
                        or not game.in_game(player.name))
                
                if player.model._log_probs:
                    reward = (player.balance - balance_in) / player.max_balance # loss per round
                    reward += 0.1 * hands # encourage to play hands
                    log_prob = torch.sum(torch.stack(player.model._log_probs))
                    hand_log_probs.append((log_prob, reward))
                    player.model._log_probs = []

            if hand_log_probs:
                loss = torch.tensor(0)
                G = 0
                for log_prob, reward in reversed(hand_log_probs):
                    G = reward + gamma * G
                    loss = loss - log_prob * G
                loss = loss / len(hand_log_probs) if hand_log_probs else 1


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            

            if episode % episode_test_debug == 0:
                with torch.no_grad():
                    mean,std = eval_model(player, GameClass, env_range, num_tests)
                player.model._log_probs = []
                print (f"Episode {episode}, mean={mean}, std={std}")

def eval_model(player, GameClass, env_range, num_tests):
    player.model.eval()
    pcts = []
    for _ in range(num_tests):
        game = GameClass(analytics=None,
                        deck_count=random.choice(env_range['deck_count']),
                        min_bet=random.choice(env_range['min_bet']),
                        max_bet=random.choice(env_range['max_bet']),
                        payout_ratio=random.choice(env_range['payout_ratio']),
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
        start_balance = player.balance
        for i in range(player_count):
            if i == pos_id:
                game.add_player(player)
            else:
                game.add_player(DefaultPlayer(name=i))

        done = False
        hands = 0
        while not done and hands < 30:
            game.start_round()
            if game.check_deal():
                game.play_game()
            
            hands += 1
            done = (player.balance >= player.max_balance 
                    or player.balance <= game.min_bet
                    or not game.in_game(player.name))
        pcts.append((player.balance - start_balance) / start_balance)
    return torch.std_mean(torch.tensor(pcts))
        
            
def reset_environment(player, GameClass, env_range):
    game = GameClass(analytics=None,
                     deck_count=random.choice(env_range['deck_count']),
                     min_bet=random.choice(env_range['min_bet']),
                     max_bet=random.choice(env_range['max_bet']),
                     payout_ratio=random.choice(env_range['payout_ratio']),
                     debug=env_range['debug'],
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
        if i == pos_id:
            game.add_player(player)
        else:
            game.add_player(DefaultPlayer(name=i))
    return game

if __name__ == "__main__":
    config = {
        'encoder_input_dim': 7,
        'encoder_hidden_dim': 64,
        'encoder_activation': torch.nn.ReLU,
        'state_dim': 10,
        'policy_hidden_dims': [32, 64, 32],
        'policy_activation': torch.nn.ReLU,
        'action_dim': 4,
        'force_play': False,
        'bet_std': 0.1,
    }

    env_range = {
        'analytics': None, 
        'deck_count': [2,3,4,5,6,7,8], 
        'min_bet': [10,25,50], 
        'max_bet': [1000, 1500, 2000], 
        'payout_ratio': [1.5],
        'balance': [500, 1000, 2000],
        'cashout_ratio': [2.5, 5],
        'players_mean': torch.tensor(3, dtype=torch.float32),
        'players_std': torch.tensor(1, dtype=torch.float32),
        'debug': False,
    }

    hyperparams = {
        'epochs': 500,
        'episodes_per_epoch': 200,
        'horizon': 30,
        'alpha': 0.01,
        'gamma': 0.98,
        'debug': False
    }

    start_balance = 1000
    cashout_multiplier = 5

    model = Network(config)
    player = TrainedPlayer(
        model=model,
        name="NetworkTrainer",
        balance=start_balance,
        max_balance=start_balance * cashout_multiplier,
    )
    GameClass = FairBlackjack

    train_network(player, 
                  GameClass, 
                  epochs = hyperparams['epochs'],
                  episodes_per_epoch = hyperparams['episodes_per_epoch'],
                  horizon = hyperparams['horizon'],
                  alpha = hyperparams['alpha'],
                  gamma = hyperparams['gamma'],
                  env_range = env_range)

    print ("Save?")
    should_save = input()
    if should_save.lower() in ['y', 'yes']:
        print ("Enter save name:")
        file_name = input().strip()
        file_path = f"../saved_models/{file_name}.pt"
        player.model.save(file_path)
        print (f"Model saved to {file_path}")