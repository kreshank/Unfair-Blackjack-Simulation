import torch
from models.network import Network
from blackjack.fair import FairBlackjack
from player.default import DefaultPlayer
from player.model import TrainedPlayer
import tqdm
import random
import numpy as np

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
    'count_deltas': [-2, -1, 0, 1, 2],
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
    'debug_level': 0,
}

hyperparams = {
    'epochs': 500,
    'episodes_per_epoch': 201,
    'horizon': 50,
    'lr': 0.003,
    'gamma': 0.98,
    'lam_confidence': 0.05,
    'lam_decision_entropy': -1,
    'lam_count_entropy': 0.01,
    'lam_reward_var': 3,
    'penalty_exit': 0.4,
    'penalty_wong': 0.05,
    'debug': False,
    'num_tests': 50,
    'episode_test_debug': 50,
    'seed': 42,
    'target_hitrate': 0.3
}

def train_network(player: TrainedPlayer, 
                  GameClass, 
                  hpars):
    random.seed(hpars['seed'])
    np.random.seed(hpars['seed'])
    torch.manual_seed(hpars['seed'])

    optimizer = torch.optim.Adam(player.model.parameters(), lr=hpars['lr'])

    for epoch in tqdm.tqdm(range(hpars['epochs']), desc="Training Epochs"):
        for episode in range(hpars['episodes_per_epoch']):
            player.model.train()
            player.model.pg_training = True
            player.model.confs = None
            game = reset_environment(player, GameClass, env_range)
            initial_balance = player.balance
            done = False 
            hands = 0
            hand_log_probs = []
            wins = []
            confidences = []
            while not done and hands < hpars['horizon']:
                game.start_round()
                balance_in = player.balance
                if game.check_deal():
                    game.play_game()

                hands += 1
                done = (player.balance >= player.max_balance 
                        or player.balance <= game.min_bet
                        or not game.in_game(player.name))
                wins.append(torch.sign(torch.tensor(player.balance - balance_in, dtype=torch.float32)))
                confidences.append(player.model.confs)

                reward = (player.balance - initial_balance) / player.max_balance # loss per round
                if player.model.last_bet <= 0:
                    reward -= hpars['penalty_exit']
                elif player.model.last_bet < game.min_bet:
                    reward -= hpars['penalty_wong']
                log_prob = torch.sum(torch.stack(player.model.log_probs))
                hand_log_probs.append((log_prob, reward))
                player.model.log_probs = []

            # baseline
            Gs = []
            G = 0
            for log_prob, reward in reversed(hand_log_probs):
                G = reward + hpars['gamma'] * G
                Gs.append(G)
            Gs = torch.tensor(list(reversed(Gs)), dtype=torch.float32)
            Gs += (initial_balance - player.balance) / player.max_balance
            Gs = (Gs - torch.mean(Gs)) / (torch.std(Gs, unbiased=False) + 1e-8) # norm adv

            loss_pg = torch.zeros(())
            for (log_prob, _), adv in zip(hand_log_probs, Gs):
                loss_pg = loss_pg - log_prob * adv

            loss_reward_var = hpars['lam_reward_var'] * torch.var(torch.tensor([r for _, r in hand_log_probs]), unbiased=False)
            loss_conf = hpars['lam_confidence'] * torch.nn.functional.binary_cross_entropy_with_logits(torch.stack(confidences), (torch.tensor(wins) > 0).float())
            loss_count_ent = hpars['lam_count_entropy'] * torch.mean(torch.stack(player.model.count_ents))
            loss_decision_ent = hpars['lam_decision_entropy'] * torch.mean(torch.stack(player.model.decision_ents))

            loss = loss_pg + loss_conf + loss_count_ent + loss_decision_ent
            loss = loss + loss_reward_var

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            player.model.decision_ents = []
            player.model.count_ents = []

            if episode % hpars['episode_test_debug'] == 0:
                std, mean, winrate = eval_model(player, GameClass, env_range, hpars['num_tests'], hpars['horizon'], hpars['target_hitrate'])
                print (f"Episode {episode}, mean={mean}, std={std}, winrate={winrate}, loss={loss.item()}")
                print (f"    Loss breakdown: PG={loss_pg.item()}, Conf={loss_conf.item()}")
                print (f"    Entropy Loss:   Count={loss_count_ent.item()}, Decision={loss_decision_ent.item()}")
                print (f"                    RVar={loss_reward_var.item()}")

def eval_model(player, GameClass, env_range, num_tests, horizon, target_hitrate):
    player.model.eval()
    player.model.pg_training = False
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
            if i == pos_id - 1:
                game.add_player(player)
            else:
                game.add_player(DefaultPlayer(name=i))

        done = False
        hands = 0
        while not done and hands < horizon:
            game.start_round()
            if game.check_deal():
                game.play_game()
            
            hands += 1
            done = (player.balance >= player.max_balance 
                    or player.balance <= game.min_bet
                    or not game.in_game(player.name))
        pcts.append((player.balance - start_balance) / start_balance)
    
    winrate = torch.sum(torch.tensor(pcts) > target_hitrate) / num_tests
    return torch.std_mean(torch.tensor(pcts)) + (winrate,)
                
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

if __name__ == "__main__":

    model = Network(config)
    player = TrainedPlayer(
        model=model,
        name="NetworkTrainer",
    )
    GameClass = FairBlackjack

    train_network(player, 
                  GameClass, 
                  hyperparams)

    print ("Save?")
    should_save = input()
    if should_save.lower() in ['y', 'yes']:
        print ("Enter save name:")
        file_name = input().strip()
        file_path = f"../saved_models/{file_name}.pt"
        player.model.save(file_path)
        print (f"Model saved to {file_path}")