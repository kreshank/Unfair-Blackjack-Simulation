import torch
from models.network import Network
from blackjack.fair import FairBlackjack
from player.default import DefaultPlayer
from player.model import TrainedPlayer
from utils.training import reset_environment, eval_model, visualize_stats_realtime
import tqdm
from pathlib import Path
import random
import numpy as np
import argparse

config = {
    'encoder_input_dim': 7,
    'encoder_hidden_dim': 64,
    'encoder_activation': torch.nn.ReLU,
    'state_dim': 10,
    'policy_hidden_dims': [32, 64, 32],
    'policy_activation': torch.nn.ReLU,
    'action_dim': 4,
    'allow_wong': False,
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
    'epochs': 5,
    'batches_per_epoch': 100,
    'episodes_per_batch': 8,
    'episodes_per_epoch': 101,
    'horizon': 30,
    'lr': 0.003,
    'gamma': 0.98,
    'lam_confidence': 0.05,
    'lam_decision_entropy': -10,
    'lam_count_entropy': -10,
    'lam_reward_var': 3,
    'penalty_exit': 0.8,
    'penalty_wong': 0.2,
    'debug': False,
    'num_tests': 100,
    'episode_test_debug': 50,
    'seed': 134,
    'target_hitrate': 0.3,
    'grad_clip_max_norm': 1.0,
}

def train_network(player: TrainedPlayer, 
                  GameClass, 
                  hpars):
    random.seed(hpars['seed'])
    np.random.seed(hpars['seed'])
    torch.manual_seed(hpars['seed'])

    optimizer = torch.optim.Adam(player.model.parameters(), lr=hpars['lr'])

    for epoch in tqdm.tqdm(range(hpars['epochs']), desc="Training Epochs"):
        for batch in tqdm.tqdm(range(hpars['batches_per_epoch']), desc="Training Batches"):
            optimizer.zero_grad()
            batch_loss_pg = torch.zeros(())
            batch_loss_reward_var = torch.zeros(())
            batch_loss_conf = torch.zeros(())
            batch_loss_count_ent = torch.zeros(())
            batch_loss_decision_ent = torch.zeros(())

            player.model.train()

            for episode in range(hpars['episodes_per_batch']):
                game = reset_environment(player, GameClass, env_range)
                initial_balance = player.balance 

                rollout_state = 0 
                hands = 0
                hand_log_probs = []
                wins = []
                confidences = []
                player.model.confs = None

                # Rollout
                while rollout_state == 0 and hands < hpars['horizon']:
                    game.start_round()
                    hand_start_balance = player.balance 

                    if game.check_deal():
                        game.play_game()

                    hands += 1

                    if player.balance >= player.max_balance:
                        rollout_state = 1
                    elif player.balance <= game.min_bet:
                        rollout_state = 2
                    elif player.model.last_bet < 0 or not game.in_game(player.name):
                        rollout_state = 3

                    wins.append(torch.sign(torch.tensor(player.balance - hand_start_balance, dtype=torch.float32)))
                    confidences.append(player.model.confs)

                    reward = (player.balance - hand_start_balance) / player.max_balance
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
                Gs += (player.balance - initial_balance) / player.max_balance
                Gs = (Gs - torch.mean(Gs)) / (torch.std(Gs, unbiased=False) + 1e-8) # norm adv

                loss_pg = torch.zeros(())
                for (log_prob, _), adv in zip(hand_log_probs, Gs):
                    loss_pg = loss_pg - log_prob * adv
                batch_loss_pg = batch_loss_pg + loss_pg

                loss_reward_var = hpars['lam_reward_var'] * torch.var(torch.tensor([r for _, r in hand_log_probs]), unbiased=False)
                batch_loss_reward_var = batch_loss_reward_var + loss_reward_var
                
                loss_conf = hpars['lam_confidence'] * torch.nn.functional.binary_cross_entropy_with_logits(torch.stack(confidences), (torch.tensor(wins) > 0).float())
                batch_loss_conf = batch_loss_conf + loss_conf
                
                loss_count_ent = hpars['lam_count_entropy'] * torch.mean(torch.stack(player.model.count_ents))
                batch_loss_count_ent = batch_loss_count_ent + loss_count_ent
                
                loss_decision_ent = hpars['lam_decision_entropy'] * torch.mean(torch.stack(player.model.decision_ents))
                batch_loss_decision_ent = batch_loss_decision_ent + loss_decision_ent

                player.model.decision_ents = []
                player.model.count_ents = []

            # Batch normalization
            batch_loss_pg /= hpars['episodes_per_batch']
            batch_loss_conf /= hpars['episodes_per_batch']
            batch_loss_count_ent /= hpars['episodes_per_batch']
            batch_loss_decision_ent /= hpars['episodes_per_batch']
            batch_loss_reward_var /= hpars['episodes_per_batch']

            batch_loss = batch_loss_pg + batch_loss_conf + batch_loss_count_ent + batch_loss_decision_ent
            batch_loss = batch_loss + batch_loss_reward_var

            optimizer.zero_grad()
            batch_loss.backward()
            #torch.nn.utils.clip_grad_norm_(player.model.parameters(), hpars['grad_clip_max_norm'])
            optimizer.step()
            
            std, mean, winrate = eval_model(player, GameClass, env_range, hpars['num_tests'], hpars['horizon'], hpars['target_hitrate'])
            print (f"Batch {batch}, mean={mean}, std={std}, winrate={winrate}, loss={batch_loss.item()}")
            print (f"    Loss breakdown: PG={batch_loss_pg.item()}, Conf={batch_loss_conf.item()}")
            print (f"    Entropy Loss:   Count={batch_loss_count_ent.item()}, Decision={batch_loss_decision_ent.item()}")
            print (f"                    RVar={batch_loss_reward_var.item()}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train or evaluate Blackjack agent"
    )

    parser.add_argument(
        "--load", type=str, default=None,
        help="Path to saved model (.pt). If provided, skip training."
    )

    parser.add_argument(
        "--visualize", action="store_true",
        help="Run real-time rollout visualization after loading/training"
    )

    parser.add_argument(
        "--tests", type=int, default=hyperparams['num_tests'],
        help="Number of rollout simulations for visualization"
    )

    parser.add_argument(
        "--horizon", type=int, default=hyperparams['horizon'],
        help="Max hands per rollout"
    )

    parser.add_argument(
        "--target-hitrate", type=float, default=hyperparams['target_hitrate'],
        help="ROI threshold for winrate metric"
    )

    parser.add_argument(
        "--update-every", type=int,default=10,
        help="Update plot every N rollouts"
    )

    args = parser.parse_args()

    model = Network(config)
    player = TrainedPlayer(
        model=model,
        name="NetworkTrainer",
    )

    GameClass = FairBlackjack

    if args.load is not None:
        model_path = Path(args.load).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model from {model_path}")
        player.model.load(model_path, config)
        player.model.eval()
    else:
        train_network(
            player,
            GameClass,
            hyperparams
        )

        print("Save?")
        should_save = input().strip().lower()
        if should_save in ["y", "yes"]:
            print("Enter save name:")
            file_name = input().strip()
            file_path = Path(__file__).resolve().parent.parent / "saved_models" / f"{file_name}.pt"
            player.model.save(file_path)
            print(f"Model saved to {file_path}")

    if args.visualize:
        visualize_stats_realtime(
            player=player,
            GameClass=GameClass,
            env_range=env_range,
            num_tests=args.tests,
            horizon=args.horizon,
            target_hitrate=args.target_hitrate,
            update_every=args.update_every,
        )