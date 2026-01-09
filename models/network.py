import torch
from models.base import Model
from utils.helpers import value_hand

class Network(torch.nn.Module, Model):
    def __init__(self, config):
        super(Network, self).__init__()

        encoder_input_dim = config['encoder_input_dim']
        encoder_hidden_dim = config['encoder_hidden_dim']
        encoder_activation = config['encoder_activation']
        state_dim = config['state_dim']
        policy_hidden_dims = config['policy_hidden_dims']
        policy_activation = config['policy_activation']
        action_dim = config['action_dim']

        self.pg_training = config.get('pg_training', False)
        self.config = config
        self.count_deltas = torch.tensor(config['count_deltas'], dtype=torch.long)
        self.count_head = torch.nn.Embedding(14, len(self.count_deltas))

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_input_dim, encoder_hidden_dim),
            encoder_activation(),
        )

        self.confidence_head = torch.nn.Sequential(
            torch.nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
            encoder_activation(),
            torch.nn.Linear(encoder_hidden_dim, 1),
        )

        self.bet_head = torch.nn.Sequential(
            torch.nn.Linear(encoder_input_dim + 1, encoder_hidden_dim),
            encoder_activation(),
            torch.nn.Linear(encoder_hidden_dim, 1),
            torch.nn.Tanh(), # to scale output from -1 to 1
        )

        layers = []
        idim = state_dim + 2
        for h_dim in policy_hidden_dims:
            layers.append(torch.nn.Linear(idim, h_dim))
            layers.append(policy_activation())
            idim = h_dim
        layers.append(torch.nn.Linear(idim, action_dim))

        self.policy = torch.nn.Sequential(*layers)

        self._log_probs = []
        self._confs = None
        self._entropies = []
        self.last_bet = 0.0

    def bet_strategy(self, state, min_bet, max_bet):
        input_tensor = torch.tensor([
            state['table_pos'],
            state['players_table'],
            state['players_play'],
            state['count'] / state['decks'],
            state['balance'] / state['max_balance'],
            min_bet / max_bet,
            state['payout_ratio'],
        ], dtype=torch.float32).unsqueeze(0)

        embedding = self.encoder(input_tensor)
        confidence = self.confidence_head(embedding)
        
        if self.pg_training:
            self._confs = confidence.squeeze()

        bet_input = torch.cat([input_tensor, confidence], dim=1)
        
        bet_mean = self.bet_head(bet_input)
        bet_std = self.config['bet_std']

        dist = torch.distributions.Normal(bet_mean, bet_std)
        if self.pg_training:
            bet_multi = dist.rsample().clamp(-1, 1)
        else:
            bet_multi = bet_mean.clamp(-1, 1)

        if self.pg_training:
            self._log_probs = [dist.log_prob(bet_multi).squeeze()]

        bet = torch.clamp(bet_multi * max_bet, min=(min_bet if self.config['force_play'] else -1),max=state['balance'])
        metadata = {'data': bet_input.squeeze().detach()}
        self.last_bet = bet.item()
        return bet.item(), metadata

    def get_policy(self, state):
        bet_input = state['metadata']['data'].detach().clone()
        value, aces = value_hand(state['hand'])
        state_tensor = torch.tensor([
            state['dealer_upcard'], 
            value, 
            aces, 
            state['cards_played']
        ], dtype=torch.float32)
        # update bet_input with dynamic values
        bet_input = bet_input.clone()
        bet_input[3] = state['count'] / state['decks']
        bet_input[4] = state['balance'] / state['max_balance']
        input_tensor = torch.cat([bet_input, state_tensor])

        logits = self.policy(input_tensor)
        mask = torch.ones(logits.shape, dtype=torch.bool)
        mask[2] = state['can_double']
        mask[3] = state['can_split']
        logits += (~mask) * -1e9  # large negative number to mask
        return logits
    
    def decision_strategy(self, state):
        policy_logits = self.get_policy(state)
        if self.pg_training:
            dist = torch.distributions.Categorical(logits=policy_logits)
            action = dist.sample()
            self._log_probs.append(dist.log_prob(action).squeeze())
        else:
            action = torch.argmax(policy_logits)
        return action

    def count_strategy(self, card):
        logits = self.count_head(torch.tensor(card, dtype=torch.long))
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        if self.pg_training:
            self._log_probs.append(dist.log_prob(idx).squeeze())
            self._entropies.append(dist.entropy().squeeze())
        return self.count_deltas[idx].item()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, config):
        self.model = Network(config)  
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        return self.model

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)

    def handle_shuffle(self):
        pass

"""
Rationale: 

We use an policy-gradient design to learn three strategies:
  1. Effective Card counting
  2. Betting Aggressiveness
  3. Gameplay Strategy (depending on count strategy)

Punish entropy in counting strategy so we have a bigger "peak" in counting distribution
Want to correlate "confidence" to higher bets to higher win rate
"""