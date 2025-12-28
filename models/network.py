import torch
from models.base import Model

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

        self.count_head = torch.nn.Embedding(14, 1)

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


    @torch.no_grad()
    def bet_strategy(self, state, min_bet, max_bet):
        table_pos = state['table_pos']
        players_table = state['players_table']
        players_play = state['players_play']
        decks = state['decks']
        input_tensor = torch.tensor(
            [table_pos, players_table, players_play, self.count, decks, self.balance, self.min_bet, self.payout_ratio], 
            dtype=torch.float32
        )
        embedding = self.encoder(input_tensor)
        confidence = self.confidence_head(embedding)
        bet_input = torch.cat([input_tensor, confidence.squeeze()], dim=0)
        bet_multi = self.bet_head(bet_input)
        bet = bet_multi.item() * max_bet
        metadata = {'data': bet_input}
        return bet, metadata


    @torch.no_grad()
    def decision_strategy(self, state):
        dealer_upcard = state['dealer_upcard']
        hand = state['hand']
        cards_played = state['cards_played']
        bet_input = state['metadata']['data']
        value, aces = self.value_hand(hand)
        state_tensor = torch.tensor(
            [dealer_upcard, value, aces, cards_played] + bet_input.tolist(), 
            dtype=torch.float32
        )
        return self.policy(state_tensor).argmax().item()


    @torch.no_grad()
    def count_strategy(self, card):
        return self.count_head(torch.tensor(card, dtype=torch.float32)).item()


    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, config):
        model = Network(config)  
        model.load_state_dict(torch.load(path))
        model.eval()
        return model