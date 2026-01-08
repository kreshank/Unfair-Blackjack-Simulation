def value_hand(hand):
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