# Unfair Blackjack Simulations
In casinos, we can notice that blindly playing by the "book" often leads to catostrophic losses, much lower than the theoretical winning percentage. 
This repository provides two things: a game Environment (different settings of Blackjack, with options for non-random shuffling), and a player Agent
which takes the place of a player. 

## Hypotheses
My personal hypothesis is that casinos often do a "rigged" shuffling that artifically groups aces and lower rank cards together, thus effectively eliminating
the player's chance of getting a natural blackjack. Mathematically, this greatly reduces the player's odds by quite a significant margin. Some other possibilities
(but less likely) is an unfair distribution of cards or burning cards frequently to either throw off player's count or alter card sequence.

This simulator allows for all of these settings to be tested and analytic data will be shown.

Notably with the TrainedPlayer class -- this is a trainable neural network that will learn 1) a card counting strategy, 2) betting strategy depending on
running count, and 3) per-hand decision strategy. The card counting strategy can also be overridden with a fixed strategy if desired (Hi-Lo, K-O, etc.).
