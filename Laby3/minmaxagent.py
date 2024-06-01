import copy
from math import inf

from exceptions import AgentException

class MinMaxAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def score(self, game):
        if game.game_over:
            if game.wins == self.my_token:
                return 1
            elif game.wins is None:
                return 0
            else:
                return -1
        return 0


    @staticmethod
    def makeMove(game, move):
        board_copy = copy.deepcopy(game)
        board_copy.drop_token(move)
        return board_copy

    def maximize_player(self, game, current_depth):
        max_score = -inf
        best_move = None

        for move in game.possible_drops():
            board_copy = self.makeMove(game, move)
            _, score = self.minmax(board_copy, current_depth - 1, False)

            if score > max_score:
                max_score = score
                best_move = move

        return best_move, max_score

    def minimize_player(self, game, current_depth):
        min_score = inf
        best_move = None

        for move in game.possible_drops():
            board_copy = self.makeMove(game, move)
            _, score = self.minmax(board_copy, current_depth - 1, True)

            if score < min_score:
                min_score = score
                best_move = move

        return best_move, min_score


    def minmax(self, game, depth, player):
        # d = 0
        if depth == 0 or game.game_over:
            return None, self.score(game)
        if player == 1:  # Player's turn (maximizing player)
            return self.maximize_player(game, depth - 1)
        else:  # Opponent's turn (minimizing player)
            return self.minimize_player(game, depth - 1)

    def decide(self, game):
        if game.who_moves != self.my_token:
            raise AgentException('not my round')
        else:
            best_move, _ = self.minmax(game, depth=4, player=True)
            return best_move
