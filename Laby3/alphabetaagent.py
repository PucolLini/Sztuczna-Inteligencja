from exceptions import AgentException
import copy

class AlphaBetaAgent:
    def __init__(self, my_token, depth=3):
        self.my_token = my_token
        self.depth = depth

    def decide(self, current_game):

        if current_game.who_moves != self.my_token:
            raise AgentException('not my round')

        game = copy.deepcopy(current_game)
        best_move = self.checkEveryMove(game, current_game)

        return best_move

    def checkEveryMove(self, game, current_game):

        alpha = float("-inf")
        beta = float("inf")
        score = float("-inf")
        best_move = None

        for column in game.possible_drops():
            game.drop_token(column)  # Wykonanie ruchu
            new_board = game
            new_score = self.alphabeta(new_board, 0, self.depth - 1, alpha, beta)  # Wywołanie funkcji alphabeta
            if new_score > score:
                score = new_score
                best_move = column
            alpha = max(alpha, score)
            game = copy.deepcopy(current_game)
        return best_move

    def maximize_player(self, board, current_depth, alpha, beta, game):
        score = float("-inf")
        for column in board.possible_drops():
            board.drop_token(column)
            score = max(score, self.alphabeta(board, 0, current_depth - 1, alpha, beta))
            alpha = max(alpha, score)
            board = copy.deepcopy(game)
            if score >= beta:
                break
        return score

    def minimize_player(self, board, current_depth, alpha, beta, game):
        score = float("inf")
        for column in board.possible_drops():
            board.drop_token(column)
            score = min(score, self.alphabeta(board, 1, current_depth - 1, alpha, beta))
            beta = min(beta, score)
            board = copy.deepcopy(game)
            if score <= alpha:
                break
        return score

    def heuristics(self, board):
        score = 0
        for four in board.iter_fours():
            score += sum(1 for field in four if field == self.my_token) ** 2
        return score

    def alphabeta(self, board, which_player_moves, current_depth, alpha, beta):
        if board._check_game_over() or current_depth == 0:
            if board.wins == self.my_token:
                return 1
            elif board.wins is None:
                return self.heuristics(board)
            else:
                return -1

        game = copy.deepcopy(board)

        if which_player_moves == 1:
            return self.maximize_player(board, current_depth, alpha, beta, game)
        else:
            return self.minimize_player(board, current_depth, alpha, beta, game)


