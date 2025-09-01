import numpy as np
import random

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        actions = np.where(valids)[0]
        if len(actions) == 0:
            print("error no legal action")
        return np.random.choice(actions)

class HumanMutsPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        
        while True:
            print("Enter your move (row col)")
            try:
                row, col = map(int, input().split())
                action = row * self.game.getBoardSize()[0] + col
                
                if valid[action]:
                    return action
                else:
                    print("Invalid move! Try again.")
            except:
                print("Invalid input! Please enter row and column as numbers.")


class GreedyMutsPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        
        for action in range(self.game.getActionSize()):
            if valids[action] == 0:
                continue
                
            next_board, _ = self.game.getNextState(board, 1, action)
            score = self.evaluate_board(next_board, 1)
            candidates.append((score, action))
        
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        
        print("error no legal action")
        return -1

    def evaluate_board(self, board, player):
        n = self.game.getBoardSize()[0]
        score = 0

        for x in range(n):
            for y in range(n):
                piece = board[x][y]
                if piece != 0:
                    piece_sign = np.sign(piece)
                    piece_value = abs(piece)
                    
                    if piece_sign == player:
                        score += piece_value * 10
                        
                        if piece_value == 3:
                            score += 5 

                        center_distance = abs(x - n/2) + abs(y - n/2)
                        score += (n - center_distance)
                        
                    else:
                        score -= piece_value * 10

                        if piece_value == 3:
                            score -= 15 

        red_count = np.sum(np.sign(board) == 1)
        yellow_count = np.sum(np.sign(board) == -1)
        
        if player == 1:
            if yellow_count == 0 and red_count > 1:
                return 10000
            elif red_count == 0 and yellow_count > 1:
                return -10000 
        else:
            if red_count == 0 and yellow_count > 1:
                return 10000 
            elif yellow_count == 0 and red_count > 1:
                return -10000 
        return score