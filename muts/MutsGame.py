from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .MutsLogic import Board
import numpy as np

class MutsGame(Game):    
    def __init__(self, n=None):
        self.n = n or 4
        self.board_size = (self.n, self.n)
    
    def getInitBoard(self):
        b = Board(self.n)
        return np.array(b.pieces)
    
    def getBoardSize(self):
        return self.board_size
    
    def getActionSize(self):
        return self.n * self.n
    
    def getNextState(self, board, player, action):
        b = Board(self.n)
        b.pieces = np.copy(board)

        move = (action // self.n, action % self.n)
        b.execute_move(move, player)

        return (b.pieces, -player)
    
    def getValidMoves(self, board, player):
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legal_moves = b.get_legal_moves(player)
        
        for row, col in legal_moves:
                action = row * self.n + col
                valids[action] = 1
        
        return np.array(valids)
    
    def getGameEnded(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        
        winner = b.get_winner()
        if winner != 0:
            return winner

        return 0
    
    def getCanonicalForm(self, board, player):
        return player * board
    
    def getSymmetries(self, board, pi):
        assert(len(pi) == self.n * self.n)
        pi_board = np.reshape(pi, (self.n, self.n))
        l = []

        l.append((board, pi))
        
        for i in range(1, 4): 
            rotated_board = np.rot90(board, i)
            rotated_pi = np.rot90(pi_board, i)
            l.append((rotated_board, rotated_pi.flatten()))
        
        flipped_h_board = np.fliplr(board)
        flipped_h_pi = np.fliplr(pi_board)
        l.append((flipped_h_board, flipped_h_pi.flatten()))

        flipped_v_board = np.flipud(board)
        flipped_v_pi = np.flipud(pi_board)
        l.append((flipped_v_board, flipped_v_pi.flatten()))

        flipped_d1_board = np.transpose(board)
        flipped_d1_pi = np.transpose(pi_board)
        l.append((flipped_d1_board, flipped_d1_pi.flatten()))

        flipped_d2_board = np.rot90(np.transpose(board), 2) 
        flipped_d2_pi = np.rot90(np.transpose(pi_board), 2)
        l.append((flipped_d2_board, flipped_d2_pi.flatten()))
        
        return l
    
    def stringRepresentation(self, board):
        #return board.tostring()
        return board.tobytes()
    
    @staticmethod
    def display(board):
        n = board.shape[0]
        for x in range(n):
            row = ' '.join(f"{' ' if np.sign(board[x][y]) != -1 else ''}{str(int(board[x][y]))}" for y in range(n))
            print(row)
        print("\n\n")