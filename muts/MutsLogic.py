'''
Board class for YourGame.
Board data:
  1=player1 piece, -1=player2 piece, 0=empty
  
Squares are stored in a 2D list, indexed by (row, col)
'''
import numpy as np
from collections import deque

class Board():
    
    def __init__(self, n=None):
        self.n = n or 4 
        self.pieces = np.zeros((self.n, self.n))        
        # self.pieces[2][1] = -3

    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self, color):
        moves = []
    
        for row in range(self.n):
            for col in range(self.n):
                if self.is_valid_move(row, col, color):
                    moves.append((row, col))
        return moves

    def has_legal_moves(self, color):
        return len(self.get_legal_moves(color)) > 0

    def is_valid_move(self, row, col, color):
        piece = self.pieces[row][col]        
        if piece == 0 or self.check_color(piece, color):
            return True
        return False

    def execute_move(self, move, color):
        row, col = move

        self.pieces[row][col] += color        
        self.pieces = self.debacle(self.pieces, color)

    def count_diff(self, color):
        count = 0
        for row in range(self.n):
            for col in range(self.n):
                if self.pieces[row][col] == color:
                    count += 1
                elif self.pieces[row][col] == -color:
                    count -= 1
        return count

    def get_winner(self):
        red = [(i, j) for i in range(self.n) for j in range(self.n)
            if np.sign(self.pieces[i][j]) == 1]
        yellow = [(i, j) for i in range(self.n) for j in range(self.n)
            if np.sign(self.pieces[i][j]) == -1]

        if len(red) == 0:
            if len(yellow) == 1 or len(yellow) == 0:
                return 0
            return -1
        elif len(yellow) == 0:
            if len(red) == 1 or len(red) == 0:
                return 0
            return 1
        return 0

    def check_color(self, piece, color):
        return piece * color > 0

    def debacle(self, pieces, color):
        queue = deque([(x, y) for x in range(self.n) for y in range(self.n) if abs(pieces[x][y]) == 4])
        directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]

        while queue:
            x, y = queue.popleft()
            pieces[x][y] = 0

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.n and 0 <= ny < self.n:
                    piece = pieces[nx][ny]
                    sign = np.sign(piece) if piece != 0 else 1
                    pieces[nx][ny] = sign * (piece + sign) * color

                    if abs(pieces[nx][ny]) == 4:
                        queue.append((nx, ny))
        return pieces


