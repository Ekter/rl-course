# simple tic tac toe
# Ekter

import numpy as np

# import copy

class TicTacModel:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1 # or -1
        self.winner = 0
        self.turns = 0
        self.trees = 0


    def get_player(self) -> int:
        return self.player

    def get_winner(self) -> int | bool:
        return self.winner

    def play(self, x: int, y: int) -> bool:
        if self.board[x, y] == 0:
            self.board[x, y] = self.player
            self.player = -self.player
            self.winner = self.check_winner()
            self.turns += 1
            return True
        return False

    def check_winner(self) -> int | bool:
        for i in range(3):
            if self.board[i, 0] == self.board[i, 1] == self.board[i, 2] != 0:
                return self.board[i, 0]
            if self.board[0, i] == self.board[1, i] == self.board[2, i] != 0:
                return self.board[0, i]
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return self.board[0, 0]
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != 0:
            return self.board[0, 2]
        return False

    def get_board(self) -> np.array:
        return self.board

    def __str__(self) -> str:
        return str(self.board) + "\n" + str(self.player) + "\n" + str(self.winner)

    def __repr__(self)-> str:
        return "/n".join(" ".join(str(cell) for cell in row) for row in self.board)

    def get_turns(self) -> int:
        return self.turns

    def get_possible_moves(self) -> list[tuple[int]]:
        return [(x, y) for x in range(3) for y in range(3) if self.board[x, y] == 0]

    def tree(self):
        if self.winner:
            return 1 if self.winner == 1 else 0
        if self.turns>=9:
            return 0.5
        res = []
        for move in self.get_possible_moves():
            # new = copy.copy(self)
            x, y = move
            self.play(x, y)
            # print(x,y)
            self.trees+=1
            res.append(self.tree())
            self.board[x, y] = 0
            self.player = -self.player
            self.turns-=1
        return sum(res)/len(res)
        # return max(res)



class Policy_Test:
    def __init__(self):
        self.game = TicTacModel()

    def hint(self):
        pass