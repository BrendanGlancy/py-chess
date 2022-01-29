import chess
import numpy as np

# I have to go you can have this code good luck
class State(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def key(self):
        return (self.board.fen(), self.board.turn, self.board.castling_rights, self.board.ep_square)

    def serialize(self):
        state = np.zeros((8,8,5))

        # 257 bits according to readme
        pp = self.board.shredder_fen()
        return pp

    def edges(self):
        return list(self.board.legal_moves)

    def value(self):
        # TODO: add neural net here
        return 1

if __name__ == "__main__":
    s = State()
    print(s.edges())
