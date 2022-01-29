from __future__ import print_function
import os
import chess
import time
import chess.svg
import traceback
import base64
from state import State

class Valuator(object):
    def __init__(self):
        import torch
        from train import Net
        vals = torch.load('nets/valuator.pth', map_location= lambda storage, loc: storage)
        self.model = Net()
        self.model.load_state_dict(vals)

    def __call__(self, s):
        brd = s.serialize()[None]
        output = self.model(torch.tensor(brd).float())
        return float(output.detach().numpy()[0][0])

# Value function

MAXVALUE = 10000
class ClassicValuator(object):
    # values of pieces on the board
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    def __init__(self):
        self.reset()
        self.memo = {}

    def reset(self):
        self.count = 0

    # a simple value function based on pieces
    def __call__(self, s):
        self.count += 1
        key = s.key()
        if not key in self.memo:
            self.memo[key] = self.value(s)
        return self.memo[key]

    def value(self, s):
        b = s.board
        # when the game is over, return a value based on the winner
        if b.is_game_over():
            # we won, return the maxvalue
            if b.result() == '1-0':
                return MAXVALUE
            # we lost, return -maxvalue
            elif b.result() == '0-1':
                return -MAXVALUE
            # draw, return 0
            else:
                return 0

        val = 0.0
        # add the value of the pieces on the board
        pm = s.board.piece_map()
        for x in pm:
            tval = self.values[pm[x].piece_type]
            if pm[x].color == chess.WHITE:
                val += tval
            else:
                val -= tval


        # add the number of legal moves
        bak = b.turn
        b.turn = chess.WHITE
        val += 0.1 * b.legal_moves.counts()
        b.turn = chess.BLACK
        val -= 0.1 * b.legal_moves.counts()

        b.turn = bak

        return val

def computer_minimax(s, v, depth, alpha, beta, big=False):
    # minimax algorithm
    if depth == 5 or s.is_game_over():
        return v(s)

    # white player is maximixing
    turn = s.board.turn
    if turn == chess.WHITE:
        ret = -MAXVALUE
    else:
        ret = MAXVALUE
    if big:
        bret = []

    # prune with beam search
    isort = []
    for e in s.board.legal_moves:
        s.board.push(e)
        isort.append((v(s), e))
        s.board.pop()
    move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)

    # beam search beyond depth 3
    if depth > 3:
        move = move[:10]

    for e in [x[1] for x in move]:
        s.board.push(e)
        tval = computer_minimax(s, v, depth+1, alpha, beta, big)
        s.board.pop()
        if big:
            bret.append((tval, e))
        if turn == chess.WHITE:
            ret = max(ret, tval)
            a = max(alpha, ret)
            if a >= beta:
                break
        else:
            ret = min(ret, tval)
            b = min(beta, ret)
            if b <= alpha:
                break # alpha cut-off
    if big:
        return ret, bret
    else:
        return ret

def explore_leaves(s, v):
    # explore the leaves of the tree
    ret = []
    start = time.time()
    v.reset()
    bval = v(s)
    eta = time.time() - start
    print("%2f -> %.2f: explored %d nodes in %.2f seconds" % (bval, cval, v.count, eta, int(v.count/eta)))
    return ret

# chess board and "engine"
s = State()
v = ClassicValuator()

def to_svg(s):
    return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')

from flask import Flask, Response, request
app = Flask(__name__)

@app.route('/')
def hello():
    ret = open('index.html').read()
    return ret.replace('start', s.board.fen())

def computer_move(s, v):
    # computer move
    move = sorted(explore_leaves(s,v), key=lambda x: x[0], reverse=s.board.turn)
    if len(move) == 0:
        return
    print("top 3:")
    for i, m in enumerate(move[:3]):
        print("%d: %s" % (i, m[1]))
    print(s.board.turn, "moving", move[0][1])
    s.board.push(move[0][1])

@app.route('/selfplay')
def selfplay():
    s = State()

    ret = '<html><head>'
    # self play
    while not s.board.is_game_over():
        computer_move(s, v)
        ret += '<img src="data:image/svg+xml;base64,%s">' % to_svg(s)
    print(s.board.result())

    return ret

@app.route('/move')
def move():
    if not s.board.is_game_over():
        move = request.args.get('move', default="")
        if move is not None and move != "":
            print("human move", move)
            try:
                s.board.push_san(move)
                computer_move(s, v)
            except Exception:
                traceback
            response = app.response_class(
                response="game over",
                status=200,
            )
            return response
        else:
            print("game is over")
            response = app.response_class(
                response="game over",
                status=200,
            )
            return response
        print("hello function ran")
        return hello()

# moves given as coordinates of a piece moved
@app.route('/move_coordinates')
def move_coordinates():
    if not s.board.is_game_over():
        source = int(request.args.get('from', default=""))
        target = int(request.args.get('to', default=""))
        promotion = True if request.args.get('promotion', default="") == "true" else False

        move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

        if move is not None and move != "":
            print("human move", move)
            try:
                s.board.push_san(move)
                computer_move(s, v)
            except Exception:
                traceback
            response = app.response_class(
                response="game over",
                status=200,
            )
            return response
        else:
            print("game is over")
            response = app.response_class(
                response="game over",
                status=200,
            )
            return response

@app.route('/newgame')
def newgame():
    s = State()
    response = app.response_class(
        response=s.board.fen(),
        status=200,
    )
    return response

if __name__ == '__main__':
    if os.getenv("SelfPlay") is not None:
        s = State()
        while not s.board.is_game_over():
            computer_move(s, v)
            print(s.board)
        print(s.board.result())
    else:
        app.run(debug=True)
