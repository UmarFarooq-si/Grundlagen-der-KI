from typing import List, Optional, Tuple

Player = str  # 'X' oder 'O'
Board = List[str]  # Länge 9, Felder: 'X', 'O', ' '

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),
             (0,3,6),(1,4,7),(2,5,8),
             (0,4,8),(2,4,6)]

def winner(b: Board) -> Optional[Player]:
    for a, c, d in WIN_LINES:
        if b[a] != ' ' and b[a] == b[c] == b[d]:
            return b[a]
    return None

def terminal(b: Board) -> bool:
    return winner(b) is not None or all(c != ' ' for c in b)

def utility(b: Board, player_max: Player='X') -> int:
    w = winner(b)
    if w is None: return 0
    return 1 if w == player_max else -1

def moves(b: Board) -> List[int]:
    return [i for i, c in enumerate(b) if c == ' ']

def apply(b: Board, i: int, p: Player) -> Board:
    nb = b[:]
    nb[i] = p
    return nb

def other(p: Player) -> Player:
    return 'O' if p == 'X' else 'X'

# 1) Reines Minimax mit Knotenzählung
def minimax(b: Board, p: Player='X') -> Tuple[int, int]:
    nodes = 1
    if terminal(b):
        return utility(b, 'X'), nodes
    if p == 'X':
        best = -10
        for i in moves(b):
            val, n = minimax(apply(b, i, p), other(p))
            nodes += n
            if val > best:
                best = val
        return best, nodes
    else:
        best = 10
        for i in moves(b):
            val, n = minimax(apply(b, i, p), other(p))
            nodes += n
            if val < best:
                best = val
        return best, nodes

# 2) Alpha-Beta mit einfacher Zugordnung
PREF = [4, 0, 2, 6, 8, 1, 3, 5, 7]  # Zentrum, Ecken, Kanten
def ordered_moves(b: Board) -> List[int]:
    avail = set(moves(b))
    return [i for i in PREF if i in avail]

def alphabeta(b: Board, p: Player='X', alpha: int=-10, beta: int=10) -> Tuple[int, int]:
    nodes = 1
    if terminal(b):
        return utility(b, 'X'), nodes
    if p == 'X':
        best = -10
        for i in ordered_moves(b):
            val, n = alphabeta(apply(b, i, p), other(p), alpha, beta)
            nodes += n
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best, nodes
    else:
        best = 10
        for i in ordered_moves(b):
            val, n = alphabeta(apply(b, i, p), other(p), alpha, beta)
            nodes += n
            if val < best:
                best = val
            if best < beta:
                beta = best
            if alpha >= beta:
                break
        return best, nodes

# Beispielaufrufe:
# start = [' '] * 9
# v1, n1 = minimax(start, 'X')
# v2, n2 = alphabeta(start, 'X')
# print(v1, n1, v2, n2)
