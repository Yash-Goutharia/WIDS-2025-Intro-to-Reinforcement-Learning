import numpy as np

N = 4
GAMMA = 0.95
THETA = 1e-4

ACTIONS = {
    "UP":    (-1,  0),
    "DOWN":  ( 1,  0),
    "LEFT":  ( 0, -1),
    "RIGHT": ( 0,  1)
}

GOAL = np.array([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12],
    [13, 14, 15,  0]
])

def is_goal(board):
    return np.array_equal(board, GOAL)


def row_solved(board, r):
    return np.array_equal(board[r], GOAL[r])


def find_blank(board):
    return tuple(np.argwhere(board == 0)[0])


def valid_move(board, action, locked_rows):
    x, y = find_blank(board)
    dx, dy = ACTIONS[action]
    nx, ny = x + dx, y + dy

    if not (0 <= nx < N and 0 <= ny < N):
        return False

    return x >= locked_rows and nx >= locked_rows


def step(board, action):
    x, y = find_blank(board)
    dx, dy = ACTIONS[action]
    nx, ny = x + dx, y + dy
    new_board = board.copy()
    new_board[x, y], new_board[nx, ny] = new_board[nx, ny], new_board[x, y]
    return new_board


def print_board(board):
    for row in board:
        print(row)
    print()

def solve_row(initial_board, row_idx):
    """
    Learns a policy to solve row `row_idx` using value iteration
    while keeping rows [0 .. row_idx-1] fixed.
    """

    V = {}
    policy = {}
    def state_key(board):
        return tuple(board[row_idx:].flatten())

    frontier = [initial_board]
    visited = set()

    while frontier:
        board = frontier.pop()
        key = state_key(board)
        if key in visited:
            continue

        visited.add(key)
        V[key] = 0.0

        for action in ACTIONS:
            if valid_move(board, action, row_idx):
                frontier.append(step(board, action))

    while True:
        delta = 0.0

        for key in visited:
            sub = np.array(key).reshape(N - row_idx, N)
            board = np.vstack([GOAL[:row_idx], sub])

            if row_solved(board, row_idx):
                continue

            old_v = V[key]
            values = []

            for action in ACTIONS:
                if valid_move(board, action, row_idx):
                    next_board = step(board, action)
                    next_key = state_key(next_board)

                    reward = 50 if row_solved(next_board, row_idx) else -1
                    values.append(reward + GAMMA * V[next_key])

            V[key] = max(values)
            delta = max(delta, abs(old_v - V[key]))

        if delta < THETA:
            break

    for key in visited:
        sub = np.array(key).reshape(N - row_idx, N)
        board = np.vstack([GOAL[:row_idx], sub])

        if row_solved(board, row_idx):
            policy[key] = None
            continue

        best_action = None
        best_value = -1e9

        for action in ACTIONS:
            if valid_move(board, action, row_idx):
                next_board = step(board, action)
                next_key = state_key(next_board)
                val = -1 + GAMMA * V[next_key]

                if val > best_value:
                    best_value = val
                    best_action = action

        policy[key] = best_action

    return policy

def hierarchical_rl_solver(start_board):
    board = start_board.copy()
    step_count = 0

    print("Initial State:")
    print_board(board)

    for row in range(N):
        policy = solve_row(board, row)

        while not row_solved(board, row):
            key = tuple(board[row:].flatten())
            action = policy[key]

            board = step(board, action)
            step_count += 1

            print(f"Step {step_count}: Move {action}")
            print_board(board)

    print(step_count)

if __name__ == "__main__":

    start = np.array([
        [1,  2,  3,  4],
        [5,  6,  0,  8],
        [9, 10,  7, 12],
        [13, 14, 11, 15]
    ])

    hierarchical_rl_solver(start)


