import numpy as np
import random
import matplotlib.pyplot as plt

# Размер игрового поля
BOARD_ROWS = 3
BOARD_COLS = 3

# Константы для определения награды
WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSE_REWARD = 0.0

# Частота разведочных (случайных) ходов агента
EPSILON = 0.05

# Количество эпизодов обучения
EPISODES = 10000

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def available_moves(self):
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, row, col, player):
        self.board[row][col] = player
        if self.check_winner(player):
            self.game_over = True
            self.winner = player
        elif len(self.available_moves()) == 0:
            self.game_over = True
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

    def check_winner(self, player):
        for i in range(BOARD_ROWS):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.current_player = 1
        self.game_over = False
        self.winner = None

class QLearningAgent:
    def __init__(self, alpha=0.9, gamma=0.1):
        self.Q_values = {}  # Словарь для хранения значений Q-функции
        self.alpha = alpha  # Скорость обучения
        self.gamma = gamma  # Коэффициент дисконтирования

    def get_Q_value(self, state, action):
        if (state, action) not in self.Q_values:
            self.Q_values[(state, action)] = 0.5  # Инициализация Q-значений в середине
        else:
             a = self.Q_values[(state, action)]
        return self.Q_values[(state, action)]

    def update_Q_value(self, state, action, reward, next_state, available_actions):
        max_next_Q = max([self.get_Q_value(next_state, a) for a in available_actions])
        current_Q = self.get_Q_value(state, action)
        # new_Q = current_Q + self.alpha * (reward + self.gamma * max_next_Q - current_Q)
        new_Q = current_Q + self.alpha * (reward + max_next_Q - current_Q)
        self.Q_values[(state, action)] = new_Q

    def choose_action(self, state, available_actions):
        if np.random.uniform(0, 1) < EPSILON:
            return random.choice(available_actions)
        else:
            Q_values = [self.get_Q_value(state, action) for action in available_actions]
            max_Q = max(Q_values)
            if Q_values.count(max_Q) > 1:
                best_moves = [i for i in range(len(available_actions)) if Q_values[i] == max_Q]
                i = random.choice(best_moves)
            else:
                i = Q_values.index(max_Q)
            return available_actions[i]

def play_game(agent, games_won):
    env = TicTacToe()
    state_values = {}  # Таблица для хранения значений ценности состояний игры
    total_reward = 0

    while not env.game_over:
        if env.current_player == 1:
            state = env.board.tobytes()
            available_moves = env.available_moves()
            action = agent.choose_action(state, available_moves)
            row, col = action
            env.make_move(row, col, 1)

            if (env.winner == 1):
                games_won += 1

            if (env.game_over):
                break

            next_state = env.board.tobytes()
            reward = WIN_REWARD if env.winner == 1 else DRAW_REWARD if env.winner is None else LOSE_REWARD
            total_reward += reward

            # Обновляем Q-значения по Уравнению Беллмана
            agent.update_Q_value(state, action, reward, next_state, env.available_moves())

            state_values[state] = max([agent.get_Q_value(next_state, a) for a in env.available_moves()])
        else:
            random_action = random.choice(env.available_moves())
            row, col = random_action
            env.make_move(row, col, 2)

    return state_values, total_reward, games_won

# Обучение агента
agent = QLearningAgent()
games_won = 0
state_values = []

total_reward = 0
episode_rewards = {}

for episode in range(EPISODES):
    local_state_values, local_reword, games_won = play_game(agent, games_won)
    total_reward += local_reword
    state_values.append(local_state_values)
    if episode % 50 == 0:
        episode_rewards[episode] = total_reward
        total_reward = 0
    if episode != 0:
        print(games_won / episode * 100)
    # if episode % 1000 == 0:
    #     print(agent.Q_values.values())
    #     print('------------------------------------------------')

# Построение кривой зависимости награды от количества шагов обучения
plt.plot(episode_rewards.keys(), episode_rewards.values())
plt.xlabel('Эпизоды')
plt.ylabel('Награда')
plt.title('Зависимость награды от количества шагов обучения')
plt.show()