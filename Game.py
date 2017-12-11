# coding=utf-8
from random import random, randint
import numpy as np
import copy

class Game(object):

    left = (-1, 0)
    right = (1, 0)
    up = (0, 1)
    down = (0, -1)

    # size 为盘面大小，alpha 为每次出现 4 作为块的概率
    def __init__(self, size=4, alpha=0.1):
        self.size = size
        self.alpha = alpha
        self.game_board = None
        self.max = 2


    # 初始化一个全部为0的board
    def _new_board(self):
        self.game_board = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                row.append(0)
            self.game_board.append(row)


    # 向board里面加入两块
    def _init_board(self):
        a = randint(0, self.size - 1)
        b = randint(0, self.size - 1)
        self.game_board[a][b] = Game.sample_block(self.alpha)  # 因为是新的版面，一定能直接加进去
        while self.game_board[a][b] != 0:
            a = randint(0, self.size - 1)
            b = randint(0, self.size - 1)
        self.game_board[a][b] = Game.sample_block(self.alpha)


    # 加入新的一块
    def _add_new_block(self):
        a = randint(0, self.size - 1)
        b = randint(0, self.size - 1)
        while self.game_board[a][b] != 0:
            a = randint(0, self.size - 1)
            b = randint(0, self.size - 1)
        self.game_board[a][b] = Game.sample_block(self.alpha)


    # 向左移动
    def _left(self):
        self._reverse()
        result = self._right()
        self._reverse()
        return result


    # 向右移动
    def _right(self):
        pay = -2
        temp_board = copy.deepcopy(self.game_board)
        for row in self.game_board:
            curr, changed = self.size - 1, True
            while True:
                if changed:
                    Game._move_helper(row, curr)
                if curr == 0 or row[curr - 1] == 0:
                    break
                if row[curr] == row[curr - 1]:
                    row[curr] *= 2
                    if row[curr] > self.max:
                        max = row[curr]
                        pay = max
                    row[curr - 1] = 0
                    changed = True
                else:
                    changed = False
                curr -= 1
        if temp_board != self.game_board:
            self._add_new_block()
        else:
            pay = -1
        return (pay, self._end())


    # 把所有的非零元素移动到end_index为结束的坑中
    @staticmethod
    def _move_helper(row, end_index):
        count = 0
        for i in range(end_index, -1, -1):
            if row[i] != 0:
                row[end_index - count], row[i] = row[i], row[end_index - count]
                count += 1


    # 向上移动
    def _up(self):
        self._transpose()
        self._reverse()
        result = self._right()
        self._reverse()
        self._transpose()
        return result


    # 向下移动
    def _down(self):
        self._transpose()
        result = self._right()
        self._transpose()
        return result


    # 将盘面左右颠倒
    def _reverse(self):
        for row in self.game_board:
            for i in range(self.size / 2):
                row[i], row[self.size - i - 1] = row[self.size - i - 1], row[i]


    # 将盘面转置
    def _transpose(self):
        for i in range(self.size):
            for j in range(i + 1, self.size):
                self.game_board[i][j], self.game_board[j][i] = self.game_board[j][i], self.game_board[i][j]


    # 判断游戏是否结束
    def _end(self):
        # 是否有空白
        for i in range(self.size):
            for j in range(self.size):
                if self.game_board[i][j] == 0:
                    return False
        #是否存在可消除的情况
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.game_board[i][j] == self.game_board[i + 1][j]:
                    return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.game_board[i][j] == self.game_board[i][j + 1]:
                    return False
        # 无法继续进行游戏
        return True


    # 新建一盘游戏
    def new_game(self):
        self._new_board()
        self._init_board()
        self.max = 2


    # 返回当前盘面的最大值
    def get_max(self):
        max_block = 0
        for row in self.game_board:
            for block in row:
                max_block = max(max_block, block)
        return max_block


    # 通过方向元组来控制移动，返回获得的奖励以及游戏是否结束
    def move(self, direction):
        if direction == (-1, 0):
            return self._left()
        elif direction == (1, 0):
            return self._right()
        elif direction == (0, 1):
            return self._up()
        elif direction == (0, -1):
            return self._down()


    # 按照概率生成 2 或者 4 的块
    @staticmethod
    def sample_block(alpha):
        assert alpha > 0 and alpha < 1
        return 4 if random() < alpha else 2


    def __str__(self):
        s = ""
        for row in self.game_board:
            s += str(row)
            s += "\n"
        return s


    def __call__(self, direction):
        return self.move(direction)


    # 获取所有可行的行动
    @staticmethod
    def get_actions():
        return (Game.left, Game.right, Game.up, Game.down)


    def get_state(self):
        return np.array(self.game_board).reshape((1, 4, 4))


    def get_preprocessed_state(self):
        raw =  np.array(self.game_board).reshape((1, 4, 4)).astype(float)
        raw[raw > 0] = np.log2(raw[raw > 0])
        return raw