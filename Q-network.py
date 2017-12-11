import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from Game import Game
import matplotlib.pyplot as plt

batch_size = 32
lr = 1e-2
epsilon = 0.9
gamma = 0.99
target_replace_iter = 100
memory_capacity = 10000
state_size = 16
shape = (1, 4, 4)
epoch = 300
batch_shape = (batch_size, 1, 4, 4)

class Net(nn.Module):

    def __init__(self, input_size=(1, 4, 4), output_size=4, alpha=0.1):
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        H_out = (input_size[1] - 1) * 2 + 2
        W_out = (input_size[2] - 1) * 2 + 2
        self.fc1 = nn.Linear(3 * H_out * W_out, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, output_size)
        self.fc0 = nn.ConvTranspose2d(self.input_size[0], 3, kernel_size=2, stride=2)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal(module.weight.data, 0, 0.1)
                nn.init.constant(module.bias.data, 0)


    def forward(self, input_vector):
        out0 = self.fc0(input_vector).view(-1, 24 * 8)
        out1 = F.leaky_relu(self.fc1(out0), negative_slope=self.alpha)
        out2 = F.leaky_relu(self.fc2(out1), negative_slope=self.alpha)
        out3 = F.leaky_relu(self.fc3(out2), negative_slope=self.alpha)
        actions_value = out3
        return actions_value


class QN_2048(object):

    def __init__(self, action_set):
        self.eval_net, self.predict_net = Net(), Net()
        self.memory = np.zeros((memory_capacity, state_size * 2 + 2))
        self.memory_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self._loss = nn.MSELoss()
        self.action_size = len(action_set)
        self.actions = action_set
        self.learn_step_counter = 0


    def choose_action(self, x):
        x = Variable(torch.unsqueeze(x, 0))
        if np.random.uniform() < epsilon:
            actions_values = self.eval_net.forward(x)
            action_index = actions_values.data.numpy().argmax()
        else:
            action_index = np.random.randint(0, self.action_size)
        return action_index, self.actions[action_index]


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s.reshape(state_size), [a, r], s_.reshape(state_size)))
        index = self.memory_counter % memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):
        if self.learn_step_counter & target_replace_iter == 0:
            self.predict_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(memory_capacity, batch_size)
        batch_memory = self.memory[sample_index, :]
        b_s = batch_memory[:, :state_size].reshape(batch_shape)
        b_s = Variable(torch.FloatTensor(b_s))
        b_a = batch_memory[:, state_size:state_size + 1].astype(int)
        b_a = Variable(torch.LongTensor(b_a))
        b_r = batch_memory[:, state_size+1:state_size+2]
        b_r = Variable(torch.FloatTensor(b_r))
        b_s_ = batch_memory[:, -state_size:].reshape(batch_shape)
        b_s_ = Variable(torch.FloatTensor(b_s_))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.predict_net(b_s_).detach()
        q_target = b_r + gamma * torch.unsqueeze(q_next.max(1)[0], 1)
        loss = self._loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train():
    qn = QN_2048(Game.get_actions())
    game = Game()
    upper_bound = []
    for i in range(epoch):
        game.new_game()
        while True:
            state = torch.FloatTensor(game.get_preprocessed_state())
            action_index, action = qn.choose_action(state)
            pay, result = game.move(action)
            qn.store_transition(state.numpy(), action_index, pay, game.get_preprocessed_state())
            if qn.memory_counter > memory_capacity:
                qn.learn()
            if result == True:
                upper_bound.append(game.get_max())
                break
        print "In epoch", i, ", max_block =", upper_bound[-1]
        print game
    plt.plot(upper_bound)
    plt.show()



if __name__ == "__main__":
    train()