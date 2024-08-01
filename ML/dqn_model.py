import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions, input_width, input_height):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_width, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_height, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, 256) 
        self.out = nn.Linear(256, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # print(f"After conv1: {x.shape}")
        x = torch.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")
        x = torch.relu(self.conv3(x))
        # print(f"After conv3: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(f"After flatten: {x.shape}")
        x = torch.relu(self.fc1(x))
        # print(f"After fc1: {x.shape}")
        x = torch.relu(self.fc2(x))
        # print(f"After fc2: {x.shape}")
        x = self.out(x)
        # print(f"Output shape: {x.shape}")
        return x

