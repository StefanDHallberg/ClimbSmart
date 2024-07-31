import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions, input_width, input_height):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the correct input size for the first fully connected layer
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_width, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_height, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, 256)  # Added another fully connected layer
        self.out = nn.Linear(256, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print(f"Shape after conv1: {x.shape}")
        x = torch.relu(self.conv2(x))
        print(f"Shape after conv2: {x.shape}")
        x = torch.relu(self.conv3(x))
        print(f"Shape after conv3: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten the tensor
        print(f"Shape after flatten: {x.shape}")
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x
