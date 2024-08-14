import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions, input_width, input_height):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the correct output size after the convolution layers
        convw = 52  # Based on printed shapes
        convh = 46  # Based on printed shapes
        linear_input_size = convw * convh * 64  # 153088 is the product of these dimensions

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, 256) 
        self.out = nn.Linear(256, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print("Shape after conv1:", x.shape)  # Debugging statement
        x = torch.relu(self.conv2(x))
        print("Shape after conv2:", x.shape)  # Debugging statement
        x = torch.relu(self.conv3(x))
        print("Shape after conv3:", x.shape)  # Debugging statement

        x = x.view(x.size(0), -1)  # Flatten the tensor
        print("Shape after flattening:", x.shape)  # Debugging statement

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x
