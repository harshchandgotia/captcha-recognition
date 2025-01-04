import torch
from torch import nn
from torch.nn import functional as f

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3,3), padding=(1,1))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3,3))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=(1,1))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.linear = nn.Linear(768, 64)
        self.drop1 = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
        self.output = nn.Linear(64, num_chars+1)

    def forward(self, images, targets=None):
        bs, c, h, w = images.size()
        # print(bs, c, h, w)
        x = f.relu(self.conv1(images))
        # print(x.size())
        x = self.max_pool1(x)
        # print(x.size())
        x = f.relu(self.conv2(x))
        # print(x.size())
        x = self.max_pool2(x) #1,64,12,50
        x = x.permute(0, 3, 1, 2) #1, 50, 64, 12
        # print(x.size())
        x = x.view(bs, x.size(1), -1)
        # print(x.size()) 
        x = self.linear(x)
        x = self.drop1(x)
        # print(x.size())
        x,_ = self.gru(x)
        # print(x.size())
        x = self.output(x)
        # print(x.size())
        x = x.permute(1,0,2)
        if targets is not None:
            log_softmax_values = f.log_softmax(x, 2)
            input_length = torch.full(
                size=(bs,),
                fill_value=log_softmax_values.size(0),
                dtype=torch.int32
            )
            # print(input_length)
            output_length = torch.full(
                size=(bs,),
                fill_value=targets.size(1),
                dtype=torch.int32
            )
            # print(output_length)
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, targets, input_length, output_length
            )
        return x, loss

if __name__ == "__main__":
    cm = CaptchaModel(19)
    img = torch.rand(5,3,75,300)
    target = torch.randint(1,20, (5,5))
    x, loss = cm(img, target)
