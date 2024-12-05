import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models


class Interpreter(nn.Module):
    def __init__(self, num_classes=2000):
        super().__init__()

        # Use GPU
        self.device = torch.device("mps")

        # --------- Build CNN layer to extract features ---------
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Use pretrained ResNet34 weights
        resnet34 = resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            *list(resnet34.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Freeze all params except last layer to learn sign language specific features
        for name, param in self.feature_extractor.named_parameters():
            if name[0] != '7':
                param.requires_grad = False

        # --------- RNN layer to process sequence of feature vectors ---------
        self.gru = nn.GRU(
            input_size=512, # feature vector size from ResNet34
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # --------- Fully connected layer for classification ---------
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Softmax(dim=1)
        )
        

        # Setup loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())


    def forward(self, x):
        normalized_tensor = self.preprocess(x)
        features = self.feature_extractor(normalized_tensor)
        gru_output, hidden_state = self.gru(features)
        predictions = self.classifier(hidden_state[-1])
        return predictions

    def fit(self, data, epochs):
        self.train()

        for e in range(epochs):
            print(f"Epoch #{e} -------")
            for x, y in data.load():
                x = torch.from_numpy(x).to(self.device)
                y = torch.from_numpy(y).to(self.device)

                pred = self(x)
                loss = self.loss_fn(pred, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def test(self, data):
        size = data.total_samples
        self.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in data.load():
                x = torch.from_numpy(x).to(self.device)
                y = torch.from_numpy(y).to(self.device)
                pred = self(x)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= data.num_batches
            correct /= data.total_samples
            print(f"Test Error:\nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

    def predict(self, x):
        self.eval()
        x = torch.from_numpy(x).to(self.device)
        pred = self(x)
        return pred.argmax(1)
