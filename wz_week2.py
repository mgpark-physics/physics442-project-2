import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device="cpu"
curr = torch.accelerator.current_accelerator(check_available=True)
match curr:
    case torch.device():
        device = curr.type 
    case None:
        device = "cpu"
print(f"Using {device} device")

wz_classes = ['z','w']

class WZDataset(Dataset):
    def __init__(self,set_number:int):
        xsim = np.load('wz_data_'+str(set_number)+'.npy')
        xlab = np.load('wz_labels_'+str(set_number)+'.npy')
        self.points = torch.from_numpy(xsim).float()
        self.labels = torch.from_numpy(xlab)
    def __len__(self):
        return self.labels.size()[0]
    def __getitem__(self, index):
        return self.points[index],self.labels[index]
        


class WZClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
                nn.Linear(2,100),
                nn.ReLU(),
                nn.Linear(100,100),
                nn.ReLU(),
                nn.Linear(100,100),
                nn.ReLU(),
                nn.Linear(100,2),
                )

    def forward(self,x):
        return self.stack(x)


class WZBinaryClassifier(nn.Module):
    """
    Shallow binary logistic regression classifier.

    The model returns a logit
        z = w1*x1 + w2*x2 + b

    The corresponding probability for class 1 is
        p = sigmoid(z).
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


def train_loop(the_model,train_data_loader,print_every=100):
    crit = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(the_model.parameters(), lr=0.01, momentum=0.9)
    sched = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)
    the_model.train()
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (data,labels) in enumerate(train_data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data.to(device)
            labels = labels.to(device)


            # forward + backward + optimize
            outputs = the_model(inputs)
            loss = crit(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every-1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/print_every:.3f}')
                running_loss = 0.0

        sched.step()
    print('Finished Training')


def train_binary_loop(the_model, train_data_loader, print_every=100):
    crit = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(the_model.parameters(), lr=0.01, momentum=0.9)
    sched = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    the_model.train()

    for epoch in range(10):

        running_loss = 0.0

        for i, (data, labels) in enumerate(train_data_loader):

            inputs = data.to(device)

            # BCEWithLogitsLoss requires floating-point labels
            # with the same shape as the model output.
            labels = labels.float().unsqueeze(1).to(device)

            outputs = the_model(inputs)
            loss = crit(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % print_every == print_every - 1:
                print(
                    f'[{epoch + 1}, {i + 1:5d}] '
                    f'loss: {running_loss / print_every:.3f}'
                )
                running_loss = 0.0

        sched.step()

    print('Finished Training')

def test_loop(the_model,test_data_loader):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in wz_classes}
    total_pred = {classname: 0 for classname in wz_classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            inputs = inputs.to(device)

            outputs = the_model(inputs)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[wz_classes[label]] += 1
                total_pred[wz_classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def test_binary_loop(the_model, test_data_loader, threshold=0.5):

    correct_pred = {classname: 0 for classname in wz_classes}
    total_pred = {classname: 0 for classname in wz_classes}

    the_model.eval()

    with torch.no_grad():

        for data, labels in test_data_loader:

            inputs = data.to(device)
            labels = labels.to(device)

            logits = the_model(inputs).squeeze(1)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= threshold
            ).long()

            for label, prediction in zip(labels, predictions):

                label_int = label.item()
                prediction_int = prediction.item()

                if label_int == prediction_int:
                    correct_pred[wz_classes[label_int]] += 1

                total_pred[wz_classes[label_int]] += 1

    for classname, correct_count in correct_pred.items():

        accuracy = (
            100
            * float(correct_count)
            / total_pred[classname]
        )

        print(
            f'Accuracy for class: '
            f'{classname:5s} is {accuracy:.1f} %'
        )




if __name__ == "__main__":

    # ---------------------------------------------------------
    # Problem 1(b)
    # Train both classifiers on the clean data set
    # and test them on the corrupted data set.
    # ---------------------------------------------------------

    clean_set_number = 2
    corrupted_set_number = 1

    clean_data = WZDataset(clean_set_number)
    corrupted_data = WZDataset(corrupted_set_number)

    clean_loader = DataLoader(
        clean_data,
        batch_size=400,
        shuffle=True
    )

    corrupted_loader = DataLoader(
        corrupted_data,
        batch_size=400,
        shuffle=False
    )

    # ----- Original deep classifier -----

    print("\nTraining original WZClassifier on clean data\n")

    deep_model = WZClassifier().to(device)

    train_loop(
        deep_model,
        clean_loader
    )

    print("\nDeep model: clean-trained, tested on corrupted data\n")

    test_loop(
        deep_model,
        corrupted_loader
    )

    # ----- New shallow binary logistic classifier -----

    print("\nTraining shallow binary classifier on clean data\n")

    shallow_clean_model = WZBinaryClassifier().to(device)

    train_binary_loop(
        shallow_clean_model,
        clean_loader
    )

    print("\nShallow model: clean-trained, tested on corrupted data\n")

    test_binary_loop(
        shallow_clean_model,
        corrupted_loader
    )


    print("\nDeep model: clean-trained, tested on clean data\n")
    test_loop(
        deep_model,
        clean_loader
    )

    print("\nShallow model: clean-trained, tested on clean data\n")
    test_binary_loop(
        shallow_clean_model,
        clean_loader
    )


    w = shallow_clean_model.linear.weight.detach().cpu().numpy()[0]
    b = shallow_clean_model.linear.bias.detach().cpu().item()

    print("\nClean-trained shallow model weights:")
    print(f"w1 = {w[0]:.6f}")
    print(f"w2 = {w[1]:.6f}")
    print(f"b  = {b:.6f}")