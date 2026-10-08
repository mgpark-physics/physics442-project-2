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



def plot_decision_boundaries(
    clean_data,
    corrupted_data,
    clean_model,
    corrupted_model,
    filename="wz_decision_boundaries.pdf"
):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(9, 4),
        sharex=True,
        sharey=True
    )

    datasets = [
        (clean_data, "clean"),
        (corrupted_data, "corrupted")
    ]

    models = [
        (clean_model, "clean-trained"),
        (corrupted_model, "corrupted-trained")
    ]

    # Common plotting range
    all_points = torch.cat(
        [clean_data.points, corrupted_data.points],
        dim=0
    ).numpy()

    xmin = all_points[:, 0].min()
    xmax = all_points[:, 0].max()
    ymin = all_points[:, 1].min()
    ymax = all_points[:, 1].max()

    xx = np.linspace(xmin, xmax, 400)

    for ax, (dataset, dataset_name) in zip(axes, datasets):

        x = dataset.points.numpy()
        y = dataset.labels.numpy()

        ax.scatter(
            x[y == 0, 0],
            x[y == 0, 1],
            s=5,
            alpha=0.15,
            label="Z"
        )

        ax.scatter(
            x[y == 1, 0],
            x[y == 1, 1],
            s=5,
            alpha=0.15,
            label="W"
        )

        # Draw both learned boundaries on each data set
        for model, model_name in models:

            w = (
                model.linear.weight
                .detach()
                .cpu()
                .numpy()[0]
            )

            b = (
                model.linear.bias
                .detach()
                .cpu()
                .item()
            )

            yy = -(w[0] * xx + b) / w[1]

            ax.plot(
                xx,
                yy,
                linewidth=2,
                label=model_name
            )

            ax.plot(
                xx,
                xx,
                ":",
                linewidth=1,
                label=r"$x_1=x_2$"
            )              

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel(r"$x_1$")
        ax.set_title(dataset_name)

    axes[0].set_ylabel(r"$x_2$")

    axes[1].legend(
        frameon=False,
        loc="best"
    )

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)



def get_binary_scores(the_model, data_loader):
    """
    Return true labels and P(class=1) for a binary classifier.
    """
    the_model.eval()

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for data, labels in data_loader:
            inputs = data.to(device)

            logits = the_model(inputs).squeeze(1)
            scores = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

    return (
        np.concatenate(all_labels).astype(int),
        np.concatenate(all_scores)
    )


def confusion_counts(labels, scores, threshold):
    """
    Confusion-matrix entries for class 1 = signal,
    class 0 = background.
    """
    predictions = (scores >= threshold).astype(int)

    TP = np.sum((labels == 1) & (predictions == 1))
    TN = np.sum((labels == 0) & (predictions == 0))
    FP = np.sum((labels == 0) & (predictions == 1))
    FN = np.sum((labels == 1) & (predictions == 0))

    return TN, FP, FN, TP


def mcc_from_counts(TN, FP, FN, TP):
    """
    Matthews correlation coefficient.
    """
    numerator = TP * TN - FP * FN

    denominator = np.sqrt(
        (TP + FP)
        * (TP + FN)
        * (TN + FP)
        * (TN + FN)
    )

    if denominator == 0:
        return np.nan

    return numerator / denominator


def binary_roc_curve(labels, scores):
    """
    Construct ROC curve without scikit-learn.
    """
    order = np.argsort(scores)[::-1]

    scores_sorted = scores[order]
    labels_sorted = labels[order]

    signal = (labels_sorted == 1)
    background = (labels_sorted == 0)

    cumulative_TP = np.cumsum(signal)
    cumulative_FP = np.cumsum(background)

    # Keep only locations where the score changes.
    distinct = np.where(np.diff(scores_sorted))[0]
    indices = np.r_[distinct, len(scores_sorted) - 1]

    TP = cumulative_TP[indices]
    FP = cumulative_FP[indices]

    n_signal = np.sum(signal)
    n_background = np.sum(background)

    TPR = TP / n_signal
    FPR = FP / n_background

    thresholds = scores_sorted[indices]

    # Explicit ROC origin
    TPR = np.r_[0.0, TPR]
    FPR = np.r_[0.0, FPR]
    thresholds = np.r_[np.inf, thresholds]

    return FPR, TPR, thresholds


def report_threshold(labels, scores, threshold, name=""):
    TN, FP, FN, TP = confusion_counts(
        labels,
        scores,
        threshold
    )

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    MCC = mcc_from_counts(
        TN,
        FP,
        FN,
        TP
    )

    print(f"\n{name}")
    print(f"threshold = {threshold:.6f}")

    print("confusion matrix:")
    print(f"[[TN={TN:5d}, FP={FP:5d}],")
    print(f" [FN={FN:5d}, TP={TP:5d}]]")

    print(f"TPR = {TPR:.6f}")
    print(f"FPR = {FPR:.6f}")
    print(f"MCC = {MCC:.6f}")





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




    print("\nTraining shallow binary classifier on corrupted data\n")

    shallow_corrupted_model = WZBinaryClassifier().to(device)

    train_binary_loop(
        shallow_corrupted_model,
        corrupted_loader
    )

    print("\nShallow model: corrupted-trained, tested on corrupted data\n")

    test_binary_loop(
        shallow_corrupted_model,
        corrupted_loader
    )

    print("\nShallow model: corrupted-trained, tested on clean data\n")

    test_binary_loop(
        shallow_corrupted_model,
        clean_loader
    )


    # Print weights

    w = (
        shallow_corrupted_model
        .linear.weight
        .detach()
        .cpu()
        .numpy()[0]
    )

    b = (
        shallow_corrupted_model
        .linear.bias
        .detach()
        .cpu()
        .item()
    )

    print("\nCorrupted-trained shallow model weights:")
    print(f"w1 = {w[0]:.6f}")
    print(f"w2 = {w[1]:.6f}")
    print(f"b  = {b:.6f}")
    print(f"w1/w2 = {w[0]/w[1]:.6f}")


    plot_decision_boundaries(
    clean_data,
    corrupted_data,
    shallow_clean_model,
    shallow_corrupted_model
    )





    # ============================================================
    # Problem 2(a)
    # ROC, AUC, confusion matrices, MCC
    # ============================================================

    print("\n\n========== Problem 2(a) ==========\n")

    # Make a reproducible 80/20 split of the clean data.
    generator = torch.Generator().manual_seed(442)

    n_train = int(0.8 * len(clean_data))
    n_test = len(clean_data) - n_train

    clean_train_data, clean_test_data = torch.utils.data.random_split(
        clean_data,
        [n_train, n_test],
        generator=generator
    )

    clean_train_loader_2 = DataLoader(
        clean_train_data,
        batch_size=400,
        shuffle=True
    )

    clean_test_loader_2 = DataLoader(
        clean_test_data,
        batch_size=1000,
        shuffle=False
    )


    # Train the shallow classifier only on the training part.
    shallow_roc_model = WZBinaryClassifier().to(device)

    print("Training shallow classifier for ROC study\n")

    train_binary_loop(
        shallow_roc_model,
        clean_train_loader_2
    )


    labels_roc, scores_roc = get_binary_scores(
    shallow_roc_model,
    clean_test_loader_2
    )

    FPR, TPR, roc_thresholds = binary_roc_curve(
        labels_roc,
        scores_roc
    )

    AUC = np.trapezoid(TPR, FPR)

    print(f"\nROC AUC = {AUC:.6f}")


    youden = TPR - FPR

    i_youden = np.argmax(youden)

    threshold_youden = roc_thresholds[i_youden]



    mcc_values = []

    for threshold in roc_thresholds[1:]:
        TN, FP, FN, TP = confusion_counts(
            labels_roc,
            scores_roc,
            threshold
        )

        mcc_values.append(
            mcc_from_counts(
                TN,
                FP,
                FN,
                TP
            )
        )

    mcc_values = np.asarray(mcc_values)

    i_mcc = np.nanargmax(mcc_values)

    threshold_mcc = roc_thresholds[1:][i_mcc]


    print(f"0.5 threshold       = {0.5:.6f}")
    print(f"Youden threshold    = {threshold_youden:.6f}")
    print(f"max-MCC threshold   = {threshold_mcc:.6f}")


    report_threshold(
    labels_roc,
    scores_roc,
    0.5,
    name="Threshold 0.5"
    )

    report_threshold(
        labels_roc,
        scores_roc,
        threshold_youden,
        name="Maximum Youden J"
    )

    report_threshold(
        labels_roc,
        scores_roc,
        threshold_mcc,
        name="Maximum MCC"
    )



    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(
        FPR,
        TPR,
        linewidth=2,
        label=fr"AUC = {AUC:.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=1
    )

    # Mark the conventional threshold 0.5
    TN, FP, FN, TP = confusion_counts(
        labels_roc,
        scores_roc,
        0.5
    )

    tpr_05 = TP / (TP + FN)
    fpr_05 = FP / (FP + TN)

    ax.scatter(
        fpr_05,
        tpr_05,
        s=40,
        label=r"$t=0.5$"
    )

    ax.scatter(
        FPR[i_youden],
        TPR[i_youden],
        s=40,
        label="max Youden"
    )

    # Location corresponding to maximum MCC
    threshold_mcc_index = np.where(
        roc_thresholds == threshold_mcc
    )[0][0]

    ax.scatter(
        FPR[threshold_mcc_index],
        TPR[threshold_mcc_index],
        s=40,
        label="max MCC"
    )

    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        "wz_roc_clean_shallow.pdf"
    )

    plt.close(fig)



    # ============================================================
    # Problem 2(b)
    # Rare-signal operating point
    # ============================================================

    print("\n\n========== Problem 2(b) ==========\n")

    n_background = np.sum(labels_roc == 0)
    n_signal = np.sum(labels_roc == 1)

    print(f"Validation background events = {n_background}")
    print(f"Validation signal events     = {n_signal}")
    print(
        "One false positive corresponds to "
        f"FPR = {1 / n_background:.6e}"
    )

    signal_to_background_before = 1.0e-4

    for max_fp in [0, 1, 2, 5, 10]:

        best = None

        for threshold in roc_thresholds[1:]:

            TN, FP, FN, TP = confusion_counts(
                labels_roc,
                scores_roc,
                threshold
            )

            if FP <= max_fp:

                tpr = TP / (TP + FN)
                fpr = FP / (FP + TN)

                if best is None or tpr > best["tpr"]:
                    best = {
                        "threshold": threshold,
                        "TN": TN,
                        "FP": FP,
                        "FN": FN,
                        "TP": TP,
                        "tpr": tpr,
                        "fpr": fpr
                    }

        print(f"\nMaximum FP allowed = {max_fp}")

        if best is not None:

            print(f"threshold = {best['threshold']:.6f}")
            print(f"TPR       = {best['tpr']:.6f}")
            print(f"FPR       = {best['fpr']:.6e}")
            print(
                f"matrix    = [[{best['TN']}, {best['FP']}], "
                f"[{best['FN']}, {best['TP']}]]"
            )

            if best["fpr"] > 0:

                selected_s_over_b = (
                    signal_to_background_before
                    * best["tpr"]
                    / best["fpr"]
                )

                print(
                    "expected selected S/B = "
                    f"{selected_s_over_b:.6e}"
                )

            else:

                print(
                    "expected selected S/B cannot be estimated "
                    "directly from zero observed false positives"
                )

            # Normalize to 10^6 pre-selection background events.
            n_background_example = 1_000_000

            n_signal_before = (
                signal_to_background_before
                * n_background_example
            )

            n_signal_flagged = (
                n_signal_before
                * best["tpr"]
            )

            print(
                "flagged signal per 10^6 background events = "
                f"{n_signal_flagged:.2f}"
            )

    print(
        "\nApproximate 95% upper limit on FPR "
        "for zero observed false positives:"
    )
    print(f"3/N_background = {3 / n_background:.6e}")


    # ------------------------------------------------------------
    # Explore the threshold region between the 0-FP and 1-FP points
    # ------------------------------------------------------------

    print("\n\n--- Fine scan between the 0-FP and 1-FP region ---\n")

    background_scores = scores_roc[labels_roc == 0]
    signal_scores = scores_roc[labels_roc == 1]

    # Sort the background scores from largest to smallest.
    background_scores_sorted = np.sort(background_scores)[::-1]

    highest_background_score = background_scores_sorted[0]
    second_highest_background_score = background_scores_sorted[1]

    print(f"highest background score        = {highest_background_score:.9f}")
    print(f"second-highest background score = {second_highest_background_score:.9f}")

    # For zero false positives, the threshold must be just above
    # the highest observed background score.
    threshold_zero_fp = np.nextafter(
        highest_background_score,
        np.inf
    )

    # For one false positive, the threshold can be lowered to just
    # above the second-highest observed background score.
    threshold_one_fp = np.nextafter(
        second_highest_background_score,
        np.inf
    )

    print(f"\noptimal threshold for FP = 0 : {threshold_zero_fp:.9f}")
    print(f"optimal threshold for FP <= 1: {threshold_one_fp:.9f}")

    # Examine every signal score lying between these two thresholds.
    # These are the actual locations where the signal efficiency changes.
    interesting_signal_scores = signal_scores[
        (signal_scores >= threshold_one_fp)
        & (signal_scores <= threshold_zero_fp)
    ]

    candidate_thresholds = np.unique(
        np.concatenate(
            (
                [threshold_zero_fp],
                interesting_signal_scores,
                [threshold_one_fp]
            )
        )
    )[::-1]

    n_background_example = 1_000_000
    n_signal_example = 1.0e-4 * n_background_example

    print(
        "\nthreshold        FP     TP      TPR          "
        "S_sel      B_sel"
    )
    print(
        "---------------------------------------------------------------"
    )

    for threshold in candidate_thresholds:

        TN, FP, FN, TP = confusion_counts(
            labels_roc,
            scores_roc,
            threshold
        )

        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)

        S_selected = n_signal_example * tpr
        B_selected = n_background_example * fpr

        print(
            f"{threshold:0.9f}   "
            f"{FP:2d}   "
            f"{TP:4d}   "
            f"{tpr:0.6f}   "
            f"{S_selected:8.3f}   "
            f"{B_selected:8.3f}"
        )