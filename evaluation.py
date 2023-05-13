import torch
import numpy as np

class Tester:
    def __init__(self):
        return

    def get_mean_accuracy(self, test_dataloader, model, classes):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images).to(device)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1



        # print accuracy for each class
        accuracyClasses = []
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            accuracyClasses += [accuracy]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        return np.mean(accuracyClasses)