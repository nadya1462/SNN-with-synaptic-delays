import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd
import seaborn as sn
import math
from Model import SNN
from definitions import *
from DataImport import *
from Classes import test, CreateClasses


def show_confusion_matrix(pred, labels):
    data = {'y_Actual': labels, 'y_Predicted': pred}
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()

def analyse(pred, labels):
    precision = accuracy_score(labels, pred)
    recall = recall_score(labels, pred, average='micro')
    F1_score = f1_score(labels, pred, average='micro')
    return precision, recall, F1_score 

def intensity(time):
    intensity_norm = (max_spike_time - time) / (max_spike_time - min_spike_time)
    intensity_scaled = (max_intensity - min_intensity) * intensity_norm + min_intensity
    return intensity_scaled

def draw_intensity(intensity_samples, minI, maxI):
    x = np.arange(-0.5, ni, 1)
    y = np.arange(-0.5, 3, 1)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_xticks(range(ni))
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Left", "Right", "Central"])
    ax.pcolormesh(x, y, intensity_samples, cmap='gray', vmin=minI, vmax=maxI)
    #for k in x:
      #plt.axvline(k, ls='-', c='w', lw=4) # white line between bars
    for k in y:
      plt.axhline(k, ls='-', c='black', lw=20)
    plt.show()

def predict(snn, test_times, classes):
    pred = []
    for times_sample in test_times:
        spike_num = np.asarray(test(snn, times_sample))
        pred += [classes[spike_num.argmax()]]
    return pred

def increase_noise(current_test_times, coef):
    for k in range(n_test):
        noise = np.random.randn(ni) * coef
        times_sample = current_test_times[k]
        new_times_sample = np.clip(times_sample + noise, min_spike_time, max_spike_time)
        current_test_times[k] = new_times_sample

def decrease_contrast(current_test_times, coef):
    for k in range(n_test):
        times_sample = current_test_times[k]
        new_times_sample = coef*times_sample + (1-coef)*sum(times_sample)/ni
        current_test_times[k] = new_times_sample

def plot_F1_score_change(f1):
    axes_x = np.array([20*math.log10(256*(0.7**i)/10) for i in range(len(f1))])
    #axes_x = np.linspace(1, len(f1), len(f1))
    plt.plot(axes_x, f1, marker = '*')
    plt.xlabel("signal / noise, Db")
    #plt.xlabel("The iteration of experiment")
    plt.ylabel("F1-score value")
    #plt.title("F1-score curve")
    plt.grid()
    plt.show()


(test_left_spike_times, test_right_spike_times, test_central_spike_times) = DataImport(False)

with open(file_path_shifts, 'rb') as f:
    shifts = np.load(f) * b2.second
snn = SNN(nn, ni, 0, shifts)

classes = CreateClasses(snn)

test_times = np.hstack([test_left_spike_times, test_right_spike_times, test_central_spike_times]).reshape(n_test, ni)
labels = ["left", "right", "central"]*(n_test//3)
acc_f1 = []

for i in range(8):
    mean_test_time_left = np.mean(test_times[0:n_test-2:3], axis = 0)
    mean_test_time_right = np.mean(test_times[1:n_test-1:3], axis = 0)
    mean_test_time_central = np.mean(test_times[2:n_test:3], axis = 0)
    intensities = np.vstack((intensity(mean_test_time_left), intensity(mean_test_time_right), intensity(mean_test_time_central)))
    #draw_intensity(intensities, min_intensity, max_intensity)

    pred = predict(snn, test_times, classes)
    #show_confusion_matrix(pred, labels)
    precision, recall, F1_score = analyse(pred, labels)
    acc_f1.append(F1_score)

    print(f"----------------------EXPERIMENT {i+1}---------------------------")
    print("Mean test time:")
    print(mean_test_time_left//1)
    print(mean_test_time_right//1)
    print(mean_test_time_central//1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1_score: ", F1_score) 
    increase_noise(test_times, 1.5)
    #decrease_contrast(test_times, 0.8)

plot_F1_score_change(np.array(acc_f1))
