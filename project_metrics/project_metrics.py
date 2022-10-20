from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.callbacks import History

import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CLASSES = ["0", "1"]

# Essas funções foram baseadas em: https://github.com/RomeroBarata/IF702-redes-neurais/blob/master/main.ipynb

def extract_final_losses(history:dict):
    """Função para extrair o melhor loss de treino e validação.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    Dicionário contendo o melhor loss de treino e de validação baseado 
    no menor loss de validação.
    """
    train_loss = history['loss']
    val_loss = history['val_loss']
    idx_min_val_loss = np.argmin(val_loss)
    return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}

def plot_training_error_curves(history:History):
    """Função para plotar as curvas de erro do treinamento da rede neural.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    A função gera o gráfico do treino da rede e retorna None.
    """
    train_loss = history['loss']
    val_loss = history['val_loss']
    
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Train')
    ax.plot(val_loss, label='Validation')
    ax.set(title='Training and Validation Error Curves', xlabel='Epochs', ylabel='Loss (MSE)')
    ax.legend()
    plt.show()

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.show()
        y_pred_scores = y_pred_scores[:, 1]
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        performance_metrics = performance_metrics + (auroc, aupr)
    return performance_metrics

def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):
    print()
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
    if auroc is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))


# def get_data(network, test_data):
#     results = [(np.argmax(network.feedforward(x)), y) for (x, y) in test_data]
#     y_true = [i for i,_ in results]
#     y_pred = [j for _,j in results]
#     return y_true,y_pred


def plot_confusion_matrix(y_true, y_pred, fontsize = 20):
    """
    This function was based on https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    """
    #y_true,y_pred = get_data(network, test_data)
    cf_matrix = confusion_matrix(y_true,y_pred)
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    s = sns.heatmap(cf_matrix, annot=labels, fmt="")
    # plt.figure(figsize = figsize)
    # plt.title(title,fontsize = fontsize + int(0.5*fontsize))
    # s = sns.heatmap(cm, annot = True, xticklabels = classes, yticklabels = classes,fmt = fmt)
    s.set_xlabel('Predicted Label',fontsize = fontsize)
    s.set_ylabel('True Label', fontsize = fontsize)
    #plt.show()

def print_classification_report(y_true,y_pred, classes=CLASSES):
    # y_true,y_pred = get_data(network, test_data)
    print(classification_report(y_true,y_pred,target_names=classes))
    