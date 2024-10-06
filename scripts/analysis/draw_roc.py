from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

fpr, tpr, thresholds = roc_curve(target, score, pos_label=1)
 
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
        lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("./roc.png")
plt.show()