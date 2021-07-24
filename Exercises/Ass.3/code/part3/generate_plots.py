import torch
import matplotlib.pyplot as plt


a_pos_acc, a_pos_loss = torch.load("a_model_pos")
b_pos_acc, b_pos_loss = torch.load("b_model_pos")
c_pos_acc, c_pos_loss = torch.load("c_model_pos")
d_pos_acc, d_pos_loss = torch.load("d_model_pos")

a_ner_acc, a_ner_loss = torch.load("a_model_ner")
b_ner_acc, b_ner_loss = torch.load("b_model_ner")
c_ner_acc, c_ner_loss = torch.load("c_model_ner")
d_ner_acc, d_ner_loss = torch.load("d_model_ner")

plt.figure(0)
plt.plot([5 * (i + 1) for i in range(len(a_pos_acc))], a_pos_acc, label="Model (a)")
plt.plot([5 * (i + 1) for i in range(len(b_pos_acc))], b_pos_acc, label="Model (b)")
plt.plot([5 * (i + 1) for i in range(len(c_pos_acc))], c_pos_acc, label="Model (c)")
plt.plot([5 * (i + 1) for i in range(len(d_pos_acc))], d_pos_acc, label="Model (d)")
plt.legend()
plt.title("POS dev Accuracy")
plt.ylabel("Accuracy (%)")
plt.xlabel("Num. examples / 100")
plt.show()
plt.savefig("POS dev Accuracy.jpg")

plt.figure(1)
plt.plot([5 * (i + 1) for i in range(len(a_ner_acc))], a_ner_acc, label="Model (a)")
plt.plot([5 * (i + 1) for i in range(len(b_ner_acc))], b_ner_acc, label="Model (b)")
plt.plot([5 * (i + 1) for i in range(len(c_ner_acc))], c_ner_acc, label="Model (c)")
plt.plot([5 * (i + 1) for i in range(len(d_ner_acc))], d_ner_acc, label="Model (d)")
plt.legend()
plt.title("NER dev Accuracy")
plt.ylabel("Accuracy (%)")
plt.xlabel("Num. examples / 100")
plt.show()
plt.savefig("NER dev Accuracy.jpg")

