import numpy as np
import pandas as pd

train_path = "../data/train.csv"
test_path = "../data/sample_submission.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

model_result_1="./audio_tag_start_kernel/freesound-prediction-data-2d-conv-reduced-lr_0.647"
model_result_2="./audio_tag_start_kernel/freesound-prediction-file"
model_result_3="./audio_crnn/crnn_result_0.61986"

LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])

pred_list = []
for i in range(10):
    pred_list.append(np.load(model_result_1+"/test_predictions_%d.npy"%i))
for i in range(10):
    pred_list.append(np.load(model_result_2+"/test_predictions_%d.npy"%i))
for i in range(10):
    pred_list.append(np.load(model_result_3+"/test_predictions_%d.npy"%i))

prediction = np.ones_like(pred_list[0])
for pred in pred_list:
    prediction = prediction*pred
prediction = prediction**(1./len(pred_list))
# Make a submission file
top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
predicted_labels = [' '.join(list(x)) for x in top_3]
test = pd.read_csv(test_path)
test['label'] = predicted_labels
test[['fname', 'label']].to_csv("ensembled_submission.csv", index=False)
