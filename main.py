import pandas as pd
import numpy as np
import torch
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import torch.nn.modules as nn
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(30)

insame_weight = ["diameter","speed"]    #number good 数值特征
same_weight = ["color"]             #要独热编码的  类别特征

X = data[insame_weight+same_weight].copy()
y = data["power"]

x=np.array(X)
y=np.array(y)


def get_preprocessor():
    return ColumnTransformer([
        ("same", Pipeline([

            ("ohe", OneHotEncoder(sparse_output=False)) #独热编码，返回稠密数组（不是稀疏矩阵，方便转 torch tensor）
        ]), same_weight),
        ("insame", Pipeline([

            ("scale", StandardScaler())
        ]), insame_weight),
    ])

X_k, X_test, y_k, y_test = train_test_split(
    X, y, test_size=0.15, random_state=70, shuffle=True
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_k, y_k, test_size=0.15, random_state=70, shuffle=True
)
print("y_train mean, std:", y_train.mean().item(), y_train.std().item())
print("y_valid mean, std:", y_valid.mean().item(), y_valid.std().item())
print("y_test  mean, std:", y_test.mean().item(), y_test.std().item())

n_train = len(X_train)
n_valid = len(X_valid)

level_1 = 28
level_2 = 14
drop_num=0.02
lr_num=0.01
class Net(nn.Module):
     def __init__(self,fea):
         super().__init__()
         self.net = nn.Sequential(
             nn.Linear(fea,level_1),
             nn.BatchNorm1d(level_1),
             nn.ReLU(),
             nn.Dropout(drop_num),
             nn.Linear(level_1,level_2),
             nn.BatchNorm1d(level_2),
             nn.ReLU(),
             nn.Dropout(drop_num),
             nn.Linear(level_2,1)
         )
         self.train_loss_history = []
         self.val_loss_history = []
         self.epoch_record = []
     def forward(self,x):
         return self.net(x)

feature=6
model = Net(feature)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_num)
loss_fn=nn.MSELoss()
preprocessor = get_preprocessor()
preprocessor.fit(X_train)
X_train = preprocessor.transform(X_train)
X_valid = preprocessor.transform(X_valid)
X_test = preprocessor.transform(X_test)

X_train = torch.as_tensor(X_train, dtype=torch.float32)
X_valid = torch.as_tensor(X_valid, dtype=torch.float32)
X_test  = torch.as_tensor(X_test,  dtype=torch.float32)

y_train = torch.as_tensor(np.asarray(y_train), dtype=torch.float32).reshape(-1, 1)
y_valid = torch.as_tensor(np.asarray(y_valid), dtype=torch.float32).reshape(-1, 1)
y_test  = torch.as_tensor(np.asarray(y_test),  dtype=torch.float32).reshape(-1, 1)

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("前面已完成，进入训练")
best_final_loss=float("inf")
cnt=0
best_final_w=None
for epoch in range(1000):

    model.train()
    total_loss = 0
    total_num=0
    for bx,by in train_loader:
        pred = model(bx)
        loss = loss_fn(pred, by)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = bx.size(0)
        total_loss += loss.item() * batch_size


    avg_loss = total_loss / n_train
    if epoch % 1 == 0:
        print("epoch:", epoch, "train_loss:",avg_loss)
    if epoch % 10 == 0:
        model.train_loss_history.append(avg_loss)

    model.eval()
    total_val_loss = 0
    total_num=0
    with torch.no_grad():
        pred=model(X_valid)
        loss_2=loss_fn(pred, y_valid).item()
        print("epoch:", epoch, "val_loss:", loss_2)
    if epoch % 10 == 0:
        model.val_loss_history.append(loss_2)
        model.epoch_record.append(epoch)

    if loss_2 < best_final_loss:
        best_final_loss = loss_2
        best_final_w = copy.deepcopy(model.state_dict())
        cnt = 0
    else:
        cnt += 1
        if cnt >= 100:
            print(f"Final model early stop epoch {epoch}")
            break

    model.eval()
    train_eval_total = 0.0
    with torch.no_grad():
        for bx, by in train_loader:
            pred = model(bx)
            loss = loss_fn(pred, by)
            train_eval_total += loss.item() * bx.size(0)
    train_eval_loss = train_eval_total / n_train
    print(f"epoch:{epoch:4d} | train_drop_loss:{avg_loss:.2f} | train_eval_loss:{train_eval_loss:.2f} | val_loss:{loss_2:.2f}")

model.load_state_dict(best_final_w)
model.eval()
with torch.no_grad():
     pred=model(X_test)
     loss_3=loss_fn(pred, y_test).item()
     print("test_loss:", loss_3)


plt.figure()
plt.plot(model.epoch_record, model.train_loss_history,label="Train Loss")
plt.plot(model.epoch_record, model.val_loss_history,label="Val Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(bottom=0, top=100)
plt.title("MLP")
plt.grid(alpha=0.3)
plt.show()

