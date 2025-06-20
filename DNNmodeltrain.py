#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset, DataLoader
from helpers import *
from tqdm import tqdm
from icecream import ic


# In[2]:


data = np.genfromtxt('data.txt', delimiter=' ')
# reshape set up hands to be normalized
data = data.reshape((-1, 21, data.shape[1]))
print(data)
# np.random.shuffle(data)
# print(data.shape)
data = torch.from_numpy(data)
# data = data[torch.randperm(data.size()[0])] # shuffles train_dataset


# In[3]:


X, y = data[..., :-1], data[..., -1]
# inspect distribution of data, if unbalanced then balance
for fingers_count in range(6):
    ic(y[y == fingers_count].shape[0])


# In[4]:


# Balance data
y_shapes = []
temp_y = y[:, 0]
unbal_indices = []
for fingers_count in torch.arange(np.unique(temp_y).shape[0]):
    unbal_indices.append(np.where(temp_y == fingers_count)[0])
    shape_ = unbal_indices[fingers_count].shape[0]
    y_shapes.append(shape_)

ic(y_shapes)
bal_indices = []
min_shape = min(y_shapes)
for fingers_count in range(len(unbal_indices)):
    bal_indices.append(unbal_indices[fingers_count][0:min_shape])

# ic(bal_indices)
bal_indices = np.array(bal_indices).flatten()
ic(bal_indices.shape)
y_new = y[bal_indices]
X_new = X[bal_indices]
ic(X_new.shape)
ic(y_new.shape)


# In[5]:


X = X_new
y = y_new

for fingers_count in range(6):
    ic(y[y == fingers_count].shape[0])


# In[6]:


# Normalize each hand relative to itself. Removes dependency of hand positioning in camera field of view
for i, hand in enumerate(X):
    X[i] = normalize_hand(hand)


# In[7]:


class HandDataset(Dataset):
    def __init__(self, data):
        self.x, self.y = data[..., :-1], data[..., -1]
        self.n_samples = y.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        ic(self.x.shape, self.y.shape)
        reshaped_y = self.y.reshape(-1, 21, 1)
        ic(reshaped_y.shape)
        return torch.cat([self.x[index], reshaped_y[index]], dim=-1)

    def split(self, test_size=.2):
        split = int(self.x.shape[0] * (1 - test_size))
        return self[:split, :], self[split:, :]  # train, test


# Combine X and y to preserve data during shuffle
reshaped_y = y.reshape(-1, 21, 1)
data = torch.cat([X, reshaped_y], dim=-1)

# Shuffle
dataset = HandDataset(data[torch.randperm(data.shape[0])])  # shuffles train_dataset
train_dataset, test_dataset = dataset.split()

# hyper parameters
### DEFINED IN helpers.py

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
# for i in range(len(train_loader)):
#     print(train_loader.dataset[i, 0, 3].item())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# In[8]:


ic(len(train_dataset))
ic(len(train_loader))
ic(len(test_dataset))
ic(len(test_loader))


# In[9]:


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


model = NeuralNet(ic(input_size), ic(hidden_size), ic(num_classes)).to(device)
# loss and optimization
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def normalize_hands(hands):
    for i, hand in enumerate(hands):
        hands[i] = torch.from_numpy(normalize_hand(hand.np()).astype(np.float32()))
    return hands


# training loop
n_total_steps = len(train_loader)
for epoch in tqdm(range(num_epochs)):
    for i, sample in enumerate(train_loader):
        # extract data from sample
        hands = sample[..., :-1].float().to(device)
        labels = sample[..., -1][:, 0].long().to(device)

        # hands = normalize_hands(hands).float()
        hands = hands.reshape(-1, input_size)

        # forward pass
        outputs = model(hands).float()
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # if epoch % 5 == 0:
    #     print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')

print('Training Complete!')


# In[10]:


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    i = 1
    for sample in test_loader:
        hands = sample[..., :-1].float().to(device)
        labels = sample[..., -1][:, 0].long().to(device)

        # hands = normalize_hands(hands).float()
        hands = hands.reshape(-1, input_size)

        outputs = model(hands)

        # value, index
        _, pred = torch.max(outputs, 1)

        # ic(i, labels.shape[0], (pred == labels).sum().item())
        i += 1
        n_samples += labels.shape[0]
        n_correct += (pred == labels).sum().item()

    acc = 100 * n_correct / n_samples

    print('accuracy =', acc)


# In[11]:


save_model(model.state_dict(), 'finalized_model.pth')

