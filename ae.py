from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform



learning_rate = 0.01
num_steps = 30000
training_epochs = 500
batch_size = 30

class Autoencoder(nn.Module):
    def __init__(self,num_input,num_hidden_1,num_hidden_2,num_hidden_3,num_hidden_4, lambda_reg=1.0):
        super(Autoencoder, self).__init__()
        self.lambda_reg = lambda_reg
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_input, num_hidden_1),
            nn.Sigmoid(),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.Sigmoid(),
            nn.Linear(num_hidden_2, num_hidden_3),
            nn.Sigmoid(),
            nn.Linear(num_hidden_3, num_hidden_4),
            nn.Sigmoid()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(num_hidden_4, num_hidden_3),
            nn.Sigmoid(),
            nn.Linear(num_hidden_3, num_hidden_2),
            nn.Sigmoid(),
            nn.Linear(num_hidden_2, num_hidden_1),
            nn.Sigmoid(),
            nn.Linear(num_hidden_1, num_input),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def loss_no_reg(self, inputs, encoded, outputs, indices, X, B):
        return self.criterion(inputs, outputs)

    def loss_mds_eucl(self, inputs, encoded, outputs, indices, X, C):
        loss_1 = self.criterion(inputs,outputs)

        r = torch.sum(inputs*inputs, 1)
        # turn r into column vector
        r = torch.reshape(r, [-1, 1])
        E = r - 2*torch.matmul(inputs, torch.transpose(inputs,0,1)) + torch.transpose(r,0,1)
        
        s = torch.sum(encoded*encoded, 1)
        # turn s into column vector
        s = torch.reshape(s, [-1, 1])
        F = s - 2*torch.matmul(encoded, torch.transpose(encoded,0,1)) + torch.transpose(s,0,1)
        loss_2 = self.criterion(E,F) 
        #loss_2 = loss_2 / (len(indices) * (len(indices) - 1))
        loss = loss_1 + self.lambda_reg * loss_2 
        return loss

    def loss_mds_mst(self, inputs, encoded, outputs, indices, X, B):
        loss_1 = self.criterion(inputs,outputs)
        #inp = M[indices]
        #r = torch.sum(B*B, 1)
        # turn r into column vector
        #r = torch.reshape(r, [-1, 1])
        #print(inputs.shape)
        #E = r - 2*torch.matmul(B, torch.transpose(B,0,1)) + torch.transpose(r,0,1)

        s = torch.sum(encoded*encoded, 1)
        # turn r into column vector
        s = torch.reshape(s, [-1, 1])
        F = s - 2*torch.matmul(encoded, torch.transpose(encoded,0,1)) + torch.transpose(s,0,1)
        loss_2 = self.criterion(B,F) 
        #loss_2 = loss_2 / (len(indices) * (len(indices) - 1))
        loss = loss_1  + self.lambda_reg * loss_2 
        return loss

    def loss_LE_mst(self, inputs, encoded, outputs, indices, X, B):
        loss_1 = self.criterion(inputs,outputs)
        #inp = M[indices]
        #r = torch.sum(B*B, 1)
        # turn r into column vector
        #r = torch.reshape(r, [-1, 1])
        #print(inputs.shape)
        #E = r - 2*torch.matmul(B, torch.transpose(B,0,1)) + torch.transpose(r,0,1)

        s = torch.sum(encoded*encoded, 1)
        # turn r into column vector
        s = torch.reshape(s, [-1, 1])
        F = s - 2*torch.matmul(encoded, torch.transpose(encoded,0,1)) + torch.transpose(s,0,1)
        loss_2 = torch.sum(F*B)
        #loss_2 = loss_2 / (len(indices) * (len(indices) - 1))
        loss = loss_1  + self.lambda_reg * loss_2 
        return loss

       

    def trained(self, X, y, indices, fct_loss):
        A = pd.DataFrame(distance_matrix(X, X), index=indices, columns=indices)
        D = nx.from_pandas_adjacency(A)
        T = nx.minimum_spanning_tree(D)
        G = nx.to_numpy_array(T)
        M = nx.floyd_warshall_numpy(T)
        M = np.squeeze(np.asarray(M))
        M = torch.tensor(M, dtype=torch.float32)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        for epoch in range(training_epochs):
            self.train()
            reconstruction_losses = 0
            for i in range(0, len(X), batch_size): 
                batch_indices = indices[i:i + batch_size]
                batch_x = X[batch_indices]
                batch_x = torch.tensor(batch_x, dtype=torch.float32)

                # Forward pass
                encoded, outputs = self(batch_x)
                loss = fct_loss(batch_x, encoded, outputs, indices, X, M[batch_indices][:,batch_indices])

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                reconstruction_losses += loss.item()
            avg_loss = reconstruction_losses / (len(X) / batch_size)
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}/{training_epochs}, Loss: {avg_loss:.4f}")

    def model_eval_f1(self, X, y, numberofanomalies):
        self.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X, dtype=torch.float32)
            encoded, predictions = self(X_test_tensor)
            predictions = predictions.numpy()

            mse = np.mean(np.power(X - predictions, 2), axis=1)
            df = pd.DataFrame(data=mse.flatten())
            #df.to_csv('file1.csv')
            error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y})
            d = error_df.sort_values('reconstruction_error',ascending=False)
            e = d.iloc[0:numberofanomalies,]
            print(e)
            return e[e.true_class == 1].shape[0]/numberofanomalies
