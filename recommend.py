#Konstan, MovieLens
import requests
import zipfile
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm_notebook as tdqm

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans #fits clusters based on movie weight


url = http://files.grouplens.org/datasets/movielens/ml-latest-small.zip -o ml-latest-small.zip

with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref: #extracts data from MovieLens file
    zip_ref.extractall( 'data')

movies_df = pd.read_csv('data/ml-latest-small/movies.csv') #defines variables for each movie and its rating
ratings_df = pd.read_csv ('data/ml-latest-small/ratings.csv')

print('The dimensions of movies dataframe are:', movies_df.shape, '\nThe dimensions of ratings dataframe are:', ratings_df.shape)

movies_df.head()#shows the movieId and corresponding information
ratings_df.head()

#maps ID to title; outputs number of users, ratings, and movies
movie_names = movies_df.set_index('movieId')['title'].to_dict()
n_users = len(ratings_df.userId.unique())
n_items = len(ratings_df.movieId.unique())
print("Number of unique users:", n_users)
print("Number of unique movies:", n_items)
print("The full rating matrix will have:", n_users*n_items, 'elements.')
print('---------')
print("Number of ratings:", len(ratings_df))
print("Therefore:", len(ratings_df) / (n_users*n_items) * 100, '% of the matrix is filled.')
print("Therefore:", len(ratings_df) / (n_users*n_items) * 100, '% of the matrix is filled.')

#matrix factorization
class MatrixFactorization(torch.nn.Module): #for user preference and characteristics
    def _init_(self, n_users, n_items, n_factors=20):
        super()._init_()
        self.user_factors = torch.nn.Embedding(n_users, n_factors) #lookup table
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_factors.weight.data.uniform_(0, 0.05) #tunable weights
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, data): 
        users, items = data[:,0], data[:,1]
        return (self.user_factors(users)*self.item_factors(items)).sum(1) #matrix multiplication

    def predict(self, user, item):
        return self.forward(user, item)

class Loader(Dataset): #reading in the ratings dataframe
    def _init_(self):
        self.ratings = ratings_df.copy()

        users = ratings_df.userId.unique() #extract userId
        movies = ratings_df.movieId.unique()

        self.userid2idx = {o:i for i,o in enumerate(users)} #unique index
        self.movieid2idx = {o:i for i,o in enumerate(movies)}

        self.idx2userid = {i:o for o,i in self.userid2idx.items()} #convert to make sure continuous ID
        self.idx2movieid = {i:o for o,i in self.userid2idx.items()}

        self.ratings.movieId = ratings_df.movieId.apply(lambda x: self.movieid2idx[x]) #return ID
        self.ratings.userId = ratings_df.userId.apply(lambda x: self.uderid2idx[x])

        self.x = self.ratings.drop(['rating', 'timestamp'], axis=1).values
        self.y = self.ratings['rating'].values
        self.x, self.y = torch.tensor(self.x), torch.tensor(self.y) #return in tensor format
    
    def _getitem_(self, index): #helper 
        return (self.x[index], self.y[index])
    
    def _len(self): #helper
        return len(self.ratings)

num_epochs = 128
cuda = torch.cuda.is_available()

print("is running on GPU:", cuda)

model = MatrixFactorization(n_users, n_items, n_factors=8)
print(model)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
if cuda:
      model = model.cuda()

loss_fn = torch.nn.MSELoss() #error

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #minimize loss

train_set = Loader() #train data
train_loader = DataLoader(train_set, 128, shuffle=True)

for it in tqdm(range(num_epochs)):
    losses = []
    for x, y in train_loader: #load values (userId/movieId, rating)
        if cuda:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
            losses.append(loss.items()) 
            loss.backward() #gradient of tensor
            optimizer.step()
    print("iter #{}".format(it), "Loss:", sum(losses) / len(losses)) 


c = 0
uw = 0
iw = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
        if c == 0:
            uw = param.data
            c +=1
        else:
            iw = param.data
            

trained_movie_embeddings = model.item_factors.weight.data.cpu().numpy() #convert weight to numpy

len(trained_movie_embeddings) #unique movie factor weight

kmeans = KMeans(n_clusters=10, random_state=0).fit(trained_movie_embeddings)

for cluster in range(10): #clusters 10 movies based on familiarity/ likeliness of user watching
    print("Cluster #{}".format(cluster))
    movs = []
    for movidx in np.where(kmeans.labels_ == cluster)[0]:
        movid = train_set.idx2movieid[movidx]
        rat_count = ratings_df.loc[ratings_df['movieId']==movid].count()[0]
        movs.append((movie_names[movid], rat_count))
    for mov in sorted(movs, key=lambda tup: tup[1], reverse=True)[:10]:
        print("\t", mov[0])

