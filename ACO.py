import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# Fungsi untuk mengganti np_choice agar bisa digunakan dalam kelas ACO
def np_choice(a, size, replace=True, p=None):
    idx = np.arange(a)
    return np.random.choice(idx, size=size, replace=replace, p=p)

# Membaca dataset
df = pd.read_csv('tourism_with_id.csv')

# Memfilter data untuk kota Surabaya saja
df_surabaya = df[df['City'] == 'Surabaya']

# Mengambil kolom yang diperlukan
required_columns = ['Place_Id', 'Place_Name', 'Price', 'Rating', 'Lat', 'Long']
df_surabaya = df_surabaya[required_columns]

# Memfilter data berdasarkan rating dan harga
df_surabaya = df_surabaya[(df_surabaya['Rating'] >= 4) & (df_surabaya['Rating'] <= 5) & 
                          (df_surabaya['Price'] >= 0) & (df_surabaya['Price'] <= 125000)]

# Mengurutkan berdasarkan rating tertinggi
df_surabaya = df_surabaya.sort_values(by='Rating', ascending=False)

# Get coordinates, prices, and ratings
coords = df_surabaya[['Lat', 'Long']].values
prices = df_surabaya['Price'].values
ratings = df_surabaya['Rating'].values

# Calculate distance matrix
distances = cdist(coords, coords, 'euclidean')

# Ant Colony Optimization Class
class AntColonyOptimization:
    def _init_(self, distances, prices, ratings, n_ants, n_best, n_iterations, decay, alpha=1, beta=1, gamma=1):
        self.distances = distances
        self.prices = prices
        self.ratings = ratings
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.best_distances = []  # To track the best distance on each iteration

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path            
            self.pheromone *= self.decay
            self.best_distances.append(all_time_shortest_path[1])  # Save the best distance on each iteration
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, score in sorted_paths[:n_best]:
            for move in path:
                if self.distances[move[0], move[1]] != np.inf:  # Avoid division by zero
                    self.pheromone[move[0], move[1]] += 1.0 / (self.distances[move[0], move[1]] + 1e-10)  # Add small value to avoid zero

    def gen_path_score(self, path):
        total_dist = 0
        total_price = 0
        total_rating = 0
        for i, j in path:
            total_dist += self.distances[i, j]
            total_price += self.prices[j]
            total_rating += self.ratings[j]
        score = self.alpha * total_dist + self.beta * total_price - self.gamma * total_rating
        return score

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_score(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], self.prices, self.ratings, visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))  # going back to where we started    
        return path

    def pick_move(self, pheromone, dist, prices, ratings, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        # Menghindari pembagian dengan nol
        dist = np.where(dist == 0, np.inf, dist)
        prices = np.where(prices == 0, np.inf, prices)

        desirability = (pheromone ** self.alpha) * ((1.0 / dist) ** self.beta) * ((ratings / prices) ** self.gamma)
        desirability_sum = desirability.sum()

        # Jika semua desirability adalah nol, pilih secara acak
        if desirability_sum == 0:
            norm_desirability = np.ones_like(desirability) / len(desirability)
        else:
            norm_desirability = desirability / desirability_sum

        move = np_choice(len(self.all_inds), 1, p=norm_desirability)[0]
        return move

# ACO parameters
n_ants = 10
n_best = 3
n_iterations = 100
decay = 0.95
alpha = 1  # weight for distance
beta = 1   # weight for price
gamma = 10  # weight for rating

# Initialize ACO
aco = AntColonyOptimization(distances, prices, ratings, n_ants, n_best, n_iterations, decay, alpha, beta, gamma)
shortest_path = aco.run()

# Print the results
print("Shortest path: ", shortest_path)
print("Starting point of the shortest path: ", shortest_path[0][0][0])

# Convert shortest_path to a DataFrame and save as CSV
path_indices = [i for i, j in shortest_path[0]]
shortest_path_df = df_surabaya.iloc[path_indices]
shortest_path_df.to_csv('shortest_path1.csv', index=False)

print("Hasil shortest path telah disimpan ke 'shortest_path.csv'")