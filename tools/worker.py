import sys
import pickle

from tqdm import tqdm

# from .barycentric import *

pi0 = int(sys.argv[1])
pi1 = int(sys.argv[2])

with open("VolumetricClosestPointQuery.pkl", "rb") as f:
    closest_point, P = pickle.load(f)

data = []
for pi, p in zip(tqdm(range(pi0, min(pi1, P.shape[0]))), P[pi0:pi1]):
    ti, bc = closest_point(p)
    data.append((pi, ti, bc.tolist()))

with open(f"rows/{pi0}-{pi1}.pkl", "wb") as f:
    pickle.dump(data, f)
