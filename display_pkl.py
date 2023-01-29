import pickle
from pprint import pprint
import inspect

low_dim_obs_file = 'data/episode3/low_dim_obs.pkl'
variation_descriptions_file = 'data/episode3/variation_descriptions.pkl'
variation_number_file = 'data/episode3/variation_number.pkl'

with open(low_dim_obs_file, 'rb') as f:
    low_dim_obs = pickle.load(f)
    pprint(inspect.getmembers(low_dim_obs))
with open(variation_descriptions_file, 'rb') as f:
    variation_descriptions = pickle.load(f)
    pprint(inspect.getmembers(variation_descriptions))
with open(variation_number_file, 'rb') as f:
    variation_number = pickle.load(f)
    pprint(inspect.getmembers(variation_number))
