

# region imports
import h5py
import numpy as np
import requests
import matplotlib.pyplot as plt
import requests

# region set-up
print('here')
from Helpers.APIHelper import APIHelper

api = APIHelper()
r = api.get(api.baseUrl)
print(r)

# get simulation
names = [sim['name'] for sim in r['simulations']]
simName = "TNG50-1"
slctdSim = names.index(simName)
sim = api.get(r['simulations'][slctdSim]['url'])

# get snapshots
snaps = api.get(sim['snapshots'])
print(f"{len(snaps)} snapshots from simulations {simName}")

# choose a snapshot and get subhalos
snapNum = 87
snap = api.get(snaps[snapNum]['url'])
subs = api.get(snap['subhalos'])

print(subs)