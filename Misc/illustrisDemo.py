
from Helpers.APIHelper import APIHelper
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import h5py
import time

start = time.time()

# API Helper
api = APIHelper()
r = api.get(api.baseUrl)

# ---------------------------------------------------------------------------------------------
# get all simulation names
names = [sim['name'] for sim in r['simulations']]

# get TNG50-1 using entries from r instead of contrustcing url
simName = 'TNG50-1'
simName = 'Illustris-1'
slctdSim = names.index(simName)
sim = api.get(r['simulations'][slctdSim]['url'])
#print(sim.keys())
print(f"number of sims: {sim['num_dm']}")

# ---------------------------------------------------------------------------------------------
# get snapshots for simulation
snapUrl = sim['snapshots']
snaps = api.get(snapUrl)
print(f"Number of snapshots in {simName}: {len(snaps)}")

# get the last snapshot
# this retrives all the metadata behind the snapshot including numeric data and urls 
snap = api.get(snaps[-1]['url'])
#print(snap)

# ---------------------------------------------------------------------------------------------
# get the subhalos 
# subhalos are childen of the snapshot we have chosen above
subs = api.get(snap['subhalos'])
subsCount = subs['count']
subsPrev = subs['previous']
subsResults = subs['results']
subsNext = subs['next']
print(f"Number of subhalos: {subsCount} With {len(subsResults)} Subfind subhalos for snapshot")
# can increase number of Subfind subhalos using limit
subs = api.get(snap['subhalos'], {'limit':220})
print(subs['next'])
# maximum number of subs resulst related to limit set (eg 220 result items can be accessed atm)
print(subs['results'][0])

# get the first 20 subhalos and sort by stellar mass desc
subs = api.get( snap['subhalos'], {'limit':20, 'order_by':'-mass_stars'} )
orderedSubs = [subs['results'][i]['id'] for i in range(20) ]

# most massive subhalo with ID == 0 has the most stars 
# can access all attributes like a dictionary
sub = api.get( subs['results'][1]['url'] )
print(f"sub id: {sub['id']}")
for item in sub.items():
    #print(item)
    pass

# ---------------------------------------------------------------------------------------------
# Access parent halo (FoF)
parentUrl = sub['related']['parent_halo'] + "info.json"
parent_fof = api.get(parentUrl)
#print(parent_fof.keys())
#parentGroup = parent_fof['GroupCM']
firstSub = parent_fof['GroupFirstSub']
#numberOfSubs = parent_fof['GroupNSubs']
print(f"First sub halo: {firstSub}")

# ---------------------------------------------------------------------------------------------
# get the hdf5 data
mpb1 = api.get( sub['trees']['sublink_mpb'] )
f = h5py.File(mpb1,'r')
print(len(f['SnapNum']))
mpb2 = api.get( sub['trees']['lhalotree_mpb'] )
with h5py.File(mpb2,'r') as f:
     print(len(f['SnapNum']))

# ---------------------------------------------------------------------------------------------
# Evolution of the subhalo position along each axis back in time
with h5py.File(mpb2,'r') as f:
    pos = f['SubhaloPos'][:]
    snapnum = f['SnapNum'][:]
    subid = f['SubhaloNumber'][:]
 
for i in range(3):
    plt.plot(snapnum,pos[:,i] - pos[0,i], label=['x','y','z'][i])
plt.legend()
plt.xlabel('Snapshot Number')
plt.ylabel('Pos$_{x,y,z}$(z) - Pos(z=0)')
#plt.show()

# ---------------------------------------------------------------------------------------------
#Image of gas density around z=1 progenitor of our subhalo 
url = sim['snapshots'] + "z=1/"
snap = api.get(url)
snapNumber = snap['number']
snapRedShift = snap['redshift']
print(f"Number: {snapNumber}, RedShift: {snapRedShift}")
# find the target subfind id at snap numbers using sublink tree
i = np.where(snapnum == snapNumber)
id = subid[i]
sub_prog_url = sim['snapshots'] + f'/{snapNumber}' + '/subhalos' + f'{id}'
sub_prog = api.get(sub_prog_url)
#print(sub_prog['pos_x'], sub_prog['pos_y'])
# Request the subhalo details, and a snapshot cutout consisting only of Gas fields Coordinates,Masses.
cutout_request = {'gas':'Coordinates,Masses'}
cutout = api.get(sub_prog_url+"cutout.hdf5", cutout_request)
print(cutout)

#2d histogram
with h5py.File(cutout,'r') as f:
    x = f['PartType0']['Coordinates'][:,0] - sub_prog['pos_x']
    y = f['PartType0']['Coordinates'][:,1] - sub_prog['pos_y']
    dens = np.log10(f['PartType0']['Masses'][:])

plt.hist2d(x,y,weights=dens,bins=[150,100])
plt.xlabel('$Delta x$ [ckpc/h]')
plt.ylabel('$Delta y$ [ckpc/h]')
plt.show()

z = 3 
ids = [41092,338375,257378,110568,260067]

sub_count = 1
plt.figure(figsize=[15,3])

for id in ids:
    url = "http://www.tng-project.org/api/TNG50-1/snapshots/z=3/subhalos/" + str(id)
    sub = api.get(url)
    
    # it is of course possible this data product does not exist for all requested subhalos
    if 'stellar_mocks' in sub['supplementary_data']: 
        # download PNG image, the version which includes all stars in the FoF halo (try replacing 'fof' with 'gz')
        png_url = sub['supplementary_data']['stellar_mocks']['image_fof']
        response = api.get(png_url)
        # make plot a bit nicer
        plt.subplot(1,len(ids),sub_count)
        plt.text(0,-20,"ID="+str(id),color='blue')
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        sub_count += 1
        
        # plot the PNG binary data directly, without actually saving a .png file
        file_object = BytesIO(response.content)
        plt.imshow(mpimg.imread(file_object))



end = time.time()
print(f"\nElapsed time: {end - start:.2f} seconds")