import pandas as pd
from scipy.io import loadmat, savemat
from scipy import stats
from scipy.spatial import Delaunay, ConvexHull
from pathlib import Path
import numpy as np

import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

_cwd = Path.cwd()
_files = _cwd.glob('*.mat')

nuclei = pd.DataFrame([])
colony = pd.DataFrame([])
for f in _files:
    file_name = str(f.name)
    folder = str(f.parent.name)
    group = str(f.parent.parent.name)
    mat = loadmat(str(f), squeeze_me=True)
    # Nuclei
    tmp_nuc = pd.DataFrame({'Volume':mat['nuclei']['Area'], 'C':mat['nuclei']['Centroid'],
                            'BB':mat['nuclei']['BoundingBox'], 'x':mat['nuclei']['x'], 'y':mat['nuclei']['y'], 'z':mat['nuclei']['z'],
                            'file':file_name, 'folder':folder, 'group':group})
    nuclei = nuclei.append(tmp_nuc, ignore_index=True)

nuclei.Volume = nuclei.Volume.astype(np.float)
nuclei[['Cx','Cy','Cz']] = pd.DataFrame(nuclei.C.to_list(), index=nuclei.index)
nuclei[['BB1','BB2','BB3','BBx','BBy','BBz',]] = pd.DataFrame(nuclei.BB.to_list(), index=nuclei.index)
nuclei = nuclei.drop(['C','BB','BB1','BB2','BB3'], axis='columns')

filtered = nuclei[ (nuclei.Volume>125) & (nuclei.Volume<5e3) ]
filtered.Volume = filtered.Volume / 0.125 # Convert to cubic microns (0.5*0.5*0.5)

# Compute distance to centroid
filtered['dx2'] = (filtered.Cx-filtered.groupby(['file','folder'])['Cx'].transform('mean'))**2
filtered['dy2'] = (filtered.Cy-filtered.groupby(['file','folder'])['Cy'].transform('mean'))**2
filtered['dz2'] = (filtered.Cz-filtered.groupby(['file','folder'])['Cz'].transform('mean'))**2

filtered['d'] = np.sqrt( filtered['dx2'] + filtered['dy2'] + filtered['dz2'] )

# Filter outliers
filtered = filtered[np.abs(stats.zscore(filtered.d)) < 3]

# Re-Compute distance to centroid
filtered['dx2'] = (filtered.Cx-filtered.groupby(['file','folder'])['Cx'].transform('mean'))**2
filtered['dy2'] = (filtered.Cy-filtered.groupby(['file','folder'])['Cy'].transform('mean'))**2
filtered['dz2'] = (filtered.Cz-filtered.groupby(['file','folder'])['Cz'].transform('mean'))**2

filtered['d'] = np.sqrt( filtered['dx2'] + filtered['dy2'] + filtered['dz2'] )

filtered['diam'] = np.mean(filtered.loc[:,['BBx','BBy','BBz']].to_numpy(), axis=1)

### Compute Colony statistics from filtered nuclei information
x = filtered.loc[0,'x']
y = filtered.loc[0,'y']
z = filtered.loc[0,'z']
ch = ConvexHull(np.array([x,y,z]).T)

# Get PDF
step_size = filtered.diam.mean()/2
bins= np.arange(filtered.d.min(), filtered.d.max()+step_size,  step_size)
bins = np.round(bins)
filtered['id-tag'] = filtered['folder'].str.cat(filtered['file'],sep='-')
output = filtered.groupby('id-tag').d.apply(lambda x: np.histogram(x, bins=bins, density=True)[0])

nuclei = filtered[['folder','file','group','Volume','Cx','Cy','Cz','d']]

colonies = pd.DataFrame(output.tolist(), columns=['pdf00','pdf01','pdf02','pdf03','pdf04','pdf05','pdf06','pdf07','pdf08','pdf09','pdf10','pdf11','pdf12'], index=output.index)
colonies['organization'] = colonies.apply(np.max, axis=1)
colonies['folder'] = colonies.index.to_series().apply(lambda x: str(x).split('-')[0])
colonies['file'] = colonies.index.to_series().apply(lambda x: str(x).split('-')[1])

grouped = filtered.groupby(['folder','file'])
for name, group in grouped:
    # ConvexHull Volume
    x = None
    for newx in group['x']:
        if x is None:
            x = newx;
        else:
            x = np.concatenate((x,newx), axis=0)
    y = None
    for newy in group['y']:
        if y is None:
            y = newy;
        else:
            y = np.concatenate((y,newy), axis=0)
    z = None
    for newz in group['z']:
        if z is None:
            z = newz;
        else:
            z = np.concatenate((z,newz), axis=0)
    points = np.concatenate((x.reshape((-1,1)), y.reshape((-1,1)), z.reshape((-1,1))),axis=1)
    colonies.loc[(colonies['file']==name[1])&(colonies['folder']==name[0]), 'ConvexHullVolume'] = ConvexHull(points).volume / 0.125
    colonies.loc[(colonies['file']==name[1])&(colonies['folder']==name[0]), 'Density'] = group.Volume.sum()/(ConvexHull(points).volume / 0.125)
    colonies.loc[(colonies['file']==name[1])&(colonies['folder']==name[0]), 'NumbNuclei'] = group.Volume.count()

    # Elongation
    mean = np.mean(points, axis=0)
    S = 1/points.shape[0]*np.dot((points-mean).T, (points-mean))
    E,_ = np.linalg.eig(S)
    colonies.loc[(colonies['file']==name[1])&(colonies['folder']==name[0]), 'Roundness'] = np.min(E) / np.max(E)

nuclei.to_csv('nuclei.csv',index=False)

xx = nuclei.loc[:, ['folder','file','group']].drop_duplicates()
colonies = colonies.merge(xx, on=['folder','file'])
colonies.to_csv('colonies.csv', index=False)
