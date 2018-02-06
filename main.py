# bibliotecas utilizadas:
import numpy as np
import scipy as sp
import pylab as pl
from sklearn import cluster
 
 
# n_clusters = número de clusters para a divisão da imagem no K_means = número de cores para a quantização da imagem
n_clusters = 8
np.random.seed(0)
 
#tenta achar lena na lib
try:
    lena = sp.lena()
except AttributeError:
    from scipy import misc
    lena = misc.lena()
 
x, y = lena.shape # pega o tamanho da lena
    
X = lena.reshape((-1, 1))  # array(n_sample, n_feature)
k_means = cluster.KMeans( n_clusters, n_init = 4) #inicia k_means com n_cluster divisões
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
print 'cores da lena quantizada:'
print values
print '         '   
   
 
 
# cordenadas minimas e maximas da lena
vmin = lena.min()   
vmax = lena.max()
 
# apos quantização com k_means cria 3 figuras
 
# figura 1: lena normal
pl.figure(1, figsize=(5, 4))
pl.imshow(lena, cmap=pl.cm.gray, vmin=vmin, vmax=256)
 
 
# figura 2: quantiza lena (com 'n_clusters' cores que são: as 'values' cores)
regular_lena = np.choose(labels.ravel(), values)
regular_lena.shape = lena.shape
pl.figure(2, figsize=(5, 4))
pl.imshow(regular_lena, cmap=pl.cm.gray, vmin=vmin, vmax=vmax)
 
# figura 3: plota histograma (cinza = figura 1, linhas azuis = cores da figura 2)
pl.figure(3, figsize=(5, 4))
pl.clf()
pl.axes([.01, .01, .98, .98])
pl.hist(X, bins=256, color='.5', edgecolor='.5') #histograma figura 1
pl.yticks(())
pl.xticks(values)
values = np.sort(values)
for x in range(n_clusters):   #plota as linhas das cores da figura 2 em azul
    pl.axvline(values[x])
 
 
# função quadratica
soma = 0
for i in range(lena.shape[0]):
    for j in range(lena.shape[1]):
        soma += int((lena[i][j] - regular_lena[i][j])**2)                    
m = (1./(lena.shape[0]*lena.shape[1]))*soma
 
print ('mse: ')
print m
 
 
pl.show()


Read more: https://felipemk.webnode.com.br/news/quantiza%c3%a7%c3%a3o-de-imagens%2c-utilizando-algoritmo-k-means/