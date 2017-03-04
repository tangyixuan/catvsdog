import models as md
from utils import *

X_train = np.zeros([2,3])
print X_train
# Y_train = np.
model = md.get_vgg()
md.print_model(model)

# model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)