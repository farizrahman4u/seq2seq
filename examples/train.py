from model import get_model
from data import get_batch
import os.path



batch_size = 32

model = get_model(teacher_force=True)


file_name = 'weights.dat'
file_name = os.path.abspath(os.path.join(__file__, os.pardir)) + '/' + file_name


counter = 0

load = True

if load:
	model.load_weights(file_name)

while True:
	batch = get_batch(100)
	model.fit([batch[0], batch[1]], batch[1], nb_epoch=1)
	#model.save_weights('weights.dat')
	#continue
	counter += 1
	if counter % 100 == 0:
		model.save_weights(file_name)
		counter = 0
