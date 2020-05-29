day= `date +%m_%d`

make all: clean train 

train:
	python3 train.py -dataset VCTK -length 6656 -batch 6 -epoch 100 -save $(day)/weights -restore saved_weight/weights-1

board:
	tensorboard --logdir=./$(day)

clean :
	rm -r $(day)
