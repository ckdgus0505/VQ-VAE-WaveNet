day= `date +%m_%d`

train:
	python3 train.py -dataset VCTK -length 6656 -batch 6 -step 10000 -save $(day)/weights

board:
	tensorboard --logdir=./$(day)

clean :
	rm -r $(day)
