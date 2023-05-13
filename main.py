from train import Trainer
import argparse


def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type = int, default = 4)

	parser.add_argument('--learning_rate', type = float, default = 0.001)

	parser.add_argument('--momentum', type = float, default = 0.9)

	parser.add_argument('--n_channels', type = int, default  = 1)

	parser.add_argument('--n_epochs', type = int, default = 5)

	return parser


if __name__ == "__main__":

	parser = get_parser().parse_args()
	trainer = Trainer(parser.batch_size, parser.n_epochs, parser.learning_rate, parser.momentum)
	trainer.run(parser.n_epochs)



	

