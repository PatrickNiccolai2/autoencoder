import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import load, save
from autoencoder import AE
from os.path import exists
'''
def load_data():
	tensor_transform = transforms.ToTensor()
	dataset = datasets.MNIST(root = "./data", train = True, download = True, transform = tensor_transform)
	loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 256)
	return loader
'''

def load_data():
	train_dataset = datasets.MNIST(
	root="./MNIST/train", train=True,
	transform=transforms.ToTensor(),
	download=True)

	train_loader = torch.utils.data.DataLoader(
	train_dataset, batch_size=256)

	test_dataset = datasets.MNIST(
	root="./MNIST/test", train=False,
	transform=transforms.ToTensor(),
	download=True)

	test_loader = torch.utils.data.DataLoader(
	test_dataset, batch_size=256)

	return train_loader, test_loader

def train():

	loader, test_loader = load_data()

	model = AE()

	loss_function = torch.nn.MSELoss()
	#optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e-8)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	epochs = 5
	outputs = []
	losses = []
	for epoch in range(epochs):
		for batch in loader:
			image, _ = batch

			# Reshaping the image to (-1, 784)
			image = image.reshape(-1, 28*28)

			# Output of Autoencoder
			reconstructed = model(image)

			# Calculating the loss function
			loss = loss_function(reconstructed, image)
			#print("loss:")
			#print(loss)

			# The gradients are set to zero,
			# the gradient is computed and stored.
			# .step() performs parameter update
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Storing the losses in a list for plotting
			losses.append(loss)

		print("epoch " + str(epoch) + ", loss " + str(loss))
		outputs.append((epochs, image, reconstructed))

	image0 = outputs[4][1]
	recons0 = outputs[4][2]
	recons0 = recons0.detach().numpy()

	for i, item in enumerate(image0):
		item = item.reshape(-1, 28, 28)
		#print(item)
		plt.imshow(item[0])
		plt.savefig("imgs/original_" + str(i) + ".png")
		if i > 10:
			break

	for i, item in enumerate(recons0):
		item = item.reshape(-1, 28, 28)
		#print(item)
		#print("item size = " + str(item.size()))
		#item = item.detach().numpy()
		plt.imshow(item[0])
		plt.savefig("imgs/recons_" + str(i) + ".png")
		if i > 10:
			break


	state_dictionary_path = "ae_state_dict.pt"
	save(model.state_dict(), state_dictionary_path)

def load():
	model_path = "ae_state_dict.pt"
	assert exists(model_path), f"The trained model {model_path} does not exist"
	#model_state = load(model_path)
	model = AE()
	model.load_state_dict(torch.load(model_path))
	model.eval()

	loader, test_loader = load_data()
	for (image, _) in loader:
		image = image.reshape(-1, 28*28)
		reconstructed = model(image)
		reconstructed = reconstructed.detach().numpy()

		for i, item in enumerate(image):
			item = item.reshape(-1, 28, 28)
			#print(item)
			plt.imshow(item[0])
			plt.savefig("imgs/original_" + str(i) + ".png")
			if i > 10:
				break

		for i, item in enumerate(reconstructed):
			item = item.reshape(-1, 28, 28)
			#print(item)
			#print("item size = " + str(item.size()))
			#item = item.detach().numpy()
			plt.imshow(item[0])
			plt.savefig("imgs/recons_" + str(i) + ".png")
			if i > 10:
				break
		
		break



if __name__=="__main__":
	train()


