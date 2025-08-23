import matplotlib.pyplot as plt 
data = [float(x) for x in open("losses.txt").read().split(",")]
plt.plot(data)
plt.title("Loss vs epochs")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.savefig("./plot.png", bbox_inches='tight')