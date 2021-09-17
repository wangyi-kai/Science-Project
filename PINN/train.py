from libs import *

class Train():
    def __init__(self, net, equation, batch_size):
        self.errors = []
        self.net = net
        self.model = equation
        self.batch_size = batch_size

    def train(self, epoch, lr):
        optimizer = optim.Adam(self.net.parameters(), lr)
        avg_loss = 0
        start = time.time()
        for e in range(epoch):
            optimizer.zero_grad()
            loss = self.model.loss(self.batch_size)
            avg_loss = avg_loss + float(loss.item())
            loss.backward()
            optimizer.step()
            end = time.time() - start
            if(e % 50 == 0):
                loss = avg_loss / 50
                print("Epoch {} - lr {} -  loss: {} - time:{}".format(e, lr, loss, end))
                avg_loss = 0
