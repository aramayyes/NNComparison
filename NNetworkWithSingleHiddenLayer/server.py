from aiohttp import web
import socketio
import numpy
import imageio
from skimage.transform import resize
import matplotlib.pyplot
import base64

from network import NeuralNetwork
from mnist import MnistdataManager

# server
sio = socketio.AsyncServer(logger=True)
app = web.Application()
sio.attach(app)

# network parameters
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
epochs = 5

m = MnistdataManager()
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, epochs)


async def index(_):
    return web.Response(text="Server is running...")


app.router.add_get('/', index)


# socket events
@sio.on('connect')
def connect(sid, _):
    print("connected ", sid)


imageNumber = 0


@sio.on('predict')
async def predict(sid, data):
    await sio.emit('predicting', room=sid)
    await sio.sleep(0)

    global imageNumber

    img_data = base64.b64decode(data)
    filename = str(imageNumber) + '.png'
    imageNumber += 1
    with open(filename, 'wb') as f:
        f.write(img_data)

    img = imageio.imread(filename)
    img = resize(img, (28, 28), mode='reflect')  # resize the image to 28x28
    img = 1. - img.astype(numpy.float32)  # invert black and white
    img = numpy.mean(img, axis=2)  # greyscale
    img = img.flatten()

    matplotlib.pyplot.imshow(img.reshape(28, 28), cmap='Greys', interpolation='None')
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.title('Input image')
    matplotlib.pyplot.show()

    o = n.predict(img)
    network_answer = numpy.argmax(o)

    print(o)
    print(network_answer)

    await sio.emit('predicted', str(network_answer), room=sid)


@sio.on('accuracy')
async def get_accuracy(sid):
    await sio.emit('calculated_accuracy', data=str(acc), room=sid)


@sio.on('disconnect')
def disconnect(sid):
    print('disconnected', sid)


acc = 0
if __name__ == '__main__':
    train_before_start = True

    if train_before_start:
        print("Training...")
        m.train(n)
        acc = m.calculate_accuracy(n)
        print("Trained")

    web.run_app(app, host='localhost')
