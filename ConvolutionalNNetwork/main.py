from aiohttp import web
import socketio
import numpy as np
import base64
import imageio
from skimage.transform import resize

from network_manager import NetworkManager

# network
network_manager = NetworkManager(4000, 64, 0.0001)

# server
sio = socketio.AsyncServer(logger=True)
app = web.Application()
sio.attach(app)


async def index(_):
    return web.Response(text="Server is running...")


app.router.add_get('/', index)


# socket events
@sio.on('connect')
def connect(sid, environ):
    print("connected ", sid)


imageNumber = 0


@sio.on('predict')
async def predict(sid, data):
    await sio.emit('predicting', room=sid)
    await sio.sleep(0)

    global imageNumber

    img_data = base64.b64decode(data)
    filename = 'last' + str(imageNumber) + '.png'
    imageNumber += 1
    with open(filename, 'wb') as f:
        f.write(img_data)

    img = imageio.imread(filename)

    img = resize(img, (28, 28), mode='reflect')  # resize the image to 28x28
    img = 1. - img.astype(np.float32)  # invert black and white
    img = np.mean(img, axis=2)  # greyscale
    img = img.flatten()

    # test the digit(image) though the neural network
    prediction_arr, answer = network_manager.predict(img)
    print(answer)
    await sio.emit('predicted', str(answer), room=sid)


@sio.on('accuracy')
async def get_accuracy(sid):
    await sio.emit('calculated_accuracy', data=str(network_manager.get_accuracy()), room=sid)


@sio.on('disconnect')
def disconnect(sid):
    print('disconnected', sid)


if __name__ == '__main__':
    train_before_start = True

    if train_before_start:
        print("Training...")
        network_manager.train()
        print("Trained")

    web.run_app(app, host='localhost')
