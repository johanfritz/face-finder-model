from face_classification_model import *
# import os
# from PIL import Image
# root='/home/johanfritz/Dokument/Python/github/face-finder-model/train/FAKE/'
# list=[]
# for subdir, dirs, files in os.walk(root):
#     for file in files:
#         path=root +file
#         image=Image.open(path).convert('L')
#         image=np.asarray(image)
#         list.append(image)
# root='/home/johanfritz/Dokument/Python/github/face-finder-model/train/REAL/'
# for subdir, dirs, files in os.walk(root):
#     for file in files:
#         path=root +file
#         image=Image.open(path).convert('L')
#         image=np.asarray(image)
#         list.append(image)
# list=np.array(list)
# np.save('not_faces.npy', list)

# network=Face_classification_model()
# notfaces=np.load('training_notfaces.npy')
# list=[]
# for k in range(np.shape(notfaces)[0]):
#     image=notfaces[k]
#     image=network.resize(image)
#     list.append(image)
# list=np.array(list)
# np.save('not_faces3.npy', list)

lr=0
iterations=2000
batch=50
ratio=2
save=True
plot=True


network=Face_classification_model()
loss=[]
tries=0
correct=0
tface=0
cface=0
tnot=0
cnot=0
faces=np.load('LFW1.npy')
notfaces=np.load('not_faces.npy')
for q in range(iterations):
    if q%ratio==0:
        rand=random.randint(0, np.shape(faces)[0]-1)
        image=faces[rand]
        lossfn, guess, delta1=network.descent(image, True, lr)
        loss.append(lossfn)
        tries+=1
        tface+=1
        print(delta1)
        if guess==0:
            correct+=1
            cface+=1
    if not q%ratio==0:
        rand=random.randint(0, np.shape(notfaces)[0]-1)
        image=notfaces[rand]
        lossfn, guess, delta1=network.descent(image, False, lr)
        loss.append(lossfn)
        tries+=1
        tnot+=1
        print(delta1)
        if guess==1:
            correct+=1
            cnot+=1
    if q%batch==0 and q>2:
        network.update()
    if q%100==0:
        print(100*q/iterations)
print(correct/tries)
print(cface/tface)
print(cnot/tnot)
if save:
    network.save()
loss=np.array(loss)
size=len(loss)
x=np.linspace(0, size, size)
if plot:
    plt.plot(x, loss, '.')
    plt.show()

