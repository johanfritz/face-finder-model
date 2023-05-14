from face_classification_torch import *
from PIL import Image
from face_classification_model import *

def potentialboxes(size, numboxes):
    xlim=15
    ylim=15
    list=[]
    while len(list)<numboxes:
        ymax=random.randint(0, size[0])
        xmax=random.randint(0, size[1])
        ymin=random.randint(0, size[0])
        xmin=random.randint(0, size[1])
        if (xmin+xlim<xmax) and (ymin+ylim<ymax):
            if not (xmax-xmin)>2*(ymax-ymin):
                if not (ymax-ymin)>2*(xmax-xmin):
                    kingen=np.array((xmin, ymin, xmax, ymax))
                    kingen-=1
                    list.append(kingen)
    return np.array(list)

def split(array, xmin, ymin, xmax, ymax):
    out=array[ymin:ymax, xmin:xmax]
    return out

def markImage(image, bndbox, n=0):
    objects=np.shape(bndbox)[0]
    for k in range(objects):
        xmin, ymin, xmax, ymax= bndbox[k, 0], bndbox[k, 1], bndbox[k, 2], bndbox[k, 3]
        image[ymin, xmin:xmax]=n
        image[ymax, xmin:xmax]=n
        image[ymin:ymax, xmin]=n
        image[ymin:ymax, xmax]=n
    return image

model=NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
oldmodel=Face_classification_model()
list=[]
# image=Image.open('Dostoevsky_1879.jpg').convert('L')
# image=np.asarray(image)
# randboxes=potentialboxes(np.shape(image), 100)
# print(randboxes)
# print(np.shape(randboxes))
# for k in range(np.shape(randboxes)[0]):
#     potentialface=split(image, randboxes[k, 0], randboxes[k, 1], randboxes[k, 2], randboxes[k, 3])
#     potentialface=oldmodel.resize(potentialface)
#     potentialface=torch.from_numpy(potentialface).type(torch.float32)
#     guess=model.knas(potentialface)
#     if guess[0]>guess[1]+0.95:
#         list.append(randboxes[k])
# list=np.array(list)
# print(list)
# print(np.shape(list))
# newimage=Image.open('Dostoevsky_1879.jpg').convert('L')
# newimage=np.asarray(newimage)
# newimage=markImage(newimage, list)
# plt.imshow(newimage)
# plt.show()
numboxes=50
originalimage=Image.open('Dostoevsky_1879.jpg').convert('L')
originalimage=np.asarray(originalimage)
randboxes=potentialboxes(np.shape(originalimage), numboxes)
for k in range(numboxes):
    potentialface=split(originalimage, randboxes[k, 0], randboxes[k, 1], randboxes[k, 2], randboxes[k, 3])
    potentialface=oldmodel.resize(potentialface)
    potentialface=torch.from_numpy(potentialface).type(torch.float32)
    guess=model.knas(potentialface)
    if torch.argmax(guess)==1:
        label='guess: FACE!!!!!'
        list.append(randboxes[k, :])
    if torch.argmax(guess)==0:
        label='guess: not face'
    #plt.imshow(potentialface)
    #plt.title(label)
    #plt.show()
print(list)
list=np.array(list)
print(list)
newimage=Image.open('Dostoevsky_1879.jpg').convert('L')
imagearray=np.asarray(newimage)
marked=markImage(imagearray, list)
plt.imshow(marked)
plt.show()