from face_classification_torch import *
from PIL import Image
from face_classification_model import *

def extrafunction(n):
    if n<0:
        return 0
    else:
        return n
specialrelu=np.vectorize(extrafunction)

def slidingboxes(size):
    list=[]
    xmax=size[1]
    ymax=size[0]
    scale=1
    x=150*scale
    y=175*scale
    xnum=3
    ynum=3
    if x>xmax or y>ymax:
        return np.array(size)
    while xnum>2 and ynum>2:
        x=150*scale
        y=175*scale
        xnum=xmax//x
        ynum=ymax//y
        x=int(xmax/xnum)
        y=int(ymax/ynum)
        for k in range(xnum):
            for m in range(ynum):
                array=np.array([k*x, m*y, (k+1)*x, (m+1)*y])
                list.append(array)
        scale+=1
    list=np.array(list)
    list-=1
    list=specialrelu(list)
    return list

def split(array, box):
    xmin, ymin, xmax, ymax=box[0], box[1], box[2], box[3]
    out=array[ymin:ymax, xmin:xmax]
    return out

def markImage(array, bndbox, n=0):
    objects=np.shape(bndbox)[0]
    for k in range(objects):
        xmin, ymin, xmax, ymax= bndbox[k, 0], bndbox[k, 1], bndbox[k, 2], bndbox[k, 3]
        array[ymin, xmin:xmax]=n
        array[ymax, xmin:xmax]=n
        array[ymin:ymax, xmin]=n
        array[ymin:ymax, xmax]=n
    return array

model=NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
oldmodel=Face_classification_model()
image=Image.open('IMG_9202.jpg').convert('L')
image=np.array(image)
boxes=slidingboxes(np.shape(image))
correctboxes=[]
for q in range(np.shape(boxes)[0]):
    potentialface=split(image, boxes[q])
    potentialface=oldmodel.resize(potentialface)
    potentialface=torch.from_numpy(potentialface).type(torch.float32)
    guess=model.knas(potentialface)
    if torch.argmax(guess)==1:
        correctboxes.append(boxes[q])
correctboxes=np.array(correctboxes)
print(correctboxes)
image=markImage(image, correctboxes)
plt.imshow(image)
plt.show()