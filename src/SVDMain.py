imgPath = 'hendrix_final.png'
finalImg = 'ReconstructedFinal.png'
combinedImage = 'FinalCombinedImage.png'
import numpy as np
import matplotlib.image as image
import  numpy.linalg as la
import matplotlib.pyplot as plt
import PIL.Image as im

def loadImage(channel):
    img = image.imread(imgPath)
    imgArray = (np.array(img))[:,:,channel]
    return imgArray

def getSVDOfImage(img, svdCount):
    u,s,vh = la.svd(img)
    nonz = []
    if svdCount == 1:
        nonZeroR = []
        for i in s:
            if i!=0:
                nonZeroR.append(i)
                nonz.append(np.log(i))
        plt.loglog(nonZeroR)
        plt.show()
        plt.plot(nonz)
        plt.show()
    return u,s,vh

def calculateFroNorms(u,s,vh,img):
    rank = la.matrix_rank(img)
    errorFrob = []
    rankM =[]
    if rank:
        for i in range(1,rank):
            rankM.append(i+1)
            copyOfSingleMatrix = np.copy(s)
            copyOfSingleMatrix[i:]=0
            diagSingle = np.diag(copyOfSingleMatrix)
            reconstructedMatrix = np.dot(u, np.dot(diagSingle, vh))
            l = np.array(reconstructedMatrix)-np.array(img)
            frbNorm = la.norm(l,ord='fro')
            errorFrob.append(frbNorm)
    return errorFrob,rankM

def plotFrobeniusNormsOfErrorMatrix(u,s,vh,img,color):
    errorR,rankR = calculateFroNorms(u,s,vh,img)
    plt.plot(rankR,errorR,color=color)

def reconstructImage(singleValueCount, u,s,vh,channel):
    s[singleValueCount:]= 0
    diagSingleValue = np.diag(s)
    reconstrutedImage = np.dot(u,np.dot(diagSingleValue,vh))
    imageName = channel+'.png'
    image.imsave(imageName,np.array(reconstrutedImage),cmap=channel)
    return imageName,reconstrutedImage

def reconstructColorImage(red,blue,green):
    nw = []
    for row in range(len(red)):
        tempN=[]
        for col in range(len(red[0])):
            temp = []
            a = r1[row,col]
            temp.append(a)
            temp.append(green[row,col])
            temp.append(blue[row,col])
            tempN.append(temp)
        nw.append(tempN)
    nw = np.array(nw)
    image.imsave(finalImg,np.array(nw))
    i = im.open(finalImg).convert("RGB")
    i.save(finalImg)
    orImg = im.open(imgPath)
    fiImg = im.open(finalImg)
    tw = orImg.size[0]  + fiImg.size[0] + 50
    mh = max(orImg.size[1],fiImg.size[1])
    newIm = im.new('RGB', (tw,mh))
    newIm.paste(orImg,(0,0))
    newIm.paste(fiImg,(orImg.size[0]+50,0))
    newIm.save(combinedImage)

def resizeImage():
    try:
        imt = im.open(imgPath)
        imt.thumbnail(size, im.ANTIALIAS)
        imt.save("Small.png", "PNG")
        return "Small.png"

    except IOError:
        print ("cannot create thumbnail for")
        return imgPath

size = 512,512
imgPath = resizeImage()
imgR = loadImage(0)
uR,sR,vhR = getSVDOfImage(imgR,1)
plotFrobeniusNormsOfErrorMatrix(uR,sR,vhR,imgR, 'red')
imgB = loadImage(2)
uB,sB,vhB = getSVDOfImage(imgB,0)
plotFrobeniusNormsOfErrorMatrix(uB,sB,vhB,imgB,'blue')
imgG = loadImage(1)
uG,sG,vhG = getSVDOfImage(imgG,0)
plotFrobeniusNormsOfErrorMatrix(uG,sG,vhG,imgG,'green')
plt.show()
imageR,r1 = reconstructImage(250,uR,sR,vhR,"Reds")
imageB,b1 = reconstructImage(250,uB,sB,vhB,"Blues")
imageG,g1 = reconstructImage(250,uG,sG,vhG,"Greens")
reconstructColorImage(r1,b1,g1)



















