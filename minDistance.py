import re
import math


THRESHOLD = 1.07

#x y z position of atom
class atom():
    def __init__(self,x,y,z):
        self.x =x
        self.y = y
        self.z =z

#measure distance between BB atom
def distance( atom1,atom2):
    return math.sqrt(math.pow(atom1.x - atom2.x,2)+math.pow(atom1.y - atom2.y,2)+math.pow(atom1.z - atom2.z,2))

def loadData():
    path = 'E:\\TiboD_MD\\pr2and3\\300700.gro'
    count=0
    list1=[]
    list2=[]
    with open(path,'r') as f:
        for line in f:
            count+=1
            arr = re.split("\s+",line)
            if len(arr) > 3:
                if count < 3664:
                    if arr[2]=="BB":
                        A = atom(float(arr[4]),float(arr[5]),float(arr[6]))
                        list1.append(A)
                else:
                    if arr[2]=="BB":
                        B = atom(float(arr[4]),float(arr[5]),float(arr[6]))
                        list2.append(B)

    return list1,list2

#get distance value between two molecules
#threshole = 0.46
def getDvalue():
    list1,list2 = loadData()
    print(list1)
    print(list2)
    distance1 =[]
    for i in range(0,len(list1)):
        A = list1[i]
        for j in range(0,len(list2)):
            B=list2[j]
            distance1.append(distance(A,B))
    print("ddd : ", len(distance1))
    return min(distance1)

#get frequence molucules atom in theshole
def getFrequenc():
    list1,list2 = loadData()
    print(len(list1))
    count = 0
    for i in range(0, len(list1)):
        for j in range(0, len(list2)):
            A = list1[i]
            B = list2[j]
            if distance(A,B) < THRESHOLD:
                print("i ",A.x, " ", A.y, " ",A.z)
                print("j ",B.x, " ", B.y, " ",B.z)
                count+=1
    return count

if __name__ == '__main__':
    # print(getDvalue())
    print(getFrequenc())