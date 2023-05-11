import numpy as np


class Point:
    def __init__(self, x, y, z, index):
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.confilt_facets = []

    def __str__(self):
        return f"point({self.x},{self.y},{self.z})"
    def __repr__(self):
        return f"point({self.x},{self.y},{self.z})"

    




def orient(p1, p2, p3, p4):
    mat = np.ones((4, 4))

    mat[1][0] = p1.x
    mat[2][0] = p1.y
    mat[3][0] = p1.z

    mat[1][1] = p2.x
    mat[2][1] = p2.y
    mat[3][1] = p2.z

    mat[1][2] = p3.x
    mat[2][2] = p3.y
    mat[3][2] = p3.z

    mat[1][3] = p4.x
    mat[2][3] = p4.y
    mat[3][3] = p4.z


    det = np.linalg.det(mat)
    if det > 0:
        return 1
    elif det < 0:
        return -1
    else:
        return 0


class Facet:
    def __init__(self, p1, p2, p3):
        self.conflict_points = []
        self.neighbours = {"p1p2": None, "p1p3": None, "p2p3": None}
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def __str__(self):
        return f"facet defined on points: {self.p1}, {self.p2}, {self.p3}"
    def __repr__(self):
        return f"facet defined on points: {self.p1}, {self.p2}, {self.p3}"

    def get_edges(self):
        return [(self.p1, self.p2), (self.p1, self.p3), (self.p2, self.p3)]

    def get_common_edge(self, other):
        edges = self.get_edges()
        other_edges = other.get_edges()
        for edge in edges:
            if edge in other_edges or edge[::-1] in other_edges:
                return edge
        return None

    def set_neighbour(self, facet, edge):
        if edge[0] == self.p1:
            if edge[1] == self.p2:
                self.neighbours["p1p2"] = facet
            elif edge[1] == self.p3:
                self.neighbours['p1p3'] = facet
        elif edge[0] == self.p2:
            if edge[1] == self.p1:
                self.neighbours["p1p2"] = facet
            elif edge[1] == self.p3:
                self.neighbours['p2p3'] = facet
        elif edge[0] == self.p3:
            if edge[1] == self.p1:
                self.neighbours["p1p3"] = facet
            elif edge[1] == self.p2:
                self.neighbours['p2p3'] = facet


def neighbours(facets):
    for facet_1 in facets:
        for facet_2 in facets:
            if facet_1 != facet_2:
                common = facet_1.get_common_edge(facet_2)
                if common is not None:
                    facet_1.set_neighbour(facet_2, common)
                    facet_2.set_neighbour(facet_1, common)







class DCEL:
    def __init__(self, points,facets):
        self.points = points
        self.facets = facets


    def __str__(self):
        return f"decl({self.facets})"

    def __repr__(self):
        return f'dcel({self.facets})'

    def findHorizens(self,p):
        horizons = []
        if len(p.confilt_facets) == 1:
            print(1)
            f = p.confilt_facets[0]
            oldLen=len(self.facets)
            # add the 3 new facets
            self.facets.append(Facet(f.p1, f.p2, p))
            self.facets.append(Facet(f.p2, f.p3, p))
            self.facets.append(Facet(f.p3, f.p1, p))
            newLen=len(self.facets)
            #remove the old facet
            self.facets.remove(f)
            #update the points that were in conflict with the deleted facet
            for i in f.conflict_points:
                i.confilt_facets.remove(f)
                for j in range (oldLen-1,newLen-1):
                    if orient(self.facets[j].p1,self.facets[j].p2,self.facets[j].p3,i) == -1:
                        i.confilt_facets.append(self.facets[j])
                        self.facets[j].conflict_points.append(i)
            print(self)



        if len(p.confilt_facets) == 2:
            print(2)

        if len(p.confilt_facets) == 3:
            print(3)

        if len(p.confilt_facets) > 3:
            print("more")


def convex_hull(points):
    facets = startpyramid(points[0], points[1], points[2], points[3])
    dcel = DCEL(points,facets)
    neighbours(facets)
    conflict(facets, points[3:])
    print(dcel)
    for i in range(4,len(points)):
        if len(points[i].confilt_facets) == 0:
            print("do nothing")
            continue
        else:
            dcel.findHorizens(points[i])




def readFile(filename):
    file1 = open(filename, 'r')
    Lines = file1.readlines()

    pointsarray = []

    count = -1
    for line in Lines:
        line = line.strip()
        l = line.split(" ")
        res = [eval(i) for i in l]
        if count >= 0:
            p = Point(res[0], res[1], res[2], count)
            pointsarray.append(p)
            count += 1
        else:
            count += 1
    return pointsarray

def startpyramid(p1,p2,p3,p4):
    mid=Point((p1.x+p2.x+p3.x+p4.x)/4,(p1.y+p2.y+p3.y+p4.y)/4,(p1.z+p2.z+p3.z+p4.z)/4,-1)
    facets=[]
    if orient(p1,p2,p3,mid)==1:
        facets.append( Facet(p1,p2,p3))
    else:
        facets.append(Facet(p1,p3,p2))
    if orient(p1,p2,p4,mid)==1:
        facets.append(Facet(p1, p2, p4))
    else:
        facets.append(Facet(p1, p4, p2))
    if orient(p1,p3,p4,mid)==1:
        facets.append(Facet(p1, p3, p4))
    else:
        facets.append(Facet(p1, p4, p3))
    if orient(p2, p3,p4, mid) == 1:
        facets.append(Facet(p2, p3, p4))
    else:
        facets.append(Facet(p2, p4, p3))
    return facets

def conflict(facets, points):
    for i in points:
        for j in facets:
            if orient (j.p1,j.p2,j.p3,i) == -1:
                i.confilt_facets.append(j)
                j.conflict_points.append(i)

def main():
    plist = readFile("sampleinput1.txt")
    # np.random.shuffle(plist)
    # facets=startpyramid(plist[0],plist[1],plist[2],plist[3])
    # neighbours(facets)
    # for facet in facets:
    #     print(f"current facet: {facet}")
    #     print(f"neighbours: {facet.neighbours}")
    #     # print(f"p2-p3 neighbour: {facet.p2p3_neighbour}")
    #     # print(f"p1-p3 neighbour: {facet.p1p3_neighbour}")
    #     print()
    # 
    # conflict(facets, plist[4:-1])
    # for i in plist:
    #     print(len(i.confilt_facets))
    # print("****")
    # for i in facets:
    #     print(len(i.conflict_points))
    convex_hull(plist)





if __name__ == '__main__':
    main()