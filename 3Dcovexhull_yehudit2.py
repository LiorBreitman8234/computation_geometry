import numpy as np

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import numpy as np


def plotfig(facets,points):
    eps=3
    # fig = pl.figure(figsize=(10, 10))  # Increase the figure size (adjust as needed)
    # ax = fig.add_subplot(111, projection='3d')
    ax = a3.Axes3D(pl.figure())
    ax.set_xlim(0, 300)  # Adjust the range for the x-axis
    ax.set_ylim(0, 300)   # Adjust the range for the y-axis
    ax.set_zlim(0, 300)   # Adjust the range for the z-axis
    for i in facets:
        if i.outer_face == True:
            vtx = [[i.p1.x, i.p1.y, i.p1.z], [i.p2.x, i.p2.y, i.p2.z], [i.p3.x, i.p3.y, i.p3.z]]
            tri = a3.art3d.Poly3DCollection([vtx])
            tri.set_color(colors.rgb2hex(np.random.rand(3)))
            tri.set_edgecolor('k')
            ax.add_collection3d(tri)
    for i in points:
        ax.text(i.x+eps, i.y+eps, i.z-2*eps, i.index, fontsize=10, color='black')

    pl.show()


def orient1(p1, p2, p3, p4):
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


def orient2(facet, p4):
    mat = np.ones((4, 4))

    mat[1][0] = facet.p1.x
    mat[2][0] = facet.p1.y
    mat[3][0] = facet.p1.z

    mat[1][1] = facet.p2.x
    mat[2][1] = facet.p2.y
    mat[3][1] = facet.p2.z

    mat[1][2] = facet.p3.x
    mat[2][2] = facet.p3.y
    mat[3][2] = facet.p3.z

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


class Point:
    def __init__(self, x, y, z, index):
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.conflict_facets = set()

    def __str__(self):
        return f"point({self.x},{self.y},{self.z})"

    def __repr__(self):
        return f"point({self.x},{self.y},{self.z})"


class Facet:
    def __init__(self, p1, p2, p3, number):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.outer_face = True
        self.number = number
        self.conflict_points = set()
        self.facet_Neighbors = [None, None, None]  # first spot opisite p1, second opisite p2, third opisite p3
        # self.neighbours = {"p1p2": None, "p1p3": None, "p2p3": None}

    def __str__(self):
        if self.outer_face == True:
            return f"facet {self.number}, {self.p1}, {self.p2}, {self.p3}"
        else:
            return ""

    def __repr__(self):
        if self.outer_face == True:
            # return f"facet defined on points: {self.p1}, {self.p2}, {self.p3}"
            return f"facet {self.number}, {self.p1}, {self.p2}, {self.p3}"
        else:
            return ""


class HorizonEdge:
    def __init__(self, p1, p2, facet_white, facet_gray):
        self.p1 = p1
        self.p2 = p2
        self.facet_white = facet_white  # in conflict with point
        self.facet_gray = facet_gray  # not in conflict with point
        # self.v1_index_in_white


class DCEL:
    def __init__(self, points, facets):
        self.points = points
        self.facets = facets
        self.facet_number = 0
        self.midpoint = None
        # self.num_facets = len(facets)

    def __str__(self):
        print("num of outer faces is ", self.get_num_outer_faces())
        return f"decl({self.facets})"

    def __repr__(self):
        return f"decl({self.facets})"

    def get_num_outer_faces(self):
        n = 0
        for i in self.facets:
            if i.outer_face == True:
                n += 1
        return n

    def create_simplex(self, p1, p2, p3, p4):
        mid = Point((p1.x + p2.x + p3.x + p4.x) / 4, (p1.y + p2.y + p3.y + p4.y) / 4, (p1.z + p2.z + p3.z + p4.z) / 4,
                    -1)
        self.midpoint = mid

        if orient1(p1, p2, p3, mid) == 1:
            self.facets.append(Facet(p1, p2, p3, self.facet_number))
            self.facet_number += 1
        else:
            self.facets.append(Facet(p1, p3, p2, self.facet_number))
            self.facet_number += 1

        if orient1(p1, p2, p4, mid) == 1:
            self.facets.append(Facet(p1, p2, p4, self.facet_number))
            self.facet_number += 1
        else:
            self.facets.append(Facet(p1, p4, p2, self.facet_number))
            self.facet_number += 1

        if orient1(p1, p3, p4, mid) == 1:
            self.facets.append(Facet(p1, p3, p4, self.facet_number))
            self.facet_number += 1
        else:
            self.facets.append(Facet(p1, p4, p3, self.facet_number))
            self.facet_number += 1

        if orient1(p2, p3, p4, mid) == 1:
            self.facets.append(Facet(p2, p3, p4, self.facet_number))
            self.facet_number += 1
        else:
            self.facets.append(Facet(p2, p4, p3, self.facet_number))
            self.facet_number += 1

        self.facets[0].facet_Neighbors = [self.facets[3], self.facets[2], self.facets[1]]
        self.facets[1].facet_Neighbors = [self.facets[3], self.facets[2], self.facets[0]]
        self.facets[2].facet_Neighbors = [self.facets[3], self.facets[1], self.facets[0]]
        self.facets[3].facet_Neighbors = [self.facets[2], self.facets[1], self.facets[0]]

        for i in self.points[4:]:
            for j in self.facets:
                if orient1(j.p1, j.p2, j.p3, i) == -1:
                    i.conflict_facets.add(j)
                    j.conflict_points.add(i)

    def find_horizon_edges(self, p):
        horizen_edges = set()

        for i in p.conflict_facets:
            if i.outer_face == True:
                h = self.check_if_horizen(i, p)
                if h != None:
                    for j in h:
                        horizen_edges.add(j)
        return horizen_edges

    def check_if_horizen(self, facet, p):
        o1 = orient2(facet.facet_Neighbors[0], p)
        o2 = orient2(facet.facet_Neighbors[1], p)
        o3 = orient2(facet.facet_Neighbors[2], p)

        if o1 == 0 or o2 == 0 or o3 == 0:
            print("4 points in the same plane\n")
            return None

        # no edges should be in conflict with all of its neighbors
        if o1 == -1 and o2 == -1 and o3 == -1:
            l = []
            for i in facet.facet_Neighbors:
                if i not in p.conflict_facets:
                    l.append(self.check_if_horizen(i, p))
        # 1 edge
        elif o1 == 1 and o2 == -1 and o3 == -1:
            m = self.find_shared_points(facet,facet.facet_Neighbors[0])
            # return [HorizonEdge(facet.p2, facet.p3, facet, facet.facet_Neighbors[0])]
            return [HorizonEdge(m[0], m[1], facet, facet.facet_Neighbors[0])]

        elif o1 == -1 and o2 == 1 and o3 == -1:
            m = self.find_shared_points(facet, facet.facet_Neighbors[1])
            # return[HorizonEdge(facet.p1, facet.p3, facet, facet.facet_Neighbors[1])]
            return [HorizonEdge(m[0], m[1], facet, facet.facet_Neighbors[1])]

        elif o1 == -1 and o2 == -1 and o3 == 1:
            m = self.find_shared_points(facet, facet.facet_Neighbors[2])
            # return [HorizonEdge(facet.p1, facet.p2, facet, facet.facet_Neighbors[2])]
            return [HorizonEdge(m[0], m[1], facet, facet.facet_Neighbors[2])]

        # 2 edge
        elif o1 == 1 and o2 == 1 and o3 == -1:
            m1 = self.find_shared_points(facet, facet.facet_Neighbors[0])
            m2 = self.find_shared_points(facet, facet.facet_Neighbors[1])
            # return [HorizonEdge(facet.p2, facet.p3, facet, facet.facet_Neighbors[0]),
            #         HorizonEdge(facet.p1, facet.p3, facet, facet.facet_Neighbors[1])]
            return [HorizonEdge(m1[0], m1[1], facet, facet.facet_Neighbors[0]),
                    HorizonEdge(m2[0], m2[1], facet, facet.facet_Neighbors[1])]

        elif o1 == -1 and o2 == 1 and o3 == 1:
            m1 = self.find_shared_points(facet, facet.facet_Neighbors[1])
            m2 = self.find_shared_points(facet, facet.facet_Neighbors[2])
            # return [HorizonEdge(facet.p1, facet.p3, facet, facet.facet_Neighbors[1]),
            #         HorizonEdge(facet.p1, facet.p2, facet, facet.facet_Neighbors[2])]
            return [HorizonEdge(m1[0], m1[1], facet, facet.facet_Neighbors[1]),
                    HorizonEdge(m2[0], m2[1], facet, facet.facet_Neighbors[2])]

        elif o1 == 1 and o2 == -1 and o3 == 1:
            m1 = self.find_shared_points(facet, facet.facet_Neighbors[0])
            m2 = self.find_shared_points(facet, facet.facet_Neighbors[2])
            # return [HorizonEdge(facet.p2, facet.p3, facet, facet.facet_Neighbors[0]),
            #         HorizonEdge(facet.p1, facet.p2, facet, facet.facet_Neighbors[2])]
            return [HorizonEdge(m1[0], m1[1], facet, facet.facet_Neighbors[0]),
                    HorizonEdge(m2[0], m2[1], facet, facet.facet_Neighbors[2])]

        # 3 edges
        elif o1 == 1 and o2 == 1 and o3 == 1:
            m1 = self.find_shared_points(facet, facet.facet_Neighbors[0])
            m2 = self.find_shared_points(facet, facet.facet_Neighbors[1])
            m3 = self.find_shared_points(facet, facet.facet_Neighbors[2])
            # return [HorizonEdge(facet.p2, facet.p3, facet, facet.facet_Neighbors[0]),
            #         HorizonEdge(facet.p1, facet.p3, facet, facet.facet_Neighbors[1]),
            #         HorizonEdge(facet.p1, facet.p2, facet, facet.facet_Neighbors[2])]
            return [HorizonEdge(m1[0], m1[1], facet, facet.facet_Neighbors[0]),
                    HorizonEdge(m2[0], m2[1], facet, facet.facet_Neighbors[1]),
                    HorizonEdge(m3[0], m3[1], facet, facet.facet_Neighbors[2])]

    def create_new_facets_and_update_conflict_points(self, horizon_edges, p):
        new_facets = set()
        for i in horizon_edges:
            # create new facet
            if orient1(i.p1, i.p2, p, self.midpoint) == 1:
                self.facets.append(Facet(i.p1, i.p2, p, self.facet_number))
                self.facet_number += 1
            else:
                self.facets.append(Facet(i.p1, p, i.p2, self.facet_number))
                self.facet_number += 1

            # update gray facets neighbor that changed
            if i.facet_gray.facet_Neighbors[0] == i.facet_white:
                i.facet_gray.facet_Neighbors[0] = self.facets[-1]
            elif i.facet_gray.facet_Neighbors[1] == i.facet_white:
                i.facet_gray.facet_Neighbors[1] = self.facets[-1]
            elif i.facet_gray.facet_Neighbors[2] == i.facet_white:
                i.facet_gray.facet_Neighbors[2] = self.facets[-1]

            # new facet insert the facet neighbor that is the gray facet
            if self.facets[-1].p1 != i.p1 and self.facets[-1].p1 != i.p2:
                self.facets[-1].facet_Neighbors[0] = i.facet_gray

            elif self.facets[-1].p2 != i.p1 and self.facets[-1].p2 != i.p2:
                self.facets[-1].facet_Neighbors[1] = i.facet_gray

            elif self.facets[-1].p3 != i.p1 and self.facets[-1].p3 != i.p2:
                self.facets[-1].facet_Neighbors[2] = i.facet_gray

            new_facets.add(self.facets[-1])

            # update conflict points
            for j in i.facet_white.conflict_points:
                if j != p:
                    try:
                        j.conflict_facets.remove(i.facet_white)
                    except:
                        pass
                    if orient2(self.facets[-1], j) == -1:
                        j.conflict_facets.add(self.facets[-1])
                        self.facets[-1].conflict_points.add(j)
            # i.facet_white.outer_face = False
            for j in i.facet_gray.conflict_points:
                if j != p:
                    try:
                        j.conflict_facets.remove(i.facet_white)
                    except:
                        pass
                    if orient2(self.facets[-1], j) == -1:
                        j.conflict_facets.add(self.facets[-1])
                        self.facets[-1].conflict_points.add(j)

        # update new facets neighbors that arent grey
        for i in new_facets:
            for j in new_facets:
                if i != j:
                    self.share_edge(i, j)

    def share_edge(self, face1, face2):
        matches = []

        if face1.p1 == face2.p1:
            matches.append((1, 1))
        elif face1.p1 == face2.p2:
            matches.append((1, 2))
        elif face1.p1 == face2.p3:
            matches.append((1, 3))

        if face1.p2 == face2.p1:
            matches.append((2, 1))
        elif face1.p2 == face2.p2:
            matches.append((2, 2))
        elif face1.p2 == face2.p3:
            matches.append((2, 3))

        if face1.p3 == face2.p1:
            matches.append((3, 1))
        elif face1.p3 == face2.p2:
            matches.append((3, 2))
        elif face1.p3 == face2.p3:
            matches.append((3, 3))

        if len(matches) == 2:
            if matches[0][0] != 1 and matches[1][0] != 1:
                face1.facet_Neighbors[0] = face2
            elif matches[0][0] != 2 and matches[1][0] != 2:
                face1.facet_Neighbors[1] = face2
            elif matches[0][0] != 3 and matches[1][0] != 3:
                face1.facet_Neighbors[2] = face2

            if matches[0][1] != 1 and matches[1][1] != 1:
                face2.facet_Neighbors[0] = face1
            elif matches[0][1] != 2 and matches[1][1] != 2:
                face2.facet_Neighbors[1] = face1
            elif matches[0][1] != 3 and matches[1][1] != 3:
                face2.facet_Neighbors[2] = face1


    def find_shared_points(self, face1, face2):
        matches = []

        if face1.p1 == face2.p1:
            # matches.append((1, 1))
            matches.append(face1.p1)
        elif face1.p1 == face2.p2:
            matches.append(face1.p1)
            # matches.append((1, 2))
        elif face1.p1 == face2.p3:
            matches.append(face1.p1)
            # matches.append((1, 3))

        if face1.p2 == face2.p1:
            # matches.append((2, 1)
            matches.append(face1.p2)
        elif face1.p2 == face2.p2:
            # matches.append((2, 2))
            matches.append(face1.p2)
        elif face1.p2 == face2.p3:
            # matches.append((2, 3))
            matches.append(face1.p2)

        if face1.p3 == face2.p1:
            # matches.append((3, 1))
            matches.append(face1.p3)
        elif face1.p3 == face2.p2:
            # matches.append((3, 2))
            matches.append(face1.p3)
        elif face1.p3 == face2.p3:
            # matches.append((3, 3))
            matches.append(face1.p3)

        return matches



    def delete_faces(self, p):
        for i in p.conflict_facets:
            i.outer_face = False

    def find_neighbors(self, facet):
        for i in self.facets:
            if i != facet and i.outer_face == True:
                self.share_edge(facet, i)

    def check_neighbors(self):
        for i in self.facets:
            if i.outer_face == True:
                self.find_neighbors(i)

    def final_answer(self):
        ans = []
        for i in self.facets:
            if i.outer_face == True:
                if i.p1.index < i.p2.index and i.p1.index < i.p3.index:
                    ans.append((i.p1.index, i.p2.index, i.p3.index))
                elif i.p2.index < i.p1.index and i.p2.index < i.p3.index:
                    ans.append((i.p2.index, i.p3.index, i.p1.index))
                elif i.p3.index < i.p1.index and i.p3.index < i.p2.index:
                    ans.append((i.p3.index, i.p1.index, i.p2.index))
        # print("********")
        # for i in ans:
        #     print(i)
        # print("********")
        sorted_lst = sorted(ans, key=lambda x: (x[0],x[1]))
        # sorted_lst2 = sorted(sorted_lst1, key=lambda x:x[1])

        return sorted_lst




def convex_hull(points):
    dcel = DCEL(points, [])

    # create the pyramid and add the original conflicts and the facet_neighbors
    dcel.create_simplex(points[0], points[1], points[2], points[3])
    # print(dcel)
    # plotfig(dcel.facets,dcel.points)
    iter = 0
    for i in range(4, len(points)):
        iter += 1
        if len(points[i].conflict_facets) == 0:
            # print("do nothing")
            continue
        else:
            # print("i have conflicts")
            h = dcel.find_horizon_edges(points[i])
            dcel.create_new_facets_and_update_conflict_points(h, points[i])
            dcel.delete_faces(points[i])
            # dcel.check_neighbors()
        # plotfig(dcel.facets,dcel.points)


    # print(dcel)
    # print()
    # edges = set()
    # for i in dcel.facets:
    #     if i.outer_face == True:
    #         edges.add((i.p1, i.p2))
    #         edges.add((i.p1, i.p3))
    #         edges.add((i.p3, i.p2))
    # for i in edges:
    #     print(i)
    # plotfig(dcel.facets,points)
    ans = dcel.final_answer()
    for i in ans:
        print(i[0], i[1], i[2])


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


def main():
    plist = readFile("sampleinput.txt")
    np.random.shuffle(plist)
    # print("printing list of points")
    # print("********")
    # for i in plist:
    #     print(i)
    # print("********")
    convex_hull(plist)


if __name__ == '__main__':
    main()
