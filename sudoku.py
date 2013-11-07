'''
Created on Mar 5, 2013

@author: chris
'''

from Tkinter import Tk, Canvas, Button, BOTH, Frame
from random import random, randint, seed
from math import sin, cos
import string
import copy
from time import time
from compiler.ast import Pass

def release(event, canvas, board):
    # canvas.create_polygon(board.cellCoordinates(*board.rowCol(event.x, event.y)))
    board.removePopup()
    board.createPopup(event.x, event.y)

def entered(row, col, event):
    pass

def newMatrix(rows, cols, seed):
    m = []
    for r in xrange(rows):
        row = []
        for c in xrange(cols):
            row.append(copy.deepcopy(seed))
        m.append(row)
    return m

def color(m, n, row, col):
    rowSection = row / m
    colSection = col / n
    red = int(150 - 80 * sin(colSection + 0.5 * rowSection))
    green = int(150 + 100 * cos(rowSection + colSection))
    blue = int(150 + 100 * sin(rowSection))
    return '#%02x%02x%02x' % (red, green, blue)

def isSubList(sub, super):
    for element in sub:
        if not element in super:
            return False
    return True

class blockIterator(object):
    def __init__(self, board, rowm, coln, m, n):
        self.rowm = rowm
        self.coln = coln
        self.m = m
        self.n = n
        self.current = [0, 0]
        self.board = board
    def __iter__(self):
        return self
    def next(self):
        if self.current[0] == self.m:
            raise StopIteration
        else:
            row = self.current[0] + self.rowm * self.m
            col = self.current[1] + self.coln * self.n
            try:
                value = self.board[row][col]
            except:
                print row, col, self.rowm, self.coln
                raise
            self.current[1] += 1
            if self.current[1] == self.n:
                self.current[1] = 0
                self.current[0] += 1
            return ((row, col), value)

class SudokuBoard(object):
    def __init__(self, m, n, canvas):
        """Creates a blank board. Use set() to assign numbers."""
        self.values = list((string.digits + string.ascii_lowercase)[1:m * n + 1])
        self.board = newMatrix(m * n, m * n, [0, self.values[:]])
        self.m = m
        self.n = n
        self.size = self.m * self.n
        self.canvas = canvas
        self.pad = 50
        self.draw()

    def __getitem__(self, index):
        return blockIterator(self.board, index[0], index[1], self.m, self.n)

    def createPopup(self, x, y):
        (row, col) = self.rowCol(x, y)
        self.popup = []
        self.popup.append(self.canvas.create_polygon(self.popupCoordinates(row, col)))

    def popupCoordinates(self, row, col):
        corners = [(+0.4, 0), (0.9, 0.5), (0.9, 2), (2.4, 2), (2.4, -2), (0.9, -2), (0.9, -0.5)]
        coords = ()
        for corner in corners:
            coords += (self.coordinate(row + corner[0], col + corner[1]))
        return coords

    def removePopup(self):
        if hasattr(self, 'popup'):
            for obj in self.popup:
                self.canvas.itemconfig(obj, state='hidden')
            del self.popup

    def rowContains(self, value, row):
        for col in xrange(self.size):
            checkAgainst = self.board[row][col][0]
            if checkAgainst == value:
                return True
        return False

    def colContains(self, value, col):
        for row in xrange(self.size):
            checkAgainst = self.board[row][col][0]
            if checkAgainst == value:
                return True
        return False

    def blockContains(self, value, row, col):
        for ((rowCheck, colCheck), checkAgainst) in self[self.getBlock(row, col)]:
            if (rowCheck, colCheck) != (row, col) and value == checkAgainst[0]:
                return True
        return False

    def isNeighbor(self, cell1, cell2):
        (row1, col1) = cell1
        (row2, col2) = cell2
        if row1 == row2 or col1 == col2: return True
        (m, n) = (self.m, self.n)
        if row1 / m == row2 / m and col1 / n == col2 / n: return True
        return False

    def refine(self, row, col):
        toRemove = []
        L = self.board[row][col][1]  # Alias
        for val in L:
            if (self.rowContains(val, row) or
                self.colContains(val, col) or
                self.blockContains(val, row, col)):
                toRemove.append(val)
        for val in toRemove:
            L.remove(val)

    def numHidden(self):
        count = 0
        for row in xrange(self.size):
            for col in xrange(self.size):
                if self.board[row][col][0] == 0:
                    count += 1
        return count

    def assignValues(self, row, col, recur):
        if recur:
            prevPercent = 100 * ((row * self.size) + col - 1) / self.size ** 2
            percent = 100 * ((row * self.size) + col) / self.size ** 2
            if percent / 5 != prevPercent / 5:
                print '%d%%' % (percent),
        if self.board[row][col][0] == 0:
            self.refine(row, col)
            # print 'refining'
            while self.board[row][col][1] == []:
                prevCol = col - 1
                prevRow = row
                if prevCol == -1:
                    prevCol = self.size - 1
                    prevRow -= 1
                if prevCol >= 0:
                    if self.board[prevRow][prevCol][0] in self.board[prevRow][prevCol][1]:
                        # Don't let the previous cell try that value again
                        self.board[prevRow][prevCol][1].remove(self.board[prevRow][prevCol][0])
                    self.board[prevRow][prevCol][0] = 0
                    self.assignValues(prevRow, prevCol, False)
                    self.board[row][col][1] = self.values[:]
                    self.refine(row, col)
                else:
                    print 'First cell has no options!'
                    break
            i = randint(0, len(self.board[row][col][1]) - 1)
            self.board[row][col][0] = self.board[row][col][1][i]
            nextCol = col + 1
            nextRow = row
            if nextCol == self.size:
                nextCol = 0
                nextRow += 1
            if nextRow != self.size:
                if recur:
                    self.assignValues(nextRow, nextCol, True)

    def isLegalBoard(self):
        for row in xrange(self.size):
            if not sorted([i[0] for i in self.board[row]]) == sorted(self.values):
                print row, sorted([i[0] for i in self.board[row]]), sorted(self.values)
                return False
        for col in xrange(self.size):
            if not sorted(self.board[row][col][0] for col in xrange(self.size)) == sorted(self.values):
                print col, sorted(self.board[row][col][0] for col in xrange(self.size)), sorted(self.values)
                return False
        for mIdx in xrange(self.m):
            for nIdx in xrange(self.n):
                values = []
                for col in xrange(mIdx * self.n, (mIdx + 1) * self.n):
                    for row in xrange(nIdx * self.m, (nIdx + 1) * self.m):
                        values.append(self.board[row][col][0])
                if sorted(values) != sorted(self.values):
                    print sorted(values), sorted(self.values)
                    return False
        return True

    def set(self, seedVal=0):
        timer = time()
        if hasattr(self, 'texts'):
            for t in self.texts:
                self.canvas.itemconfig(t, state='hidden')
                del t
        self.values = list((string.digits + string.ascii_lowercase)[1:self.size + 1])
        self.board = newMatrix(self.size, self.size, [0, self.values[:]])
        self.texts = []
        seed(seedVal)
        difficulty = random()
        print 'Generating board...',
        self.assignValues(0, 0, True)
        elapsed = time() - timer
        output = '%0.3f sec' % (elapsed) if elapsed > 1 else '%0.3f msec' % (elapsed * 1000)
        print '100%'
        print 'Time: %s' % (output)
        print 'Checking board\'s validity...',
        if not self.isLegalBoard():
            raise ValueError('Illegal puzzle!')
            print 'testing failed'
        else:
            print 'success!'
            # return
        print 'Reversing solution...',
        self.hideValues()
        for row in xrange(self.size):
            for col in xrange(self.size):
                value = self.board[row][col][0]
                if value != 0:
                    self.texts.append(self.canvas.create_text(self.coordinate(row, col),
                                                              text=str(value),
                                                              font='Arial 24 bold'))
                marks = self.board[row][col][1]
                if marks != []:
                    self.texts.append(self.canvas.create_text(self.coordinate(row, col),
                                                              text=str(marks),
                                                              font='Arial 2 bold'))

    def resetPencilMarks(self):
        for row in xrange(self.size):
            for col in xrange(self.size):
                value = self.board[row][col][0]
                pencilMarks = []
                self.board[row][col] = [value, pencilMarks]

    def search(self, goodSiblings):
        if (self.size ** 2 - self.size) / 2 < self.numHidden():
            return
        print '%02d%%' % ((self.numHidden() * 100) / (((self.size ** 2) - self.size) / 2)),
        children = []
        goodChildren = []
        for method in self.inverses:
            children.extend(self.childrenVia(method))  # Candidates
        for method in self.forwards:
            for child in children:
                if isSubList(self.parentsVia(child, method), goodSiblings):
                    goodChildren.append(child)
        if len(goodChildren) == 0:
            return
        maximalHeuristic = -1
        for child in goodChildren:
            heur = self.heuristic(child)
            if heur > maximalHeuristic:
                bestChild = child
                maximalHeuristic = heur
        self.board = bestChild
        self.search(goodChildren)

    def initialCandidates(self):
        candidates = []
        for row in xrange(self.size):
            for col in xrange(self.size):
                candidate = copy.deepcopy(self.board)
                candidate[row][col][1].append(candidate[row][col][0])
                candidate[row][col][0] = 0
                candidates.append(candidate)
        return candidates

    def childrenVia(self, method):
        children = []
        for row in xrange(self.size):
            for col in xrange(self.size):
                children.extend(method(self.board, row, col))
        return children

    def getBlock(self, row, col):
        (m, n) = (self.m, self.n)
        return (row / m, col / n)

    def parentsVia(self, child, method):
        parents = []
        for row in xrange(self.size):
            for col in xrange(self.size):
                parents.extend(method(child, row, col))
        return parents

    def defineForwards(self):
        def singleCandidate(board, row, col):
            parents = []
            if board[row][col][0] == 0 and len(board[row][col][1]) == 1:
                newBoard = copy.deepcopy(board)
                newBoard[row][col][0] = board[row][col][1][0]
                newBoard[row][col][1] = []
                parents.append(newBoard)
            return parents
        def exclusion(board, row, col):
            parents = []
            if board[row][col][0] != 0:
                newBoard = copy.deepcopy(board)
                for otherRow in xrange(self.size):
                    for otherCol in xrange(self.size):
                        if self.isNeighbor((row, col), (otherRow, otherCol)):
                            if board[otherRow][otherCol][1] != []:
                                if board[row][col][0] in board[otherRow][otherCol][1]:
                                    newBoard[otherRow][otherCol][1].remove(board[row][col][0])
            return parents
        def blockIntersection(board, row, col):
            parents = []
            for value in self.values:
                rows = set()
                cols = set()
                for ((row, col), element) in self[self.getBlock(row, col)]:
                    if value in element[1]:
                        rows.add(row)
                        cols.add(col)
                if len(rows) == 1:
                    # This value is only in one row for this block.
                    # Therefore, the value cannot exist anywhere else in the row
                    newBoard = copy.deepcopy(board)
                    rowChange = rows.pop()
                    for colChange in xrange(self.size):
                        if self.getBlock(rowChange, colChange) != self.getBlock(row, col):
                            if value in self.board[rowChange][colChange][1]:
                                self.board[rowChange][colChange][1].remove(value)
                    parents.append(newBoard)
                if len(cols) == 1:
                    newBoard = copy.deepcopy(board)
                    colChange = cols.pop()
                    for rowChange in xrange(self.size):
                        if self.getBlock(rowChange, colChange) != self.getBlock(row, col):
                            if value in self.board[rowChange][colChange][1]:
                                self.board[rowChange][colChange][1].remove(value)
                    parents.append(newBoard)
                # Now check if the row or col only contains pencil marks for a value within one block
                blocks = set()
                for colObs in xrange(self.size):
                    if value in self.board[row][colObs][1]:
                        blocks.add(self.getBlock(row, colObs))
                if len(blocks) == 1:
                    newBoard = copy.deepcopy(board)
                    blockChange = blocks.pop()
                    for (pos, ele) in self[blockChange]:
                        if pos[0] != row and value in ele[1]:
                            newBoard[pos[0]][pos[1]][1].remove(value)
                    parents.append(newBoard)
                # Similarly for rows
                blocks = set()
                for rowObs in xrange(self.size):
                    if value in self.board[rowObs][col][1]:
                        blocks.add(self.getBlock(rowObs, col))
                if len(blocks) == 1:
                    # Only one block
                    newBoard = copy.deepcopy(board)
                    blockChange = blocks.pop()
                    for (pos, ele) in self[blockChange]:
                        if pos[1] != col and value in ele[1]:
                            newBoard[pos[0]][pos[1]][1].remove(value)
                    parents.append(newBoard)
            return parents
        def coveringSet(board, row, col):
            parents = []
            return parents
        def xWing(board, row, col):
            parents = []
            return parents

        self.forwards = [singleCandidate,
                         exclusion,
                         blockIntersection,
                         coveringSet,
                         xWing]

    def defineInverses(self):
        def singleCandidate(board, row, col):
            children = []
            if board[row][col][1] == [] and board[row][col][0] != 0:
                newBoard = copy.deepcopy(board)
                newBoard[row][col][1] = [board[row][col][0]]
                newBoard[row][col][0] = 0
                children.append(newBoard)
            return children
        def exclusion(board, row, col):
            children = []
            if board[row][col][0] != 0:
                newBoard = copy.deepcopy(board)
                for otherRow in xrange(self.size):
                    for otherCol in xrange(self.size):
                        if self.isNeighbor((row, col), (otherRow, otherCol)):
                            if board[otherRow][otherCol][1] != []:
                                if board[row][col][0] not in board[otherRow][otherCol][1]:
                                    newBoard[otherRow][otherCol][1].append(board[row][col][0])

            return children
        def blockIntersection(board, row, col):
            children = []
            return children
        def coveringSet(board, row, col):
            children = []
            return children
        def xWing(board, row, col):
            children = []
            return children

        self.inverses = [singleCandidate,
                         exclusion,
                         blockIntersection,
                         coveringSet,
                         xWing]

    def heuristic(self, child):
        return random()

    def hideValues(self):
        timer = time()
        self.defineForwards()
        self.defineInverses()
        self.resetPencilMarks()
        children = self.initialCandidates()
        index = randint(0, len(children) - 1)
        child = children[index]
        self.board = child
        self.search(children)
        elapsed = time() - timer
        output = '%0.3f sec' % (elapsed) if elapsed > 1 else '%0.3f msec' % (elapsed * 1000)
        print
        print 'Time: %s' % (output)
        print

    def rowCol(self, x, y):
        x -= self.pad
        y -= self.pad
        width = int(self.canvas.cget('width')) - (2 * self.pad)
        height = int(self.canvas.cget('height')) - (2 * self.pad)
        size = self.size
        col = x * size / width
        row = y * size / height
        return (row, col)

    def coordinate(self, row, col):
        x = self.pad + (int(self.canvas.cget('width')) - (2 * self.pad)) * (col + 0.5) / (self.size)
        y = self.pad + (int(self.canvas.cget('height')) - (2 * self.pad)) * (row + 0.5) / (self.size)
        return (x, y)

    def cellCoordinates(self, row, col):
        corners = [(-0.5, -0.5), (-0.5, +0.5), (+0.5, +0.5), (+0.5, -0.5)]
        coords = ()
        for corner in corners:
            coords += (self.coordinate(row + corner[0], col + corner[1]))
        return coords

    def draw(self):
        self.squares = []
        for row in xrange(self.size):
            for col in xrange(self.size):
                self.squares.append(self.canvas.create_polygon(self.cellCoordinates(row, col),
                                           fill=color(self.m, self.n, row, col),
                                           outline='#000000', width=0))
                self.canvas.tag_bind(self.squares[-1], '<Enter>', lambda(e): entered(row, col, e))

    def deleteObjects(self):
        pass


def playSudoku(m, n):
    width = 800
    height = 800

    root = Tk()
    canvas = Canvas(root, width=width, height=height, background='#ffffff')

    board = SudokuBoard(m, n, canvas)
    # board.set()

    buttonFrame = Frame(root)
    setButton = Button(buttonFrame, text="New", command=lambda: board.set(seedVal=random()))
    setButton.pack()

    canvas.bind('<ButtonRelease-1>', lambda(e): release(e, canvas, board))

    canvas.pack()
    buttonFrame.pack()

    root.mainloop()

if __name__ == '__main__':
    playSudoku(2, 2)
