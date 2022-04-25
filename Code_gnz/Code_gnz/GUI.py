import tkinter as tk
from Point import Point
import numpy as np


def draw_circle(canvas, x: float, y: float, r: float):
    canvas.create_oval(x - r, y - r, x + r, y + r)

def Sample():
    root = tk.Tk()
    root.title("Create your own sample")
    
    S = []
    
    canvas = tk.Canvas(root, width = 150, height = 150)
    canvas.pack(side="top", fill="both", expand=True)
    r = 2
    
    def click_S(event):
        p = Point(event.x, event.y)
        S.append(p)
        draw_circle(canvas, p.x, p.y, r)
    
    root.bind('<Button-1>', click_S)
    root.mainloop()
    return S

def Display(L):
    root = tk.Tk()
    root.title("Result")
    
    xmin, xmax, ymin, ymax = 0, 0, 0, 0
    
    for p in L:
        xmin = min(p.x, xmin)
        xmax = max(p.x, xmax)
        ymin = min(p.y, ymin)
        ymax = max(p.y, ymax)
    
    print(xmin, xmax, ymin, ymax)
    canvas = tk.Canvas(root, width = xmax - xmin + 10, height = ymax - ymin + 10)
    canvas.pack(side="top", fill="both", expand=True)
    r = 2
    
    for p in L:
        draw_circle(canvas, p.x - xmin + 5, p.y - ymin + 5, r)
    
    def click(event):
        x, y = event.x, event.y
        print(x - 5, y - 5)
    
    root.bind('<Button-1>', click)
    root.mainloop()

def Display_dict(dL, xmax, ymax):
    root = tk.Tk()
    root.title("Result")
        
    canvas = tk.Canvas(root, width = xmax + 10, height = ymax + 10)
    canvas.pack(side="top", fill="both", expand=True)
    r = 2
    
    for L in dL.values():
        for p in L:
            draw_circle(canvas, p.x + 5, p.y + 5, r)
    
    def click(event):
        x, y = event.x, event.y
        print(x - 5, y - 5)
    
    root.bind('<Button-1>', click)
    root.mainloop()

def Display_dict2(dL, xmax, ymax):
    # Display in the form of a matrix
    M = np.zeros((xmax, ymax))
    for L in dL.values():
        for p in L:
            i, j = int(p.x), int(p.y)
            M[i, j] += 1
            if j>0:
                M[i, j-1] += 0.6
            if j<ymax-1:
                M[i, j+1] += 0.6
            if i>0:
                M[i-1, j] += 0.6
                if j>0:
                    M[i-1, j-1] += 0.3
                if j<ymax-1:
                    M[i-1, j+1] += 0.3
            if i<xmax-1:
                M[i+1, j] += 0.6
                if j>0:
                    M[i+1, j-1] += 0.3
                if j<ymax-1:
                    M[i+1, j+1] += 0.3
    return M
