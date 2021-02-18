from tkinter import *
from tkinter import filedialog
import os

root = Tk()

# resize root window
root.geometry("500x500")

def open_simulation_program():
    #open the program
    os.system('python 3dSimulation.py')

def open_rendering_program():
    #open the program
    os.system('python 3dRender.py')

def open_castle_program():
    #open the program
    os.system('python GVI.py')

my_button = Button(root, text="Simulation Program", command=open_simulation_program)
my_button.pack(pady=20)
my_label = Label(root, text="")
my_label.pack(pady=20)

my_button = Button(root, text="3D Render", command=open_rendering_program)
my_button.pack(pady=10)
my_label = Label(root, text="")
my_label.pack(pady=20)

my_button = Button(root, text="Enable gesture control", command=open_castle_program)
my_button.pack(pady=10)
my_label = Label(root, text="")
my_label.pack(pady=20)

root.mainloop()