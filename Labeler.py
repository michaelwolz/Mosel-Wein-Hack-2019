#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Philipp
#
# Created:     17.02.2019
# Copyright:   (c) Philipp 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class MainWindow:
    """ Main Class. Create GUI and all functions.
    """
    def __init__(self, master):
        self.master = master
        self.label = 0
        self.master.title("Labeler")
        self.labelVar = StringVar()
        self.labelVar.set('0')

        # adding UI widgets
        self.next_button = Button(self.master, text="Next Image", command=self.next)
        self.next_button.grid(row = 1,column = 2,sticky = W+E)

        self.pickdir_button = Button(self.master, text="Pick Image Directory", command=self.pickImageDir)
        self.pickdir_button.grid(row = 1, column = 1, sticky = W+E)

        self.label_start = Button(self.master, text="Label: Start", command=self.labelStart)
        self.label_start.grid(row = 1, column = 5, sticky = W+E)

        self.label_stop = Button(self.master, text="Label: Stop", command=self.labelStop)
        self.label_stop.grid(row = 1, column = 6, sticky = W+E)

        self.label_entry = Entry(self.master, width = 2, textvariable = self.labelVar)
        self.label_entry.grid(row = 1, column = 4, sticky = W+E)

        self.close_button = Button(self.master, text="Exit", command=master.destroy)
        self.close_button.grid(row = 1, column = 7, sticky = W+E)

        # Key bind for mouse scroll-wheel 'down' to switch to next image
        self.master.bind('<Button-5>', self.next)

    def labelStart(self):
        """ Set label to 1
        """
        self.label = 1
        self.labelVar.set('1')

    def labelStop(self):
        """ Set label to 0
        """
        self.label = 0
        self.labelVar.set('0')

    def next(self, event = 0):
        """ Iterate one step through image list.
        """
        # First rename last image according to label
        if self.label == 1:
            os.rename(self.img_now, os.path.split(self.img_now)[0] + '/1_' + os.path.split(self.img_now)[1])
        else:
            os.rename(self.img_now, os.path.split(self.img_now)[0] + '/0_' + os.path.split(self.img_now)[1])

        # Iterate to next image and update UI widget
        try:
            self.img_now = next(self.file)
            self.img = ImageTk.PhotoImage(Image.open(self.img_now).resize((900, 750)))
            self.panel.configure(image = self.img)
        except StopIteration:
            messagebox.showerror("Error!", "Letztes Bild erreicht!")

    def pickImageDir(self):
        """ Let user pick frame directory, build a file list and show the first
            image.
        """
        self.imageDir = filedialog.askdirectory()

        # Search for .jpg files in frame directory
        self.files = []
        for root, dirs, files in os.walk(self.imageDir):
            for file in files:
                if file.endswith('.jpg'):
                    self.files.append(os.path.join(root, file))

        # Start iteration through .jpg files
        self.file = iter(self.files)
        self.img_now = next(self.file)

        # Create image widget with first image
        self.img = ImageTk.PhotoImage(Image.open(self.img_now).resize((800, 632)))
        self.panel = Label(self.master, height = 632, width = 800, image = self.img)
        self.panel.photo = self.img
        self.panel.grid(row = 2, column = 1, columnspan = 7)

################################################################################
# main function
################################################################################
if __name__ == '__main__':
    # starting GUI
    root = Tk()
    gui = MainWindow(root)

    root.mainloop()