import tkinter as tk
from tkinter import simpledialog

class PopUp():

    def __init__(self, root):
        self.graph = -2
        self.imf = -2
        self.sinbin = -2
        self.mtl = -2
        self.norm = False

        self.root = root

    def fourth_popup(self,first_choice, second_choice, third_choice):
        def on_fourth_choice(fourth_choice):
            #print(f"fourth choice: {fourth_choice}")
            fourth_window.destroy()
            self.graph = first_choice
            self.imf = second_choice
            self.sinbin = third_choice
            if first_choice != 1:
                self.mtl = fourth_choice
            else:
                self.norm = fourth_choice
            self.root.quit()

        if first_choice != 1:
            fourth_window = tk.Toplevel(self.root)
            fourth_window.title("Metallicity Choice")

            tk.Label(fourth_window, text=f"Graph: {first_choice}, IMF Choice: '{second_choice}',  Star Type: '{third_choice}', Choose Metallicity: ").pack(pady=10)
            choices4 = ["em5","em4","001","002","003","004","006","008","010","014","020","030","040"] 
            for i, option in enumerate(choices4):
                tk.Button(fourth_window, text=choices4[i], command=lambda opt=option: on_fourth_choice(opt)).pack(pady=5)
        else:
            #print(f"No metallicity required - Graph: '{first_choice}', IMF Choice: '{second_choice}' Star Type: {third_choice}")
            fourth_window = tk.Toplevel(self.root)
            fourth_window.title("Metallicity Choice")

            tk.Label(fourth_window, text=f"Initial Graph Selection: {first_choice} \n supernova-{third_choice}-imf{second_choice} \nIMF Choice: '{second_choice}' Star Type: '{third_choice}'\n Normalise Againts ccSNII?").pack(pady=10)
            choices4 = [True,False] 
            choicesVisual4 = ["Yes","No"] 
            for i, option in enumerate(choices4):
                tk.Button(fourth_window, text=choicesVisual4[i], command=lambda opt=option: on_fourth_choice(opt)).pack(pady=5)


    def third_popup(self,first_choice, second_choice):
        def on_third_choice(third_choice):
            #print(f"third choice: {third_choice}")
            third_window.destroy()
            self.fourth_popup(first_choice, second_choice, third_choice)

        third_window = tk.Toplevel(self.root)
        third_window.title("Star Type Choice")

        tk.Label(third_window, text=f"Graph: {first_choice}, IMF Choice: '{second_choice}', Choose Single or Binary:").pack(pady=10)
        choices3 = ['sin','bin']
        choicesVisual3 = ['Single', 'Binary']
        for i, option in enumerate(choices3):
            tk.Button(third_window, text=choicesVisual3[i], command=lambda opt=option: on_third_choice(opt)).pack(pady=5)

    def second_popup(self,first_choice):
        def on_second_choice(second_choice):
            #print(f"Second choice: {second_choice}")
            second_window.destroy()
            self.third_popup(first_choice, second_choice)

        second_window = tk.Toplevel(self.root)
        second_window.title("IMF Choice")

        tk.Label(second_window, text=f"Graph Choice: {first_choice}, Choose IMF: ").pack(pady=10)
        choices2 = ['_chab100','_chab300','100_100','100_300','135_100','135_300','135all_100','170_100','170_300']
        choicesVisual2 = ['Chabrier 100','Chabrier 300','IMF 100_100','IMF 100_300','IMF 135_100','IMF 135_300','IMF 135all_100','IMF 170_100','IMF 170_300']
        for i, option in enumerate(choices2):
            tk.Button(second_window, text=choicesVisual2[i], command=lambda opt=option: on_second_choice(opt)).pack(pady=5)

    def first_popup(self):
        def on_first_choice(first_choice):
            #print(f"First choice: {first_choice}")
            first_window.destroy()
            self.second_popup(first_choice)

        first_window = tk.Toplevel(self.root)
        first_window.title("Graph Choice")

        tk.Label(first_window, text="Choose graph type:").pack(pady=10)
        choices1 = [0, 1, 2]
        choicesVisual1 = ["Age vs SN Rate", "Metallicity vs SN Rate", "Photon rate vs SN Rate"]
        for i,option in enumerate(choices1):
            tk.Button(first_window, text=choicesVisual1[i], command=lambda opt=option: on_first_choice(opt)).pack(pady=5)

class SecondaryPopUp():

    def __init__(self, root, graph, imf1, sinbin1, mtl1, norm1):
        self.graph = graph
        self.imf = -2
        self.sinbin = -2
        self.mtl = -2
        self.norm = norm1

        self.root = root
        
        self.imf1 = imf1
        self.sinbin1 = sinbin1
        self.mtl1 = mtl1

    def fourth_popup(self,first_choice, second_choice, third_choice):
        def on_fourth_choice(fourth_choice):
            #print(f"fourth choice: {fourth_choice}")
            fourth_window.destroy()
            self.graph = first_choice
            self.imf = second_choice
            self.sinbin = third_choice
            self.mtl = fourth_choice
            
            self.root.quit()

        #print(first_choice)
        if first_choice != 1:
            fourth_window = tk.Toplevel(self.root)
            fourth_window.title("Metallicity Choice")

            tk.Label(fourth_window, text=f"Initial Graph Selection: {self.graph} \n supernova-{self.sinbin1}-imf{self.imf1}.z{self.mtl1} \nIMF Choice: '{second_choice}' Star Type: '{third_choice}'\n Choose another metallicity").pack(pady=10)
            choices4 = ["em5","em4","001","002","003","004","006","008","010","014","020","030","040"] 
            for i, option in enumerate(choices4):
                tk.Button(fourth_window, text=choices4[i], command=lambda opt=option: on_fourth_choice(opt)).pack(pady=5)
        else:
            self.graph = first_choice
            self.imf = second_choice
            self.sinbin = third_choice
            self.mtl = -1
            self.root.quit()


    def third_popup(self,first_choice, second_choice):
        def on_third_choice(third_choice):
            #print(f"third choice: {third_choice}")
            third_window.destroy()
            self.fourth_popup(first_choice, second_choice, third_choice)

        third_window = tk.Toplevel(self.root)
        third_window.title("Star Type Choice")
        
        if self.graph != 1:
            tk.Label(third_window, text=f"Initial Graph Selection: {self.graph} \n supernova-{self.sinbin1}-imf{self.imf1}.z{self.mtl1} \n IMF Choice:'{second_choice}'\n Choose Single or Binary:").pack(pady=10)
        else:
            tk.Label(third_window, text=f"Initial Graph Selection: {self.graph} \n supernova-{self.sinbin1}-imf{self.imf1} \n IMF Choice:'{second_choice}'\n Choose Single or Binary:").pack(pady=10)
        choices3 = ['sin','bin']
        choicesVisual3 = ['Single', 'Binary']
        for i, option in enumerate(choices3):
            tk.Button(third_window, text=choicesVisual3[i], command=lambda opt=option: on_third_choice(opt)).pack(pady=5)

    def second_popup(self,first_choice):
        def on_second_choice(second_choice):
            #print(f"Second choice: {second_choice}")
            second_window.destroy()
            self.third_popup(first_choice, second_choice)

        second_window = tk.Toplevel(self.root)
        second_window.title("IMF Choice")

        if self.graph != 1:
            tk.Label(second_window, text=f"Initial Graph Selection: {self.graph} \n supernova-{self.sinbin1}-imf{self.imf1}.z{self.mtl1}  \nChoose another IMF:  ").pack(pady=10)
        else:
            tk.Label(second_window, text=f"Initial Graph Selection: {self.graph} \n supernova-{self.sinbin1}-imf{self.imf1}  \nChoose another IMF:  ").pack(pady=10)
        choices2 = ['_chab100','_chab300','100_100','100_300','135_100','135_300','135all_100','170_100','170_300']
        choicesVisual2 = ['Chabrier 100','Chabrier 300','IMF 100_100','IMF 100_300','IMF 135_100','IMF 135_300','IMF 135all_100','IMF 170_100','IMF 170_300']
        for i, option in enumerate(choices2):
            tk.Button(second_window, text=choicesVisual2[i], command=lambda opt=option: on_second_choice(opt)).pack(pady=5)

    def first_popup(self):
        self.second_popup(self.graph)