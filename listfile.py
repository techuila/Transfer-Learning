import os

# Place location of your training folder
folder = ""

filenames= os.listdir(folder) # get all files' and folders' names in the current directory

result = []
for filename in filenames: # loop through all the files and folders
    if os.path.isdir(os.path.join(os.path.abspath(folder), filename)): # check whether the current object is a folder or not
        result.append(filename)
        
result.sort()

f= open('list.txt','w')
for index,filename in enumerate(result):
    f.write("%s \n"%(filename))

f.close()