import os,random,string,shutil


# scan the src dir, and scan the each subdir ,then rename the file
#
# the result : ********_dirName_0000.jpg
#
#
#
#

path=".";
src="./src";
#path=raw_input("src path:");
directorys=os.listdir(src);
i=0;
def rename():
	
	for directory in directorys:
	    print "####",directory," is doing ####";
	    path=src+"/"+directory;
	    files=subfilesName(path)
	    i=0;
	    for f in files:
		print os.path.join(f),"...."
	    	if os.path.isfile(os.path.join(path,f))==True:
			i=i+1;
			name='{0}_{1:0{2}d}'.format(directory,i,4);
			print os.path.join(path,f),os.path.join(path,name);
			os.rename(os.path.join(path,f),os.path.join(path,name));
			print f,"-->",name

# get sub dirs
def subdirs(path):
    dl = [];
    for i in os.walk(path, False):
        for d in i[1]:
            dl.append(os.path.join(path, d))
    return dl

#get sub file
def subfiles(path):
    fl = [];
    for i in os.walk(path, False):
        for f in i[2]:
            fl.append(os.path.join(path, f))
    return fl

def subfilesName(path):
    fl = [];
    for i in os.walk(path, False):
        for f in i[2]:
            fl.append(f)
    return fl


def random_str(randomlength=8):
    str = ''
    chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
    length = len(chars) - 1
    random = Random()
    for i in range(randomlength):
        str+=chars[random.randint(0, length)]
    return str

def random_string(randomlength=8):
    a = list(string.ascii_letters)
    random.shuffle(a)
    return ''.join(a[:randomlength])
rename();
