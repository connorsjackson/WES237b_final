import os, random, csv, numpy as np


N = 5000000;random.seed(N); set_num = 7;sN = str(N);sBaseDir = './Dataset/';DEBUG_MODE = False

os.chdir("C:\\_RLS_GoogleDrive\\Jupyter_NOTEBOOK\\FFT")
sPath = sBaseDir+str(set_num).strip()
try:
    os.makedirs(sPath)  #'./Dataset/1'
except OSError:
    pass
os.chdir("C:\\_RLS_GoogleDrive\\Jupyter_NOTEBOOK\\FFT\\Dataset\\"+str(set_num).strip()) #"C:\\_RLS_GoogleDrive\\Jupyter_NOTEBOOK\\FFT\\Dataset\\1"

# Dataset 1
### Input files #####################
sFileName = "input0.raw";text_file = open(sFileName, "w")
text_file.write('# (' + sN + ', 1)' + '\n');  # (8, 1)
for x in range(N):
    rn = random.uniform(-99.00, 99.00);sFormat = f"{rn:.2f}" + ' \n';text_file.write(sFormat);
text_file.close()

sFileName = "input1.raw";text_file = open(sFileName, "w")
text_file.write('# (' + sN + ', 1)' + '\n')
for x in range(N):
    rn = random.uniform(-99.00, 99.00);sFormat = f"{rn:.2f}" + ' \n';text_file.write(sFormat);
text_file.close()

### Read Input files ################
arr_re, arr_im = np.empty(shape=N, dtype=float), np.empty(shape=N, dtype=float)
sFileName = 'input0.raw'
with open(sFileName) as f:
    reader = csv.reader(f, delimiter=' ')
    i = 0
    for row in reader:
        if i==0:
            i = i + 1;continue
        else:
            arr_re[i-1] = float(row[0].strip());i = i + 1

sFileName = 'input1.raw'
with open(sFileName) as f:
    reader = csv.reader(f, delimiter=' ')
    i = 0
    for row in reader:
        if i==0:
            i = i + 1;continue
        else:
            arr_im[i-1] = float(row[0].strip());i = i + 1

if bool(DEBUG_MODE):
    print("\n*** input0 & input1 files data:")
    for ith in range(N):
        Re  = str(arr_re[ith])
        Im = str(arr_im[ith])
        print( Re+'\t' + "+" + '\t'+ Im +"j")

### FFT(complex input) ################
input_Complex = arr_re + (arr_im * 1j)
X = np.fft.fft(input_Complex)
real_part = X.real
imaginary_part = X.imag

sFileName = "expected0.raw";text_file = open(sFileName, "w")
text_file.write('# (' + sN + ', 1)' + '\n')
for x in range(N):
    val = real_part[x.__index__()];sFormat = f"{val:.2f}" + ' \n';text_file.write(sFormat);
text_file.close()


sFileName = "expected1.raw";text_file = open(sFileName, "w")
text_file.write('# (' + sN + ', 1)' + '\n');
for x in range(N):
    val = imaginary_part[x.__index__()];sFormat = f"{val:.2f}" + ' \n';text_file.write(sFormat);
text_file.close()

if bool(DEBUG_MODE):
    print("\n*** Expected0 & expected1:")
    print("X real_part:")
    print(real_part)
    print("X imaginary_part:")
    print(imaginary_part)