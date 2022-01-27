# A direct Jacobian total Lagrangian explicit dynamics finite element algorithm for real-time simulation of hyperelastic materials (MIT License)
![GitHub](https://img.shields.io/github/license/jinaojakezhang/DJTLED)
![GitHub top language](https://img.shields.io/github/languages/top/jinaojakezhang/DJTLED)
<p align="center"><img src="https://user-images.githubusercontent.com/93865598/148030709-0b253319-6f5b-48f4-9757-20488482e6de.PNG"></p>
This is the source repository for the paper:

| Zhang, J. (2021). A direct Jacobian total Lagrangian explicit dynamics finite element algorithm for real-time simulation of hyperelastic materials. *International Journal for Numerical Methods in Engineering*, 122(20), 5744-5772. [doi:10.1002/nme.6772](https://onlinelibrary.wiley.com/doi/10.1002/nme.6772). |
| --- |

Please cite the above paper if you use this code for your research.

If this code is helpful in your projects, please help to :star: this repo or recommend it to your friends. Thanks:blush:
## Environment:
- Windows 10
- Visual Studio 2017
-	OpenMP
## How to build:
1.	Download the source repository.
2.	Visual Studio 2017->Create New Project (Empty Project)->Project->Add Existing Item->DJTLED.cpp.
3.	Project->Properties->C/C++->Language->OpenMP Support->**Yes (/openmp)**.
4.	Build Solution (Release/x64).
## How to use:
1.	(cmd)Command Prompt->build path>project_name.exe input.txt. Example: <p align="center"><img src="https://user-images.githubusercontent.com/93865598/148030725-ce2624a0-1bc9-41d5-a3d8-2f7a6d38b9fe.PNG"></p>
2.	Output: U.vtk, Undeformed.vtk
## How to visualize:
1.	Open U.vtk. (such as using ParaView)
<p align="center"><img src="https://user-images.githubusercontent.com/93865598/148030735-8c3eb5b6-dbf1-4fbb-8866-e24c1f4f6ed9.PNG"></p>

## How to make input.txt:
1.	NH.inp (Abaqus input) is provided in the “models”, which was used to create NH_n1.txt.
## Material types:
1.	Neo-Hookean hyperelastic material.
## Boundary conditions (BCs):
1.	Node index: Disp, FixP.
2.	Element index: Gravity.
## Notes:
1.	Node and Element index can start at 0, 1, or any but must be consistent in a file.
2.	Index starts at 0: *.txt.
3.	Index starts at 1: *_n1.txt.
## Feedback:
Please send an email to jinao.zhang@hotmail.com. Thanks for your valuable feedback and suggestions.
