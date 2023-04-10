# Cooking with CUDA
--------------------------------------

* Basics
	* Parallel Programming
	* What is PTX
	* What's present in your CUDA toolkit / driver update
* Harware Basics
	* Understanding your GPU Architecture
	* Roofline Model
	* Arithmetic Intensity
	* Memory Hierarchy
* Tensor Core
	* Architecture
	* Documentation
	* Programming Tensor Cores
		* CUTLASS
* Debugging Tools
	* CUDA-gdb
	* NSight
	* other tools (memcheck, racecheck, sanitizer)
	* Inspecting Assembly
* How to get better performance
	* CUDA Streams
	* CUDA Graphs
	* Software Pipelining
* Useful Libraries : 
	* Parallel Compute : 
		* NCCL
		* Thrust
		* cuRAND
		* cuFFT
	* Software Pipelining
	* Deep Learning Stack
		* cuDNN
		* Tensor-RT
		* CUTLASS
	* Data Science Stack
		* RAPIDS
	* Linear Algebra
		* cuBLAS
		* cuTensor
		* CUTLASS
		* cuSPARSE
	* CUDA For HPC
	* nvGraph
	* nvFuser
