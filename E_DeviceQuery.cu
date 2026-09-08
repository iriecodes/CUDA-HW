//Name: Irie Slabach
//Device query
//nvcc E_DeviceQuery.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

/*
 Purpose:
 To learn how to find out what is on the GPU(s) in your machine and if you even have a GPU.
*/

/*
 Explain what you did to fix the code:
I added the missing fields and added the descriptions as inline comments.
 	https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
 
*/

//Include files
#include <stdio.h>
#include <stdlib.h>
//Defines

//Global variables

//Function prototypes
void cudaErrorCheck(const char*, int);

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);

		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");

		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

		//Added below
		printf("ECC enabled: %d\n", prop.ECCEnabled); //Device has ECC support enabled
		printf("Access policy max window size: %zu\n", prop.accessPolicyMaxWindowSize); //Maximum cudaAccessPolicyWindow::num_bytes
		printf("Async engine count: %d\n", prop.asyncEngineCount); //Number of asynchronous engines
		printf("Can map host memory: %d\n", prop.canMapHostMemory); //Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
		printf("Can use host pointer for registered memory: %d\n", prop.canUseHostPointerForRegisteredMem); //Device can access host registered memory at same virtual address as CPU
		printf("Cluster launch: %d\n", prop.clusterLaunch); //Device supports cluster launch
		printf("Compute preemption supported: %d\n", prop.computePreemptionSupported); //Device supports Compute Preemption
		printf("Concurrent kernels: %d\n", prop.concurrentKernels); //Can possibly execute multiple kernels concurrently
		printf("Concurrent managed access: %d\n", prop.concurrentManagedAccess); //Can coherently access managed memory concurrently with CPU
		printf("Cooperative launch: %d\n", prop.cooperativeLaunch); //Supports cooperative kernels via cudaLaunchCooperativeKernel
		printf("Deferred mapping CUDA array supported: %d\n", prop.deferredMappingCudaArraySupported); //Supports deferred mapping CUDA arrays and mipmapped arrays
		printf("Device NUMA config: %d\n", prop.deviceNumaConfig); //NUMA configuration, cudaDeviceNumaConfig enum
		printf("Device NUMA ID: %d\n", prop.deviceNumaId); //NUMA node ID of GPU memory
		printf("Direct managed memory access from host: %d\n", prop.directManagedMemAccessFromHost); //Host can directly access managed memory without migration
		printf("Global L1 cache supported: %d\n", prop.globalL1CacheSupported); //Supports caching globals in L1
		printf("GPUDirect RDMA flush writes options: %u\n", prop.gpuDirectRDMAFlushWritesOptions); //Bitmask per cudaFlushGPUDirectRDMAWritesOptions
		printf("GPUDirect RDMA supported: %d\n", prop.gpuDirectRDMASupported); //1 if supports GPUDirect RDMA APIs, else 0
		printf("GPUDirect RDMA writes ordering: %d\n", prop.gpuDirectRDMAWritesOrdering); //See cudaGPUDirectRDMAWritesOrdering enum
		printf("GPU PCI device ID: %d\n", prop.gpuPciDeviceID); //Combined 16-bit PCI device ID and 16-bit vendor ID
		printf("GPU PCI subsystem ID: %d\n", prop.gpuPciSubsystemID); //Combined 16-bit PCI subsystem ID and subsystem vendor ID
		printf("Host native atomic supported: %d\n", prop.hostNativeAtomicSupported); //Device-host link supports native atomic operations
		printf("Host NUMA ID: %d\n", prop.hostNumaId); //Closest host NUMA node ID or -1 if unsupported
		printf("Host NUMA multinode IPC supported: %d\n", prop.hostNumaMultinodeIpcSupported); //Supports HostNuma location IPC between nodes
		printf("Host register read-only supported: %d\n", prop.hostRegisterReadOnlySupported); //Supports cudaHostRegisterReadOnly
		printf("Host register supported: %d\n", prop.hostRegisterSupported); //Supports host memory registration via cudaHostRegister
		printf("Integrated: %d\n", prop.integrated); //Device integrated vs discrete
		printf("IPC event supported: %d\n", prop.ipcEventSupported); //Supports IPC Events
		printf("Multi-GPU board: %d\n", prop.isMultiGpuBoard); //Device on multi-GPU board
		printf("L2 cache size: %d bytes\n", prop.l2CacheSize); //L2 cache size bytes
		printf("Local L1 cache supported: %d\n", prop.localL1CacheSupported); //Supports caching locals in L1
		printf("LUID: ");
		for (int j = 0; j < 8; j++) printf("%02x", prop.luid[j]);
		printf("\n"); //8-byte locally unique identifier, undefined on TCC/non-Windows
		printf("LUID device node mask: %u\n", prop.luidDeviceNodeMask); //LUID device node mask, undefined on TCC/non-Windows
		printf("Major compute capability: %d\n", prop.major); //Major compute capability
		printf("Managed memory: %d\n", prop.managedMemory); //Supports allocating managed memory
		printf("Max blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor); //Max resident blocks per multiprocessor
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); //Max size of each grid dimension
		printf("Max surface 1D: %d\n", prop.maxSurface1D); //Max 1D surface size
		printf("Max surface 1D layered: (%d, %d)\n", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]); //Max 1D layered surface dimensions
		printf("Max surface 2D: (%d, %d)\n", prop.maxSurface2D[0], prop.maxSurface2D[1]); //Max 2D surface dimensions
		printf("Max surface 2D layered: (%d, %d, %d)\n", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]); //Max 2D layered surface dimensions
		printf("Max surface 3D: (%d, %d, %d)\n", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]); //Max 3D surface dimensions
		printf("Max surface cubemap: %d\n", prop.maxSurfaceCubemap); //Max cubemap surface dimensions
		printf("Max surface cubemap layered: (%d, %d)\n", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]); //Max cubemap layered surface dimensions
		printf("Max texture 1D: %d\n", prop.maxTexture1D); //Max 1D texture size
		printf("Max texture 1D layered: (%d, %d)\n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]); //Max 1D layered texture dimensions
		printf("Max texture 1D mipmap: %d\n", prop.maxTexture1DMipmap); //Max 1D mipmapped texture size
		printf("Max texture 2D: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]); //Max 2D texture dimensions
		printf("Max texture 2D gather: (%d, %d)\n", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]); //Max 2D texture dimensions for gather
		printf("Max texture 2D layered: (%d, %d, %d)\n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]); //Max 2D layered texture dimensions
		printf("Max texture 2D linear: (%d, %d, %d)\n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], prop.maxTexture2DLinear[2]); //Max width,height,pitch for 2D textures bound to pitched memory
		printf("Max texture 2D mipmap: (%d, %d)\n", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]); //Max 2D mipmapped texture dimensions
		printf("Max texture 3D: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]); //Max 3D texture dimensions
		printf("Max texture 3D alt: (%d, %d, %d)\n", prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]); //Max alternate 3D texture dimensions
		printf("Max texture cubemap: %d\n", prop.maxTextureCubemap); //Max cubemap texture dimensions
		printf("Max texture cubemap layered: (%d, %d)\n", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]); //Max cubemap layered texture dimensions
		printf("Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); //Max size of each block dimension
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock); //Max threads per block
		printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor); //Max resident threads per multiprocessor
		printf("Memory bus width: %d bits\n", prop.memoryBusWidth); //Global memory bus width bits
		printf("Memory pool supported handle types: %u\n", prop.memoryPoolSupportedHandleTypes); //Bitmask of handle types supported with mempool IPC
		printf("Memory pools supported: %d\n", prop.memoryPoolsSupported); //Supports cudaMallocAsync/cudaMemPool family
		printf("Minor compute capability: %d\n", prop.minor); //Minor compute capability
		printf("MPS enabled: %d\n", prop.mpsEnabled); //Contexts shared via MPS
		printf("Multi-GPU board group ID: %d\n", prop.multiGpuBoardGroupID); //Unique identifier for group on same multi-GPU board
		printf("Pageable memory access: %d\n", prop.pageableMemoryAccess); //Supports coherently accessing pageable memory without cudaHostRegister
		printf("Pageable memory access uses host page tables: %d\n", prop.pageableMemoryAccessUsesHostPageTables); //Accesses pageable memory via host page tables
		printf("PCI bus ID: %d\n", prop.pciBusID); //PCI bus ID
		printf("PCI device ID: %d\n", prop.pciDeviceID); //PCI device ID
		printf("PCI domain ID: %d\n", prop.pciDomainID); //PCI domain ID
		printf("Persisting L2 cache max size: %zu bytes\n", prop.persistingL2CacheMaxSize); //Max L2 persisting lines capacity bytes
		printf("Registers per multiprocessor: %d\n", prop.regsPerMultiprocessor); //32-bit registers available per multiprocessor
		printf("Reserved: ");
		for (int j = 0; j < 56; j++) printf("%d ", prop.reserved[j]);
		printf("\n"); //Reserved for future use
		printf("Reserved shared memory per block: %zu\n", prop.reservedSharedMemPerBlock); //Shared memory reserved by CUDA driver per block
		printf("Shared memory per block opt-in: %zu\n", prop.sharedMemPerBlockOptin); //Max shared memory per block usable by opt-in
		printf("Shared memory per multiprocessor: %zu\n", prop.sharedMemPerMultiprocessor); //Shared memory per multiprocessor
		printf("Sparse CUDA array supported: %d\n", prop.sparseCudaArraySupported); //Supports sparse CUDA arrays/mipmapped arrays
		printf("Stream priorities supported: %d\n", prop.streamPrioritiesSupported); //Supports stream priorities
		printf("Surface alignment: %zu\n", prop.surfaceAlignment); //Alignment requirements for surfaces
		printf("TCC driver: %d\n", prop.tccDriver); //1 if Tesla using TCC, else 0
		printf("Texture pitch alignment: %zu\n", prop.texturePitchAlignment); //Pitch alignment requirement for texture references
		printf("Timeline semaphore interop supported: %d\n", prop.timelineSemaphoreInteropSupported); //External timeline semaphore interop supported
		printf("Unified addressing: %d\n", prop.unifiedAddressing); //Shares unified address space with host
		printf("Unified function pointers: %d\n", prop.unifiedFunctionPointers); //Supports unified pointers
		printf("UUID: ");
		for (int j = 0; j < 16; j++) printf("%02x", prop.uuid.bytes[j]);
		printf("\n"); //16-byte unique identifier
		printf("\n");
	}	
	return(0);
}

