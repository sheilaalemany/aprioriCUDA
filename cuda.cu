#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 1024
#define MAX_NUMBER_BLOCKS 2496

/*******************************************************
*               RUNTIME MEASURING METHODS              *
*******************************************************/
struct timeval start, end; 

void starttime(){
	gettimeofday(&start,0);
}

void endtime(const char* c){
	gettimeofday(&end, 0);
	double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
	printf("%s: %f ms\n", c, elapsed);
}

/******************************************************
*   	           CUDA METHODS                       *
******************************************************/

__global__ void validSets(int* fTable, int cardinality, int nCr, int mSupport){
	int tIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if((tIndex < (cardinality + 1) * nCr) && (tIndex % (cardinality + 1) == cardinality)){
		if(fTable[tIndex] < mSupport){
			fTable[tIndex] = 0;
		}
	}
}

__global__ void counting(int* fTable, char* tTable, int row, int col, int nCr, int cardinality){
	
	__shared__ int cache[THREADS_PER_BLOCK]; //cache memory that is shared by all the threads within a block
	int bIndex = blockIdx.x; //the index value of the core
	int cacheIndex = threadIdx.x; //each thread within a core has a corresponding cache index where it stores its values

	//enter a block loop where the core index must remain lower than the amount of item sets present in the frequency table
	//at the end of each iteration the core index is increased by the amount of cores being used and loops again if possible
	for(int h = bIndex; h < nCr; h+= gridDim.x){
		
		int tIndex = threadIdx.x; //the index value of the individual thread
		int sum = 0; //keeps track of how many times an item set has been found
		int found; //a boolean value that indicates whether an item set is present within a transaction; either 0 or 1

		//enter a thread loop where i represents which transaction being scanned. Each thread within a core scans a
		// different transaction; the loop is necessary since there aren't enough threads for each transaction. Whenever
		// a scan is done i is incremented by th number of threads per block
		for(int i = tIndex; i < row; i+= blockDim.x){

			found = 1;

			//enter a loop where j represents the specific item within an item set; the iterations within the for loop
			// is dependent on the cardinality of the item sets
			for(int j = 0; j < cardinality; j++){
				
				//if an item indicated in the frequency table is not found in the transaction found is set to 0; i.e. false
				if(tTable[i * col + (fTable[bIndex * (cardinality + 1) + j])] != '1'){
					found = 0;
				}
			}	

			//if found equals 1 then the sum variable is incremented by 1
			if(found == 1){
				sum++;
			}	
		}
		
		//once any given thread exits the thread the thread loop it stores its sum value to its corresponding cache index 
		cache[cacheIndex] = sum;
		
		//the threads are synced before the overall sum is calculated to ensure all threads have finished counting;
		__syncthreads();

		//the cache is then reduced to obtain the total sum for any given item set every iteration adds two cache location 
		//together until the sum is stored at cache[0]
		int k = THREADS_PER_BLOCK/2;
		while(k != 0){
			if(cacheIndex < k){
				cache[cacheIndex] += cache[cacheIndex + k];
			}
			__syncthreads();
			k /= 2;
		}

		//takes the overall of the item set for the core index that is monitoring this specific item set and enters it into the 
		//corresponding count column within the frequency table
		if(cacheIndex == 0){
			fTable[bIndex * (cardinality + 1) + cardinality] = cache[0];
		}
		__syncthreads();
		//the core index value is incremented by the number of cores being used
		bIndex += gridDim.x;
	}
}


//factorial function
long int factorial(int x){
	int count = x;
	while (count > 1){
		x = x * (count - 1);
		count--;
	}
	if(x == 0){
		x = 1;
	}	
	return x;	
}

//combinatorics function
long int nCr(int n, int r){
	int y;
	int z;
	int w = n - 1;
	int init = n;
	int x;
	if(r > (n-r)){
	y = r;	
	}
	else{
		y = (n-r);
	}

	z = n - y;
	while(z > 1){
		n = n * w;
		w--;
		z--;
	}
	if( r > (init - r)){
		x = n/factorial(init - r);
	}
	else{
		x = n/factorial(r);
	}
	
	return  x;

}

int main() {
	
	/************************************************************************************
	*                                  Variable Declarations                            *
	************************************************************************************/	
	FILE *fPointer;
	int max = 0; 
	int size = 0; //Contains the number of lines in the given database
	int cardinality = 1; //Contains the initial cardinality of the item sets
    	int temp;
	int i = 0;
	int j, k, num, count;
	int mSupport = 8000; //Contains the support count; set to approx 10% of all transactions
	char val;
	int numBlocks = 0; 
	//While loop that traverses through the database and returns the number of transactions  
	fPointer = fopen("retail.dat", "r"); 
   	fscanf(fPointer, "%c", &val);
   	while(!feof(fPointer)){
        	if(val == '\n'){
            		size++;
        	}
       		fscanf(fPointer, "%c", &val);
    	}
    	fclose(fPointer);

    	fPointer = fopen("retail.dat", "r");
   	fscanf(fPointer, "%d", &temp);
	
	//Traverses through each transaction in order to find the max value.
    	while(!feof(fPointer)){
        	fscanf(fPointer, "%d", &temp);
        	if(max < temp){
            		max = temp;
        	}
    	}	
   	fclose(fPointer);

	printf("DATA FILE PARSED\n");
	printf("============================================\n");	
	printf("Total number of transactions found: %d\n", size);
	printf("Maximum number of unique items found: %d\n", max+1);
	printf("============================================\n");	
	printf("APRIORI IMPLEMENTATION BEGINS\n");

	starttime();

	//Creation of table
	char *cTable = (char*)malloc(sizeof(char) * (max + 1) * size); //Allocates an array of characters for each transaction	
	
	for(i=0; i < (max+1)*size; i++) {
	//	memset(cTable[i], '\0', sizeof(char) * (max + 1) * size); //Initialize all values to 0.
		cTable[i] = '\0';
	}

    	char line[400];
    	char *cNum;
    	fPointer = fopen("retail.dat", "r");
	for(i = 0; i < size; i++){
		fgets(line, 400, fPointer);

        	cNum = strtok(line, " \n");
        	
		while(cNum != NULL){
            		num = atoi(cNum);
            		cTable[i * (max + 1) + num] = '1';
            		cNum = strtok(NULL, " \n");
        	}	
    	}

	//Creating copy of transaction table in the video card memmory
	char* gpuT;
	cudaMalloc(&gpuT, size * (max + 1) * sizeof(char));
	cudaMemcpy(gpuT, cTable, (size * (max + 1) * sizeof(char)), cudaMemcpyHostToDevice);

	//Creates a frequency table of item sets with a Cardinality of 1; where the array index represents the item 
	//number. All the items have their counts initially set to zero
	int * fTable = (int *)malloc((max + 1) * (cardinality + 1) * sizeof(int));
	for(i = 0; i < max + 1; i++){
		fTable[i * (cardinality + 1)] = i;
		fTable[(i * (cardinality + 1)) + cardinality] = 0;
	}

	int* gpuF;
	cudaMalloc(&gpuF, (max + 1) * (cardinality + 1) * sizeof(int));
	cudaMemcpy(gpuF, fTable, (max + 1) * (cardinality + 1) * sizeof(int), cudaMemcpyHostToDevice);

	//setting the number of cores to be used by the gpu
	numBlocks = (max + 1);
	if(numBlocks > MAX_NUMBER_BLOCKS){
		numBlocks = MAX_NUMBER_BLOCKS;
	}
	counting<<< numBlocks, THREADS_PER_BLOCK>>>(gpuF, gpuT, size, (max + 1), (max + 1),  cardinality);
	
	//setting the number of cores to be used by the gpu
	numBlocks = (max + 1) * (cardinality + 1)/ THREADS_PER_BLOCK + 1;
	if(numBlocks > MAX_NUMBER_BLOCKS){
		numBlocks = MAX_NUMBER_BLOCKS;
	}
	validSets<<< numBlocks, THREADS_PER_BLOCK>>>(gpuF, cardinality, max + 1, mSupport);
	cudaMemcpy(fTable, gpuF, ((max + 1) * (cardinality + 1) * sizeof(int)), cudaMemcpyDeviceToHost);
	cudaFree(gpuF);

	//invalidating elements that are below the support count and counting the remaining eligible elements
	count = 0;
	for(i = 0; i < (max + 1); i++){
		if (fTable[i * (cardinality + 1) + cardinality] != 0){
			count++;
		}
	}

	//creating new table consisting of only valid items
        int iTable[count];
        j = 0;
        for(i = 0; i < (max + 1); i++){
                if (fTable[i * (cardinality + 1) + cardinality] != 0){
                        iTable[j] = fTable[i * (cardinality + 1)];			
                        j++;
                }
        }

	//creating a tabel to hold the current valid items item and their the a variable for the count of the count
	int * vTable = iTable;
	int lastCount = count;

	while(count > 1){
		cardinality++;

		//temporary array that will hold the new item sets		
		int temp[nCr(count, cardinality) * (cardinality + 1)];

		//array of previous items sets
		int oldSets[nCr(lastCount, cardinality - 1) * cardinality];

		//array that hold one old item set for insertion into table
		int oldEntry[cardinality - 1];

                //function populates old  item set
                k = 0;
                if(cardinality - 1 <= lastCount){
                        for(i = 0; (oldEntry[i] = i) < cardinality - 2; i++); 
                        for(i = 0; i < cardinality - 1; i++){
                                oldSets[(k * cardinality) + i] = vTable[oldEntry[i]];
                        }
                        k++;
                        for(;;){
                                for( i = cardinality - 2; i >= 0 && oldEntry[i] == (lastCount - (cardinality - 1) + i); i--);
                                if(i < 0){
                                        break;
                                }
                                else{
                                        oldEntry[i]++;
                                        for(++i; i < cardinality - 1; i++){
                                                oldEntry[i] = oldEntry[i - 1] + 1;
                                        }
                                        for(j = 0; j < cardinality - 1; j++){
                                                oldSets[(k * cardinality) + j] = vTable[oldEntry[j]];
                                        }
                                        k++;
                                }
                        }
                }

                for(i = 0; i < nCr(lastCount, cardinality - 1); i++){
                        oldSets[(i * cardinality) + cardinality - 1] = 0;
                }

		//array that will hold the information for a single item set before it is added to the 
		//array of all item sets
		int entry[cardinality];

		//function populates new item set
		k = 0;
		if(cardinality <= count){
			for(i = 0; (entry[i] = i) < cardinality - 1; i++);			
			for(i = 0; i < cardinality; i++){
				temp[(k*(cardinality + 1)) + i] = vTable[entry[i]];
			}
			k++;
			for(;;){
				for( i = cardinality - 1; i >= 0 && entry[i] == (count - cardinality + i); i--);
				if(i < 0){
					break;
				}
				else{
					entry[i]++;
					for(++i; i < cardinality; i++){
						entry[i] = entry[i - 1] + 1;
					}
					for(j = 0; j < cardinality; j++){
						temp[(k*(cardinality + 1)) + j] = vTable[entry[j]];
					}
					k++;
				}
			}
		}


						      
		for(i = 0; i < nCr(count, cardinality); i++){
			temp[(i*(cardinality + 1)) + cardinality ] = 0;
		}

		//counting the amount of instances of the item sets amongst the transactions
		int * gpuSet;
		cudaMalloc(&gpuSet, sizeof(int) * (cardinality + 1) * nCr(count, cardinality));
		cudaMemcpy(gpuSet, temp, sizeof(int) * (cardinality + 1) * nCr(count, cardinality), cudaMemcpyHostToDevice);
		numBlocks = nCr(count, cardinality);
		if(numBlocks > MAX_NUMBER_BLOCKS){
			numBlocks = MAX_NUMBER_BLOCKS;
		}
		counting<<< numBlocks, THREADS_PER_BLOCK>>>(gpuSet, gpuT, size, max + 1, nCr(count, cardinality), cardinality);
		cudaMemcpy(temp, gpuSet, sizeof(int) * (cardinality + 1) * nCr(count, cardinality), cudaMemcpyDeviceToHost);
		cudaFree(gpuSet);
		
                //counting the amount of instances of the item sets amongst the transactions
		cudaMalloc(&gpuSet, sizeof(int) * cardinality * nCr(lastCount, cardinality - 1));
		cudaMemcpy(gpuSet, oldSets, sizeof(int) * cardinality * nCr(lastCount, cardinality - 1), cudaMemcpyHostToDevice);
		numBlocks = nCr(lastCount, cardinality - 1);
		if(numBlocks > MAX_NUMBER_BLOCKS){
			numBlocks = MAX_NUMBER_BLOCKS;
		}
		counting<<< numBlocks, THREADS_PER_BLOCK>>>(gpuSet, gpuT, size, max + 1, nCr(lastCount, cardinality - 1), cardinality - 1);
		cudaMemcpy(oldSets, gpuSet, sizeof(int) * cardinality * nCr(lastCount, cardinality - 1), cudaMemcpyDeviceToHost);
		cudaFree(gpuSet);

		//invalidating elements that are below the support count and counting the remaining eligible elements
        	int tCount = count;
		lastCount = tCount;
		count = 0;
        	for(i = 0; i < nCr(tCount, cardinality); i++){
                	if (temp[(i*(cardinality + 1)) + cardinality] < mSupport){
                        	temp[(i * (cardinality + 1)) + cardinality] = 0;
                	}	
                	else{
                        	count++;
                	}
        	}		

		//set Table of valid items
		char valid[max + 1];
		for(i = 0; i <= max; i++){
			valid[i] = '\0';
		}

		for(i = 0; i < nCr(tCount, cardinality); i++){
			for(j = 0; j < cardinality; j++){
				if(temp[(i * (cardinality + 1)) + cardinality] > 0){
					valid[temp[(i * (cardinality + 1)) + j]] = '1';
				}
			}
		}

        	//creating new table consisting of only valid items
        	int rTable[count];
		count = 0;
        	j = 0;
        	for(i = 0; i <= max; i++){
                	if (valid[i] == '1'){
                        	rTable[j] = i;
                        	j++;
				count++;
	                }
        	}	
		vTable = rTable;

		if(count == 0){
			printf("\n=============== MOST FREQUENT SUBSETS ================\n");
	   
	        	for(i = 0; i < nCr(lastCount, cardinality - 1); i++){
				if(oldSets[(i * cardinality) + (cardinality-1)] > mSupport){
                                        printf("Set: {");
                                }
               			for(j = 0; j < cardinality; j++){
					if(oldSets[(i * cardinality) + (cardinality-1)] > mSupport){
                               			if(j == cardinality - 1){
							printf("}\t\tCount: %d\n", oldSets[(i * cardinality) + j]);
						}
						else{
							printf("'%d'", oldSets[(i * cardinality) + j]);
						}
                       		 	}	
               		 	}        
			}
			printf("\n");	
		}
	}

	endtime("Total Parallelized Implementation Time" );
}

