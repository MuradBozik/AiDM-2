# AiDM-A2

Basic steps for each measure are;
1. Create the sparse input matrix
2. Check the measure
   - Jaccard Distance
     * Create random permutations of the rows
     * For each column (user) find the index of first non-zero rating 
     * Create the minhash signature matrix M 
   - Cosine Distance
     * Create random vectors with uniform distribution between ```[-1,+1]```, ```shape=(h, movie_number)```
     * Using dot product with input matrix, create sketch matrix M
   - Discrete Cosine Distance 
     * Create random vectors with uniform distribution between ```[-1,+1]```, ```shape=(h, movie_number)```
     * Modify input matrix as non-zero elements = 1
     * Using dot product with modified input matrix, create sketch matrix M
3. Calculate total number of bands could be generated using given row values
4. Select randomly b bands from them
5. For each band of matrix M use a hash function to divide each User into one bucket
6. Get candidates by filtering bucket elements ``` len(bucket[key]) >= 2 ```
7. Check candidate pairs by calculating distance between them and filter by threshold
8. Write similar pairs into txt file


Implementation notes
* Creating sparse input matrix takes too long (7 mins). Not to repeat this process each run, 
after creating the sparse input matrix, We saved it as **sparse_input_matrix.npz** in data folder. 
The process of creating this matrix can also be seen in our code as commented out way.
* For banding strategy we decided not to use overlapping bands. So ```b*r <= M.shape[0]```. 
* Buckets is a dictionary object, key is bucket index created by hash function and value is set of user ids which is actually column index of input matrix
* We are using huge number of buckets, but dictionary object doesn’t store any empty bucket
* We didn’t store the row permutations, we permuted and calculated using one pass on all rows
* We created one hash function to divide users into buckets, the parameters are randomly selected. Since we used numpy seed value it would create same hash function in every run.

Experimentation notes
- [ ] Use different number of parameters (h, b, r)
- [ ] Monitor the time of calculation for each method. If it is more than 30 mins per measure tune parameters or change the strategy of checking candidate pairs.
- [ ] Repeat experiments with different seed values to see consistency of algorithm.
- [ ] Make analysis of results of different experiments for each measure













