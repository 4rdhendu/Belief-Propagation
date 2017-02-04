/* Inspired from the scipy library 
 * Contains some functions as required in the ASSAR */

public class sciJava {

	public int[][] Liner2DInterpolator(int ptX[], int ptY[], int newptX[], int newptY[]) {
		return null;
		
	}
	
	public int[][] CloughToucher2DInterpolator(int ptX[], int ptY[], int newptX[], int newptY[]){
		return null;
		
	}
	
	public int[][] arrayReshape(int[][] matrix, int width, int height) throws Exception{
		int new_matrix[][] = new int[height][width];
		int origHeight = matrix.length;
		int origWidth = matrix[0].length;
		if(origHeight*origWidth != height*width) {
			throw new Exception("The size of the array remains unchanged.");
		}
		
		// To one-dimensional Array
		int[] matrix1D = new int[origHeight * origWidth];
        int index=0;
        int i,j;
        for(i=0; i<origHeight; i++){
            for(j=0; j<origWidth; j++){
                matrix1D[index++] = matrix[i][j];
            }
        }    
       // Reshaping the array
       index = 0;
       for(i=0; i<height; i++){
           for(j=0; j<width; j++){
               new_matrix[i][j] = matrix1D[index++];
                }
     
            }
		return new_matrix;
	}
	
	public static int[] argmax(double[][] matrix, int axis) {
		/* Find the argmax along the user-defined axis */
		int i, j;
		int[] max={};		
		/* for each row */
		if(axis==1) {
			max = new int[matrix.length];
			for(i=0;i<matrix.length;i++) {
				max[i]=0;
				for(j=0;j<matrix[0].length-1;j++) {
					if(matrix[i][j+1]>matrix[i][j]) {
						max[i]=j+1;
					}
				}
			}
		}
		
		/* for each column */
		if(axis==0) {
			max = new int[matrix[0].length];
			for(i=0;i<matrix[0].length;i++) {
				max[i]=0;
				for(j=0;j<matrix.length-1;j++) {
					if(matrix[j+1][i]>matrix[j][i]) {
						max[i]=j+1;
					}
				}
			}
		}
		
		return max;
	}
	
	public static int[] argmin(double[][] arr){
		
		//default index
		int[] index_argmin = new int[] {0, 0};
		double min = arr[0][0];
		
		int rows = arr.length;
		int cols = arr[0].length;
		
		for(int i = rows-1; i >= 0; i--){
			
			for(int j = cols-1; j >= 0; j--){
				
				if(arr[i][j] < min){
					
					index_argmin[0] = i;
					index_argmin[1] = j;
					min = arr[i][j];
				}
				
			}
			
		}
		
		return index_argmin;
		
	}
	
}	