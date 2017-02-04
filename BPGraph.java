/**
 * 
 */

import java.util.*;
//import java.util.concurrent.*;
/**
 * @class BPGraph.java
 * @date 19/06/2013
 * @brief This code is implements the belief propagation optimization algorithm as proposed by Felzenszwalb & Huttenlocher 
 */
public class BPGraph {
	/*
	 * D is the list of n matrices osize m X m, containing the unary terms
	 * edges - list of n integer lists containing edges, normally symetrical
	 * weights - list of n float lists corresponding to ->
	 * 
	 *  n -> number of nodes
	 *  m -> size of the unary term matrices
	 *  msgs -> list of n arrays of m X m float arrays, containing messages received in the previous iterations
	 *  bestlabel -> array of 2-item arrays of the currently best solution
	 */
	
	//variable descriptions are given above
	public List<double[][]> D = new ArrayList<double[][]>();
	List<int[]> edges = new ArrayList<int[]>();
	List<double[]> weights = new ArrayList<double[]>();
	
	//this may very wel be the same thing as tempThread
	List<double[][][]> msgs = new ArrayList<double[][][]>();
	
	//Don't knw if these both are same or different -> I will resolve this issue in a little while
	List<int[]> bestlabel = new ArrayList<int[]>();
	List<int[]> best_labels = new ArrayList<int[]>();
	
	//to enable multiple values return by a function
	List<int[]> best = new ArrayList<int[]>();
	double energy;
	double best_energy;
	
	//array which would be used for the multithreading thing -> call by reference kind of a thing
	List<double[][][]> tempThread;
	
	int n, m;

	/**
	 * This function initialises the msgs that will be passed from each node to its neighbour
	 * @param Nothing needs to be passed - This function directly works on the class variables
	 * @date 20/06/2013
	 */
	void init_msgs(){
		this.n = D.size();
		//double[][] buffer = D.get(0);
		
		System.out.println("Number of nodes, n = " + n);
		
		this.m = D.get(0).length;
		
		System.out.println("m = " + m);
		
		assert D.get(0)[0].length==this.m;
		
		int numedg;
		
		//this list would be used for multithreading purpose
		tempThread = new ArrayList<double[][][]>();
		
		//not needed msgs will handle everything 
		
		//so that it would be easy to set specific msgs during ||lisation
		for(int i = 0; i<this.n; i++){
			tempThread.add(null);
		}
		
		System.out.println("Initialising the tempThread vavriable to all nulls............");
		
		//initialises tthe msgs and sets them all to 0
		for(int i = 0; i<this.n; i++){
			numedg = edges.get(i).length;
			
			//double[][][] buffer = new double[this.m][this.m][numedg]; 
			//Arrays.fill(buffer, 0.0);
			
			System.out.println("numedg = " + numedg);
			//Note -> maybe this initialisation is not required 
			msgs.add(new double[this.m][this.m][numedg]);
		}
		
	}
	
	/**
	 * 
	 * @param offsets
	 * @param nproc
	 * @return
	 */
	double[][] calc_msg(int p, int q, double c, int[][] offsets){
		int[] shift = new int[2]; 
		
		int[] nr = new int[2];
		
		int sqx = 0, spx = 0, sqy = 0, spy = 0;
		
		double[][] dp;
		
		double[][] r;
		
		int rows, cols;
		
		double max;
		
		int[] e;
		
		int n;
		
		double[][][] m;
		
		int s;
		
		if(offsets != null){
			shift[0] = offsets[q][0] - offsets[p][0];
			shift[1] = offsets[q][1] - offsets[p][1];
			
			sqx = Math.max(0, shift[0]);
			sqy = Math.max(0, shift[1]);
			
			spx = Math.max(0, -1*shift[0]);
			spy = Math.max(0, shift[1]);
			
			nr[0] = this.m;
			nr[1] = this.m;
			
			/*System.out.println("shift[0] shift[1] " + shift[0] + shift[1]);
			System.out.println("sqx sqy " + sqx + sqy);
			System.out.println("spx spy " + spx + spy);
			System.out.println("nr[0] nr[1] " + nr[0] + nr[1]);*/
		}
		
		dp = this.D.get(p);

		if(dp == null){
			if(offsets == null){
				r = new double[this.m][this.m];
				//Arrays.fill(r, 0);
			}
			else{
				rows = nr[0] + (int)Math.abs(shift[0]);
				cols = nr[1] + (int)Math.abs(shift[1]);
				
				r = new double[rows][cols];
				//Arrays.fill(r, 0);
			}
		}
		else{
			if(offsets == null){
				r = new double[dp.length][dp[0].length];
				for(int i = 0; i<dp.length; i++){
					r[i] = dp[i].clone();
				}
			}
			else{
				max = dp[0][0];
				for(int i = 0; i<dp.length; i++){
					for(int j = 0; j<dp[0].length; j++){
						if(dp[i][j] > max){
							max = dp[i][j];
						}
					}
				}
				
				rows = nr[0] + (int)Math.abs(shift[0]);
				cols = nr[1] + (int)Math.abs(shift[1]);
				r = new double[rows][cols];
				
				//Arrays.fill(r, max);
				
				for(int i = 0; i<rows; i++){
					Arrays.fill(r[i], max);
				}
				int iterRows = 0;
				int iterCol = 0;
				for(int i = spx; i< spx + nr[0]; i++){
					for(int j = spy; j < spy + nr[1]; j++){
						r[i][j] = dp[iterRows][iterCol];
						iterCol += 1;
					}
					iterRows+=1;
					iterCol = 0;
				}
			}
		}
		
		//edge indices
		e = this.edges.get(p);
		n = e.length;
		
		//messages recieved by p
		m = this.msgs.get(p);
		
		for(int i = 0; i<n; i++){
			//neighbor index
			s = e[i];
			
			if(s!=q){
				if(offsets == null){
					for(int x = 0; x<r.length; x++){
						for(int y = 0; y<r[0].length; y++){
							r[x][y]+= m[x][y][i]; 
						}
					}
				}
				else{
					int iterCol = 0, iterRows = 0;
					for(int x = spx; x< spx + nr[0]; x++){
						for(int y = spy; y < spy + nr[1]; y++){
							//iterCol+=1;
							r[x][y] += m[iterRows][iterCol][i];
							iterCol += 1;
						}
						iterRows+=1;
						iterCol = 0;
					}
				}
			}
		}
		
		//segregtools sgrgtl = new segregtools();
		double[][] rout = segregtools.q_envelope2D(c, r);
		
		/*System.out.println("The value of rout as returned from the envelope2D function in segregtools");
		for(int i = 0; i<rout.length; i++){
			for(int j = 0; j<rout[0].length; j++){
				System.out.print(rout[i][j] + " ");
			}
			System.out.println();
		}*/
		
		if(offsets == null){
			return rout;
		}
		else{
			double[][] rbuffer = new double[nr[0]][nr[1]];
			//copy the array rout from the specified indexes into rbuffer and return it
			for(int x = 0; x<nr[0]; x++){
				for(int y = 0; y<nr[1]; y++){
					rbuffer[x][y] = rout[sqx + x][sqy + y];
				}
			}
			return rbuffer;
		}
	}
	
	/*
	 * The ||lisation thing
	 */
	
	/*
	 *     def msgs_for_node(self,q,offsets=None):
            e=self.edges[q]
            ne=len(e)
            mfq=np.zeros((ne,self.m,self.m))
            for i in xrange(ne): # go over all neighbors
                p=e[i]
                mfq[i]=self.calc_msg(p,q,self.weights[q][i],offsets=offsets)
            return mfq

    def msgs_for_nodes(self,q0,q1,queue,offsets):
      #print "msgs_for_nodes q0=",q0, " q1=",q1
      m=[]
      for q in xrange(q0,q1):
        m.append(self.msgs_for_node(q,offsets=offsets))
        #print "msgs_for_nodes q0=",q0, " q1=",q1, " done"
      queue.put((q0,q1,m))


    def send_all_msgs(self,offsets=None,nproc=1):
        """ Return all messages """
        if nproc==1:
          return self.send_all_msgs_seq(offsets=offsets)
        if nproc==-1:
          ncpu=multiprocessing.cpu_count()
        else:
          ncpu=nproc
        ncpu=multiprocessing.cpu_count()
        queue=multiprocessing.Queue()
        chunksize=int(math.ceil(float(self.n)/ncpu))
        prs=[]
        for q in xrange(0,self.n,chunksize):
          pr=multiprocessing.Process(target=self.msgs_for_nodes,
            args=(q,min(q+chunksize,self.n),queue,offsets))
          pr.start()
          prs.append(pr)
        print "Processes started"
        newmsgs=self.n*[None]
        #pdb.set_trace()
        for i in xrange(len(prs)):
          q0,q1,m=queue.get()
          newmsgs[q0:q1]=m
        for pr in prs:
          pr.join()
        return newmsgs

	 */
	/*
	 * protected void assignmentParallel () {
	computeDistGrid();
	Logging.logMsg(" -> fast parallel assignement running...");

	// put minimal distances to maximum
	for(float[] subarray : distances2D) {
		Arrays.fill(subarray, Float.MAX_VALUE);
	}

	final ThreadAssignment[] threads = new ThreadAssignment[Threading.nbAvailableThread()];
	int deltaImg = (int) Math.ceil(width / (float)threads.length);
	int endRange;

	for (int iThread = 0; iThread < threads.length; iThread++) {

		// Concurrently run in as many threads as CPUs
		threads[iThread] = new ThreadAssignment(img2D, gridSize, distGrid, clusterPosition, clusterColour, distances2D, labels2D);
		// for all regular regions
		// because of a rounding the last has to cover rest of image
		//endRange = (iThread < (threads.length-1)) ? (iThread+1)*deltaImg : width;
		endRange = (iThread+1)*deltaImg;
		threads[iThread].setRangeImg(iThread*deltaImg, endRange, 0, height);

	}
	
	//this has been done
	Threading.startAndJoin(threads);

}
	 */
	//here is the starting point for ||lisation
	List<double[][][]> send_all_msgs(int[][] offsets){
		//List<double[][][]> temp = new ArrayList<double[][][]>();
		
		//will be used for ||lisation -> counting the number of processors
		
		//use the Runtime.getRuntime().availableProcessors()
		//int ncpu = Runtime.getRuntime().availableProcessors();
		int ncpu = 3;
		
		//This list would be used for facilitating in the multithreading thing
		List<ThreadingSendMsgs> threads = new ArrayList<ThreadingSendMsgs>();
				
		//chunk to be processed at once
		int chunksize = (int)Math.ceil((float)(this.n)/ncpu);
		
		//dividing the data between threads
		for(int q = 0; q<this.n; q+=chunksize){
			threads.add(new ThreadingSendMsgs(q, (int)Math.min(q+chunksize, this.n), offsets, this));
			//threads.get(threads.size()-1).setRange(q, (int)Math.min(q+chunksize);
		}
		
		//starting the || things and waiting till each thread finishes
		Threading.startAndJoin(threads);
		//List<double[][][]> temp = send_all_msgs_seq(offsets);
		
		
		return tempThread;
	}
	

	List<double[][][]> send_all_msgs_seq(int[][] offsets){
		//Returns all the msgs
		
		//newmsgs = []
		
		//long startTime, estimTime, startTime1, estimTime1;
		
		List<double[][][]> newmsgs= new ArrayList<double[][][]>();
		
		int[] e;
		int ne;
		
		//this would act as an input to the fn calc_msg
		int p;
		
		double[][][] mfq;
		double[][] buffer;
	
		//List<double[][][]> mfq = new ArrayList<double[][][]>();
		
		//msgs for node q
		
		//startTime = System.nanoTime();
		for(int q = 0; q<this.n; q++){
			e = this.edges.get(q);
			ne = e.length;
			
			mfq = new double[this.m][this.m][ne];
			//mfq = np.zeros(ne, self.m, self.m); -> this has to be a 3D matrix
			
			//Arrays.fill(mfq, 0);
			
			for(int i = 0; i<ne; i++){
				p = e[i];
				//startTime1 = System.nanoTime();
				buffer = calc_msg(p, q, this.weights.get(q)[i], offsets);
				//estimTime1 = System.nanoTime() - startTime1;
				//System.out.println("Calculating the messages took " + (double)estimTime1/1000000 + "ms.");

				for(int r = 0; r<m; r++){
					for(int c = 0; c<m; c++){
						mfq[r][c][i] = buffer[r][c];
					}
				} 
			}
			newmsgs.add(mfq);
		}
		//estimTime = System.nanoTime() - startTime;
		//System.out.println("Sending all message seq took " + (double)estimTime/1000000 + "ms.");
		
		return newmsgs;
	}
	
	/**
	 * Returns the currently best labeling and the energy
	 * @param takes the offset as the input -> It is optional -> Lets see what to do about it
	 * 
	 */
	void estimate(int[][] offsets){
		double uenergy = 0;
		double[][] b;
		
		//to store the output of the argmin function
		int[] ind;
		this.best.clear();
		
		//sciJava sj = new sciJava();
		
		//for extracting a single array of integers from the edges list
		int[] e;
		
		//used in adding edge energy
		int p;

		//dx and dy
		int dx, dy;
		for(int q = 0; q<this.n; q++){
			b = get_belief(q);
			//System.out.println("b ----------------------------------");
			/*for(int i = 0; i<b.length; i++){
				for(int j = 0; j<b[0].length; j++){
					System.out.print(b[i][j] + " ");
				}
				System.out.println();
			}*/
			ind = sciJava.argmin(b);			
			//System.out.println("ind = " + ind[0] + " "  + ind[1]);
			best.add(ind);

			//System.out.println("best till now ----------------------------------");			
			/*for(int i = 0; i<best.size(); i++){
				for(int j = 0; j<best.get(0).length; j++){
					System.out.print(best.get(i)[j] + " ");
				}
				System.out.println();
			}*/
			
			if(this.D.get(q) != null){
				uenergy+= D.get(q)[ind[0]][ind[1]];
				//System.out.println("uenergy = " + uenergy);
			}
		}
		
		//add edge energy
		double penergy = 0;
		for(int q = 0; q<this.n; q++){
			
			e = this.edges.get(q);
			int ne = e.length;
			
			for(int i = 0; i<ne; i++){
				
				p = e[i];
				if(offsets == null){
					dx = this.best.get(q)[0] - this.best.get(p)[0];
					dy = this.best.get(q)[1] - this.best.get(p)[1];
					//System.out.println("in if");
					//System.out.println("dx, dy = " + dx + " " + dy);
				}
				else{
					dx = this.best.get(q)[0] - this.best.get(p)[0] + offsets[q][0] - offsets[p][0];
					dy = this.best.get(q)[1] - this.best.get(p)[1] + offsets[q][1] - offsets[p][1];
					//System.out.println("in else");
					//System.out.println("dx, dy = " + dx + " " + dy);
					
				}
				penergy+= 0.5*this.weights.get(q)[i]*(dx*dx + dy*dy);
				//System.out.println("penergy = " + penergy);
			}
		}
		this.energy = uenergy + penergy;
		System.out.println("uenergy = " + uenergy + " penergy = " + penergy + " total energy = " + energy);
	}

	double[][] get_belief(int p){
		double[][] r = new double[m][m];
		
		//We will have to think something about this statement -> this isn't coming into action anyway
		if(D.get(p).length == 0 || D.get(p) == null){
			Arrays.fill(r, 0.0);
		}
		else {
			//unary term
			for(int i = 0; i < m; i++)
				r[i] = D.get(p)[i].clone();
		}
		
		//edge indices
		int[] e = edges.get(p);
		
		int n = e.length;
		
		//msgs recieved by p
		double[][][] m = this.msgs.get(p);
		
		//go over all neighbors
		for(int i = 0; i<n; i++){
		//the n here refers to the local variable n	
			for(int x = 0; x < this.m; x++){
				for(int y = 0; y < this.m; y++) {
					//message recieved from e[i] -> adding all 
					r[x][y]+=m[x][y][i];
					//this depends on our original assumption that we had made of the msgs dimensions
				}
			}
		}
		return r;	
	}
	
	/**
	 * This function is the main function to implement belief popagation
	 * @param offsets These are optional i.e. they can be none as well as can have a null value
	 * @param maxit Maximum iteration
	 * @param abstol Absolute tolerance
	 * @param reltol Relative tolerance
	 */
	void solve(int[][] offsets, int maxit, double abstol, double reltol){
		init_msgs();
		//chk msgs after this
		
		this.best_energy = Double.POSITIVE_INFINITY;
		
		//loop variable
		int i;
		
		//List<int[]> best_labels = new ArrayList<int[]>();
		
		for(i = 0; i<this.n; i++){
			this.best_labels.add(new int[] {0, 0});
		}
		
		//int[][] best_labels = new int[n][2];
		//Arrays.fill(best_labels, 0);
		
		double reldiff = Double.POSITIVE_INFINITY;
		
		for(i = 0; i<maxit; i++){
			estimate(offsets);
			//this fn modifies the energy and the best variable
			
			if(i>0 && this.best_energy!=0){
				reldiff = Math.abs(this.energy - this.best_energy)/Math.abs(this.best_energy);
				//System.out.println("Setting the value of reldiff. New reldiff = " + reldiff);
			}

			System.out.println("i = " + i + " energy = " + this.energy + " prev best_energy = " + this.best_energy + " reldiff = " + reldiff);
			//energy, labels = self.estimate(offsets=offsets)
			
			if(this.energy < this.best_energy){
				
				//System.out.println("Checking if the new energy is better than the best_energy.");
				this.best_energy = this.energy;
				
				//think about this
				this.best_labels = this.best;
				
			}
			else{
				//System.out.println("The break occured from the else.");
				break;
			}
			
			if(i > 0 && (Math.abs(this.energy) < abstol || reldiff < reltol)){
				
				break;
				
			}
			
			this.msgs = send_all_msgs(offsets);
			
			System.out.println("End of iteration " + i);
		}
		if(i < maxit -1) {
			if(this.energy>this.best_energy) {
				System.out.println("Terminated because energy started to increase.");
			}
			else {
				System.out.println("Convergence detected.");
			}
		}
		else{
			System.out.println("Maximum number of iterations reached");
		}
		
		
		int shift = (this.m - 1)/2;
		int sz = this.best_labels.size();

		for(int j = 0; j<sz; j++){
			this.best_labels.get(j)[0] = this.best_labels.get(j)[0] - shift;
			this.best_labels.get(j)[1] = this.best_labels.get(j)[1] - shift;
		}
		
	}
}
