import java.util.Arrays;

class ThreadingSendMsgs extends Thread {
    
	//class variables
	int q0, q1;
	int[][] offsets;
	
	//BPGraph obj -> will be useful to update the msg array
	BPGraph bpThread;
	
	//this list would be updated by each thread -> may need synchronization (lets see)
	
	
	//ThreadingSendMsgs constructor
	public ThreadingSendMsgs(int startRange, int endRange, int[][] offsets, BPGraph bpThread){
		this.q0 = startRange;
		this.q1 = endRange;
		this.offsets = offsets;
		this.bpThread = bpThread;
	}
	
	/*
	 * /*
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

	 */
	
	/**
	* the main body of the thread
	*/
	@Override
	public void run() {
		//This object would be used to call the relevant fns from the BPGraph class
		
		//BPGraph bpThreading = new BPGraph();
		System.out.println("In run........");
		for(int q = q0; q<q1; q++){
			int[] e = bpThread.edges.get(q);
			int ne = e.length;
			
			double[][][] mfq = new double[bpThread.m][bpThread.m][ne];
			double[][] buffer = new double[bpThread.m][bpThread.m];
			System.out.println("1....mfq.length = "+mfq[0][0].length);
			for(int i = 0; i<ne; i++){
				int p = e[i];
				buffer = bpThread.calc_msg(p, q, bpThread.weights.get(q)[i], offsets);
				for (int ig = 0; ig < bpThread.m; ig++) {
					for (int jg = 0; jg < bpThread.m; jg++) {
						mfq[ig][jg][i] = buffer[ig][jg];
					}
			    }
			}
			System.out.println("mfq.length = "+mfq[0][0].length);
			bpThread.tempThread.set(q, mfq);
			//compoundThing.add(index, mfq);
		}
	}
}