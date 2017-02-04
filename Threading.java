import java.util.List;

//package sc.fiji.CMP_BIA.tools;

//import ij.Prefs;

public class Threading {


	/**
	 * Create a List of Threads -> as large as the number of processors available.
	 */
	public static int nbAvailableThread() {
		//return Prefs.getThreads();
		return Runtime.getRuntime().availableProcessors();
	}
    
	
	/**
	 * Start all given threads and wait on each of them until all are done.
	 */
	public static void startAndJoin(List<ThreadingSendMsgs> threads) {
     
		//starting the thresds
		for (int ithread = 0; ithread < threads.size(); ++ithread) {
			threads.get(ithread).setPriority(Thread.NORM_PRIORITY);
			threads.get(ithread).start();
		}
     
		//joining the threads
		try {
			for (int ithread = 0; ithread < threads.size(); ++ithread)
				threads.get(ithread).join();
		} catch (InterruptedException ie) {
			throw new RuntimeException(ie);
		}

	}

}