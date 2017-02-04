
/**
 * 
 */
import org.junit.Ignore;
import org.junit.Test;

/**
 * 
 * @class Test_Beliefpropogation.java
 * @category Image Registration
 * @author Ardhendu Shekhar Tripathi
 * @author Atin Mathur
 * @author Jiri Borovec
 * @version 1.01
 * @date 19/06/2013
 * @brief This class tests the main belief propogation program ->
 *        BeliefPropogation.java
 * 
 */
public class BPGraphTest {

	@Test
	public void try_BPGraph_offset() {
		//MyTools.printTitle("Belief Propogation");

		BPGraph bpGrph = new BPGraph();

		// m is an input parameter to the function make_unary_potential
		int m = 5;
		
		//correct
		int[][] offsets = { { 1, 0 }, { 0, 0 }, { 0, 0 } };

		bpGrph.D.add(makeUnaryPotential(m, -3 - offsets[0][0], -offsets[1][0]));
		bpGrph.D.add(makeUnaryPotential(m, 1, 0));
		bpGrph.D.add(makeUnaryPotential(m, 0, 2));

		bpGrph.edges.add(new int[] { 1, 2 });
		bpGrph.edges.add(new int[] { 0, 2 });
		bpGrph.edges.add(new int[] { 0, 1 });

		bpGrph.weights.add(new double[] { 0.3, 0.3 });
		bpGrph.weights.add(new double[] { 0.3, 0.6 });
		bpGrph.weights.add(new double[] { 0.3, 0.6 });

		long startTime = System.currentTimeMillis();
		bpGrph.solve(offsets, 100, 1e-3, 1e-3);
		long estimTime = System.currentTimeMillis() - startTime;
		System.out.println("Belief Propagation took "
				+ Long.toString(estimTime) + "ms.");

		System.out.println("best_energy/e = " + bpGrph.best_energy);
		System.out
				.println("best_label after termination ------------------------------------ ");

		for (int i = 0; i < bpGrph.best_labels.size(); i++) {
			for (int j = 0; j < bpGrph.best_labels.get(i).length; j++) {
				System.out.print(bpGrph.best_labels.get(i)[j] + "   ");
			}
			System.out.println();
		}

	}

	@Ignore
	double[][] makeUnaryPotential(int m, int ix, int iy) {
		int n = 2 * m + 1;

		int[][] xx = new int[n][n];
		int[][] yy = new int[n][n];

		int[] baseArr = new int[n];

		double[][] out = new double[n][n];

		for (int i = -1 * m; i <= m; i++) {
			baseArr[i + m] = i;
		}

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				xx[j][i] = baseArr[j];
				yy[i][j] = baseArr[j];
			}
		}

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				out[i][j] = Math.pow(xx[i][j] - ix, 2.0)
						+ Math.pow(yy[i][j] - iy, 2.0);
			}
		}
		return out;
	}

	/**
	 * This function does the same work as the __main__ in the
	 * test_beliefpropagation.
	 * 
	 * @param srgs
	 *            -> an optional string type variable
	 */
	public static void main(String[] args) {

		BPGraphTest tstBelief = new BPGraphTest();

		tstBelief.try_BPGraph_offset();

	}
}