/* Translated from segregtools.pyx */
import java.util.*;

public class segregtools {
	
	/* class variables */
	double[][] P={};
	int totpix=0;
	
	public int[][] superpixel_overlaps(int[][] sl1, int[][] sl2, int[] nsize, int[] shift1, 
		int[] shift2) {
		/* additionally also pass the number of rows and columns in the two 
		 * matrices sl1 and sl2 -> rsl1, csl1, rsl2, csl2 The number of rows 
		 * and columns would come from the class TwoImages -> using the BufferedImage.
		*/
		int[][] o = {};
		int i, j;
		
		int nx1 = sl1.length;	//rsl1 
		int ny1 = sl1[0].length;	//csl1
		int nx2 = sl2.length;	//rsl2
		int ny2 = sl2[0].length;	//csl2
		//double il;
		
		Hashtable<Integer, Integer> h = new Hashtable<Integer, Integer>();
		
		/*I don't understand from where does this 'l' come. 
		It is not the letter 'l' but one - '1'*/
		int maxlb = 1 << 16;
		int ll1, ll2, c, i1, j1, i2, j2;
		
		/*It should be possible to iterate over a small range but I do not think
		it makes any difference*/
		for(i = 0; i<nsize[0]; i++){
			for(j = 0; j<nsize[1]; j++){
				
				i1 = i - shift1[0];
				j1 = j - shift1[1];
				if(i1>=0 && i1<nx1 && j1>=0 && j1<=ny1){
					ll1 = sl1[i1][j1];
					if(ll1 == -1){
						continue;
					}
					assert(ll1<maxlb && ll1>0);
					i2 = i - shift2[0]; 
					j2 = j - shift1[1];
					
					if(i2>=0 && i2<nx2 && j2>=0 && j2<ny2) {
						
						ll2 = sl2[i2][j2];
						if(ll2 == -1)
							continue;
						assert(ll2<maxlb && ll2>=0);
						
						c = (ll1 << 16) | ll2;
						
						/*h[c]+ = 1; This needs to be implemented in the form of a 
						hashmap -> Think*/
						if(!h.containsKey(new Integer(c))){
							h.put(new Integer(c), new Integer(1));
						}
						else{
							Integer n = h.get(new Integer(c));
							h.put(new Integer(c), new Integer(++n));
						}
											
					}
					
				}
			}
		}
		int numel = h.size();
		o = new int[numel][3];
		
		Iterator<Map.Entry<Integer, Integer>> it = h.entrySet().iterator();
		
		int mask = (1 << 16) - 1;
		int val;
		
		i = 0;
		//iterate through the HashMap
		while (it.hasNext()) {
			
			Map.Entry<Integer, Integer> entry = it.next();
			val = entry.getValue();
			o[i][0] = val >> 16;
			o[i][1] = val & mask;
			o[i][2] = entry.getValue();
			i+= 1;
			// Remove entry if key is null or equals 0.
		}
		return o;
	}
	
	@SuppressWarnings("unused")
	public Set<Integer> close_superpixels(int[][] labels, int x, int y, int r, int rsl, int csl) {
		
		Set<Integer> s = new HashSet<Integer>();
		//set of type integers
		
		int i, j;
		int nx = rsl;
		int ny = csl;
		
		int i0 = Math.max(0,  x-r);
		int i1 = Math.min(nx - 1, x + r);
		
		int j0 = Math.max(0, y - r);
		int j1 = Math.min(ny - 1, y + r);
		int dx, dy;
		int r2 = r*r;
		
		for(i = i0; i<=i1; i++){
			for(j = j0; j<=j1; j++){
				dx = i - x;
				dy = j - y;
				if(dx*dx + dy*dy < r2){
					boolean chk = s.add(labels[i][j]);
					//can put a check here to see if the current label was added or not.					
				}
			}
		}
		return s;
	}

	public static int kernelize_size(int n, double order) {
		if (order == 1.0)
			return n;
		else if (order == 2.0) 
			return n*(n+1)/2;
		else if (order == 3.0)
			return n*(n+1)*(n+2)/6;
		else
			return -1;
	}

	@SuppressWarnings("unused")
	public static void fast_kernelize(double[] v, double order, double[] y) {
		/* Like kernelize but stores the result into an
		 * output y to avoid allocation
		 */
		int n=v.length;
		int i, j, k, l, m;
		
		if(order==1.0) {
			for(i=0;i<n;i++) {
				y[i] = v[i];
			}
		}	
		else if(order==2.0){
			m=n*(n+1)/2;
			l=0;
			for(i=0; i<n; i++) {
				for(j=i;j<n;j++) {
					y[l] = v[i]*v[j];
					l++;
				}
			}
			
		}
		else if(order==3.0) {
			m=n*(n+1)*(n+2)/6;
			l=0;
			for(i=0; i<n; i++) {
				for(j=i;j<n;j++) {
					for(k=j;k<n;k++) {
						y[l] = v[i]*v[j]*v[k];
						l++;
					}
				}
			}
		}
		
	}
	
	public static double[] kernelize(double[] v, double order) {
		/* Given a vector, apply a polynomial "kernel" of a given 
		 * order to it. We assume that v[0]=1. If order=1, the 
		 * vector returned as is, not copies */
		
		assert v[0]==1.0;
		
		double[] y = {};
		int n = v.length;
		int i, j, k, l, m;
		
		if(order==1.0) {
			return v;
		}
		else if(order==2.0) {
			m=n*(n+1)/2;
			y=new double[m];
			Arrays.fill(y, 0);
			l=0;
			for(i=0;i<n;i++) {
				for(j=i;j<n;j++) {
					y[l]=v[i]*v[j];
					l+=1;
				}
			}
			assert (l==m);
			return y;
		}
		else if(order==3.0) {
			m=n*(n+1)*(n+2)/6;
			y=new double[m];
			Arrays.fill(y, 0);
			l=0;
			for(i=0;i<n;i++) {
				for(j=i;j<n;j++) {
					for(k=j;k<n;k++) {
						y[l]=v[i]*v[j]*v[k];
						l+=1;
					}
				}
			}
			assert (l==m);
			return y;
		}
		else
			System.err.println("kernelize : Unsupported Order");
		return null;
	}
	
	public double[][] mean_over_superpixels(int[][] labels, double[][][] fp, int nsuper) {
		/* Each superpixel is assigned a mean of the features calculated for its
		 * pixels. An element is added in front and set to 1. */
		int m=labels.length;
		int n=labels[0].length;
		assert nsuper>0;
		int nf=fp[0][0].length;
		int[] c = new int[nsuper];
		double[][] features = new double[nsuper][nf+1];
		int i, j, k, l;
		
		assert fp.length==m;
		assert fp[0].length==n;
		
		for(i=0;i<nsuper;i++) {
			features[i][0]=1;
		}
		
		for(i=0;i<m;i++) {
			for(j=0;j<n;j++) {
				k=labels[i][j];
				assert k>=0 && k<nsuper;
				for(l=0;l<nf;l++) {
					features[k][l+1]+=fp[i][j][l];
				}
				c[k]+=1;
			}
		}
		
		for(i=0;i<nsuper;i++) {
			for(l=0;l<nf;l++) {
				features[i][l+1]/=c[i];
			}
		}
		
		return features;
	}
	
	public static int[][] find_boundary_points(int[][] sl, int[] l) {
	/* Given superpixel labels sl (coordinate->superpixel) and a segmentation
	 * (superpixel->class), return a n X 2 array of points containing coordinates
	 * of boundary points, i.e, points where at least three classes coincide */
	int nx=sl.length;
	int ny=sl[0].length;
	
	int i, j, ll1, ll2, ll3, ll4, l1, l2, l3, l4;
	int[][] o; //output
	Vector<Integer> v = new Vector<Integer>();
	
	for(i=0;i<nx-1;i++) {
		for(j=0;j<ny-1;j++) {
			ll1=sl[i][j];
			ll2=sl[i][j+1];
			ll3=sl[i+1][j];
			ll4=sl[i+1][j+1];
			int chk=0;
			chk+=(ll1!=ll2)?1:0;chk+=(ll1!=ll3)?1:0;chk+=(ll1!=ll4)?1:0;
			chk+=(ll2!=ll3)?1:0;chk+=(ll2!=ll4)?1:0;chk+=(ll3!=ll4)?1:0;
			if(chk>=5) {
				l1=l[ll1];l2=l[ll2];l3=l[ll3];l4=l[ll4];
				chk=0;
				chk+=(l1!=l2)?1:0;chk+=(l1!=ll3)?1:0;chk+=(l1!=l4)?1:0;
				chk+=(l2!=l3)?1:0;chk+=(l2!=l4)?1:0;chk+=(l3!=l4)?1:0;
				if(chk>=1) {
					v.add(i);
					v.add(j);
				}
			}
		}
	}
	assert v.size()%2==0;
	o=new int[v.size()/2][2];
	for(i=0;i<o.length;i++) {
		o[i][0]=v.elementAt(2*i);
		o[i][1]=v.elementAt(2*i+1);
	}
	return o;	
	}
	
	public void assemble_probability_matrix(int ncls, int[][] overlaps,
		double[][] z0, double[][] z1) {
		/* Helper function for TwoImages:mut_inf 
		 It is faster to normalize after */
		P = new double[ncls][ncls];
		int i, j, k, q0, q1, q2;
		int n=overlaps.length;
		int[] q;
		int fdim=z0[0].length;
		
		for(i=0;i<n;i++) {
			q=overlaps[i];
			q0=q[0]; q1=q[1]; q2=q[2];
			for(j=0;j<fdim;j++) {
				for(k=0;k<fdim;k++) {
					P[j][k]+=q2*z0[q0][j]*z1[q1][k];
				}
			}
			totpix+=q2;
		}
		for(i=0;i<P.length;i++) {
			for(j=0;j<P[0].length;j++) {
				P[i][j] = P[i][j]/totpix;
			}
		}
	}

	public void mutual_inf_gradient(int ncls, int[][] overlaps, double[][]z0, double[][] z1, 
			double[][] f0, double[][] f1, double[][] P, double[][] gradP, double[][][] grad,
			double totpix, double kernel_order) {
		/* Helper function for TwoImages.mutual_inf. grad is modified inplace*/
		int n = overlaps.length;
		int[] q;
		double w, ww;
		int i, k, l, kk, ll, j, q2;
		int fdim = f0[0].length;
		double[] v0 = new double[kernelize_size(fdim, kernel_order)];
		double[] v1 = new double[kernelize_size(fdim, kernel_order)];
		double[] z0q0, z1q1;
			
		/* Prof. Jan Kybic commented this part
		double max1, max2, thr0, thr1;
		//Calculate the maximum probability for each feature vector
		double alpha=0.1; // how many to consider
		double[] mp0={},mp1={}, mp={};
		//assign to mp0 the max(z0,1)
		//assign to mp1 the max(z1,1)	
		for(i=0;i<z0.length;i++) {
			max1=z0[i][0];
			max2=z1[i][0];
			for(k=1;k<z0[0].length;k++) {
				if(max1<z0[i][k])
					max1=z0[i][k];
				if(max2<z1[i][k])
					max2=z1[i][k];
			}
			mp0[i]=max1;
			mp1[i]=max2;
		}
		mp=mp0;
		Arrays.sort(mp);
		thr0=mp[(int) Math.ceil(0.1*z0.length)];
		mp=mp1;
		Arrays.sort(mp);
		thr1=mp[(int) Math.ceil(0.1*z1.length)];
		*/
		for(i=0;i<n;i++) {
			q=overlaps[i];
			fast_kernelize(f0[q[0]], kernel_order, v0);
			fast_kernelize(f1[q[1]], kernel_order, v1);
			z0q0=z0[q[0]];
			z1q1=z0[q[1]];
			q2=q[2];
			for(k=0;k<ncls;k++) {
				for(l=0;l<ncls;l++) {
					w=q2*gradP[k][l]*z1q1[l]/totpix;
					for(kk=0;kk<ncls;kk++) {
						int chk = k==kk?1:0;
						ww=w*z0q0[k]*(chk-z0q0[kk]);
						for(j=0;j<fdim;j++) {
							grad[0][kk][j]+=ww*v0[j];
						}
					}
					w=q[2]*gradP[k][l]*z0q0[k]/totpix;
					for(ll=0;ll<ncls;ll++) {
						int chk1 = l==ll?1:0;
						ww=w*z1q1[l]*(chk1-z1q1[ll]);
						for(j=0;j<fdim;j++) {
							grad[l][ll][j]+=ww*v1[j];
						}
					}
				}
			}
		}
		
	}
	
	public static int[][] distribute_labels_to_pixels(int[] classes, int[][] labels) {
		/* Each pixel is assigned a value of the corresponding superpixel feature */
		int m=labels.length;
		int n=labels[0].length;
		int nsuper=classes.length;
		int[][] o = new int[m][n];
		int i, j, k;
		for(i=0;i<m;i++) {
			for(j=0;j<n;j++) {
				k=labels[i][j];
				if(k==-1) {
					o[i][j]=-1;
				} 
				else {
					assert k>=0 && k<nsuper;
					o[i][j]=classes[k];
				}	
			}
		}
		return o;
	}
	
	public static int[] superpixel_sizes(int[][] labels, int nsuper) {
		/* Count the size of each superpixel in pixels.*/
		int m=labels.length;
		int n=labels[0].length;
		assert nsuper>0;
		int[] c = new int[nsuper];
		
		int i, j, k;
		
		for(i=0;i<m;i++) {
			for(j=0;j<n;j++) {
				k=labels[i][j];
				assert k>=0 && k<nsuper;
				c[k]+=1;
			}
		}
		return c;
	}
	
	public static double[][][] overlay_classes(int[] y1, int[] y2, int[][] l1, int[][] l2,
			int[] nsize, int[] shift1, int[] shift2, double[][] classes) {
		double[][][] o = new double[nsize[0]][nsize[1]][3];
		int ncls=classes.length;
		int i, j, l, c1, c2, i1, j1, i2, j2;
		int nl1 = y1.length;
		int nl2 = y2.length;
		int ll1, ll2;
		int nx1 = l1.length;
		int ny1 = l1[0].length;
		int nx2 = l2.length;
		int ny2 = l2[0].length;
		
		for(i=0;i<nsize[0];i++) {
			for(j=0;j<nsize[1];j++) {
				i1=i-shift1[0];
				j1=j-shift1[1];
				if(i1>=0 && i1<nx1 && j1>=0 && j1<ny1) {
					ll1=l1[i1][j1];
					if(ll1==-1)
						c1=0;
					else {
						assert ll1>=0 && ll1<nl1;
						c1=y1[ll1];
					}
					assert c1>=0 && c1<ncls;
					for(l=0;l<3;l++) {
						o[i][j][l]=classes[c1][l]/2;
					}
				}
				i2=i-shift2[0];
				j2=j-shift2[1];
				if(i2>=0 && i2<nx2 && j2>=0 && j2<ny2) {
					ll2=l2[i2][j2];
					if(ll2==-1)
						c2=0;
					else {
						assert ll2>=0 && ll2<nl2;
						c2=y1[ll2];
					}
					assert c2>=0 && c2<ncls;
					for(l=0;l<3;l++) {
						o[i][j][l]=classes[c2][l]/2;
					}
				}
			}
		}
		return o;
	}
	
	public static void overlay_labels_over_image(double[][] img, int[] lblr, int[] l, double[][] cols, float opacity) {
		/* Helper for show_classes labels.
		 * img is flattened and will be overwritten*/
		assert img[0].length==3 && cols[0].length==3;
		int ncols=cols.length;
		int npix=img.length;
		assert lblr.length==npix;
		int i, li, q;
		double[] col;
		int nsuper = l.length;
		
		for(i=0;i<npix;i++) {
			li=lblr[i];
			assert li<nsuper;
			li=l[li];
			if(li>=0) {
				assert li<ncols;
				col=cols[li];
				for(q=0;q<3;q++) {
					img[i][q]=(col[q]*opacity+img[i][q]*(1-opacity));
				}
			}
		}
	}
	
	public static double[][] q_envelope2D(double c, double[][] r) {
		double[][] rout=new double[r.length][r[0].length];
		rout=q_envelope2D_out(c,r,rout);
		return rout;
	}
	
	public static double[][] q_envelope2D_out(double c, double[][] h, double[][] D) {
		/* Apply q_envelope in both directions */
		int n=h.length;
		int m=h[0].length;
		int i, j;
		double[] e, f;
		e=new double[n];
		f=new double[n];
		for(i=0;i<n;i++) {
			D[i]=q_envelope_out(c,h[i],D[i]);
		}	
		for(i=0;i<m;i++) {
			for(j=0;j<n;j++) {
				e[j]=D[j][i];
			}
			f=q_envelope_out(c,e,f);
			for(j=0;j<n;j++) {
				D[j][i]=f[j];
			}
		}
		return D;
	}
	
	public static double[] q_envelope_out(double c, double[] h, double[] D) {
		/* Given c and h[0..n-1]
		 * Calculate for each q in [0...n-1],
		 * min_p (c (q-p)**2 +h[p] ) using the linear F&H algorithm
		 * */
		int n=h.length;
		int q;
		int vj=0; // To make Cython compiler happy
		double s=0.0, den;
		int j=0;
		// Treat the special case
		if(c==0) {
			Arrays.fill(D, minv(h));
			return D;
		}
		
		double eps=1e-6;
		double inf=1e308;
		int[] v = new int[n];
		double[] z = new double[n+1];
		v[0]=0;
				
		z[0]=-1*inf;
		z[1]=inf;
		for(q=1;q<n;q++) {
			while(true) {
				vj=v[j];
				den=2*c*(q-vj);
				if(den!=0) {
					s=((h[q]+c*q*q)-(h[vj]+c*vj*vj))/den;
				}
				else {
					s=inf;
				}
				if(s<=z[j] && j>0) {
					j-=1;
				}
				else {
					j+=1;
					v[j]=q;
					z[j]=s;
					z[j+1]=inf;
					break;
				}
			}
		}
		j=0;
		for(q=0;q<n;q++) {
			while(z[j+1]-eps<q) {
				j+=1;
			}
			D[q]=c*Math.pow((q-v[j]),2)+h[v[j]];
		}
		return D;
	}
	
	
	public static double minv(double[] v) {
		/* Find a minimum of a vector */
		int i;
		double z;
		double r=1e308; // A big number
		for(i=0;i<v.length;i++) {
			z=v[i];
			if(z<r)
				r=z;
		}
		return r;
	}
}