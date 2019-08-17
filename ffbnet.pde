// Fixed filter bank neural networks in the processing language (Java+easy graphics etc.)
// www.processing.org
// Rather than use n adjustable weighted sums per layer which take about n squared operations
// use n fixed weighted sums that can be computed efficiently using a transform (eg. FFT, WHT.)
// However something must bend, you make the nonlinear functions individually adjustable by
// parameterizing them.
// It's a swap around from a conventional neural network with has adjustable weighted sum filters
// and a fixed nonlinear function, to a fixed filter bank and adjustable nonlinear functions.
// The computational cost is cut from n squared operations per layer to n.log(n) operations.
FFBNet parent;
FFBNet child;
float parentCost;
float[][] inputVecs;
float[][] targetVecs;
float[] work;
int dim=64;
int layerDepth=5;
int mutate=25;
float precision=25f;

void setup() {
  work=new float[dim];
  inputVecs=new float[dim][];
  targetVecs=new float[dim][];
  for (int i=0; i<dim; i++) {
    inputVecs[i]=new float[dim];
    targetVecs[i]=new float[dim];
    inputVecs[i][i]=1f;
    for (int j=0; j<dim; j++) {
      targetVecs[i][j]=sin(0.002*(i+dim)*j);
    }
  }
  parentCost=Float.POSITIVE_INFINITY;
  parent=new FFBNet(dim, layerDepth);
  child=new FFBNet(dim, layerDepth);
  size(256, 256);
  frameRate(100);
}

void draw() {
  for (int iter=0; iter<1000; iter++) {
    System.arraycopy(parent.weights, 0, child.weights, 0, parent.weights.length);
    for (int i=0; i<mutate; i++) {
      int rIdx=(int)random(0, child.weights.length);
      float v=child.weights[rIdx];
      float m=2f*exp(random(-precision, 0f));
      if (random(-1f, 1f)<0f) m=-m;
      m+=v;
      if (m>1f) m=v;
      if (m<-1f) m=v;
      child.weights[rIdx]=m;
    }
    float childCost=0f;
    for (int i=0; i<dim; i++) {
      child.recall(work, inputVecs[i]);
      for (int j=0; j<dim; j++) {
        float d=targetVecs[i][j]-work[j];
        childCost+=d*d;
      }
    }
    if (childCost<parentCost) {
      parentCost=childCost;
      float[] t=parent.weights;
      parent.weights=child.weights;
      child.weights=t;
    }
  }
  int ex=frameCount%dim;
  java.util.Arrays.fill(work, 0f);
  work[ex]=1f;
  parent.recall(work, work);
  background(0); // clear screen
  for (int i=0; i<dim; i++) {
    fill(255, 0, 127); // draw color
    ellipse(i*4, 127+ 120*targetVecs[ex][i], 5, 5);
    fill(127, 0, 255); // draw color
    ellipse(i*4, 127+120*work[i], 5, 5);
  }
}  

// Fixed Filter Bank Neural Network 
class FFBNet { 
  int vecLen;
  int depth;
  float sc;
  float[] buffer;
  float[] weights;

  //   
  // inputLen must be 2,4,8,16,32... (int power of 2)
  FFBNet(int inputLen, int netDepth) {
    vecLen=2*inputLen;// double up the input dimension to allow ResNet type behavior etc.
    depth=netDepth;
    double s=Math.sqrt(1.0/vecLen); //scaling for 1 WHT
    sc=(float)(1.7*s*Math.pow(s, 1.0/depth)); // scaling for switch slope function, WHT and final WHT
    buffer=new float[vecLen];                 // 1.7=magic number obtained by trial and error
    weights=new float[2*depth*vecLen];
    for (int i=0; i<weights.length; i++) {
      weights[i]=1f-2f*(float)Math.random(); // random initialization between -1 and 1
    }
  }

  void recall(float[] resultVec, float[] inVec) {
    int n=vecLen>>1;  // vecLen/2 Ie. length of inVec
    // sum squared of inVec
    float sumsq=0f;
    for (int i=0; i<n; i++) {
      sumsq+=inVec[i]*inVec[i];
    }
    // sphering adjustment value
    float adj = 1f/ (float) Math.sqrt((sumsq/n) + 1e-20f);
    // prepare buffer. copy inVec to upper and lower half
    // adjust vector length to a constant value (sphering)
    // apply fixed random pattern of sign flips to
    // spread out the frequency spectrum
    int h=123456; // LCG seed
    for (int i=0; i<n; i++) {
      h*=0x9E3779B9;  // LCG pseudorandom generator
      h+=0x6A09E667;
      float v=adj*inVec[i];
      // assign to buffer (high and low) with random sign flip
      int iv=Float.floatToRawIntBits(v);
      buffer[i]=Float.intBitsToFloat((h&0x80000000)^iv); //msb of h
      buffer[i+n]=Float.intBitsToFloat(((h+h)&0x80000000)^iv); // second msb of h
    }  
    int wIdx=0; // weight index
    for (int i=0; i<depth; i++) { 
      whtBuffer();	
      for (int j=0; j<vecLen; j++, wIdx+=2) {
        float b=buffer[j];
        // switch slope at zero nonlinear function
        // with scaling factor sc for WHTs, nonlinear function
        float wt=b<0f?  weights[wIdx]:weights[wIdx+1];
        buffer[j]=sc*b*wt;
      }
    }
    whtBuffer();  // final WHT, smooths out switching noise from nonlinear functions etc.
    System.arraycopy(buffer, 0, resultVec, 0, resultVec.length);
  }

  // Walsh Hadamard Transform of buffer
  // No scaling appled (vector length after transform is greater)
  // Acts as a fixed filter bank of non-adjustable weighted sums.
  // with time complexity O(nlog(n))
  void whtBuffer() {
    int i, j, hs=1;
    float a, b;
    while (hs<vecLen) {
      i=0;
      while (i<vecLen) {
        j=i+hs;
        while (i<j) {
          a=buffer[i];
          b=buffer[i+hs];
          buffer[i]=a+b;
          buffer[i+hs]=a-b;
          i+=1;
        }
        i+=hs;
      }
      hs+=hs;
    }
  }
}
