FFBNet net;
float[] work;
void setup() {

  for (int i=1; i<20; i++) {
    work=new float[1024];
    work[1]=1f;
    net=new FFBNet(1024, i);
    net.recall(work, work);
    float sum=0;
    for (int j=0; j<work.length; j++) {
      sum+=work[j]*work[j];
    }
    println(i+"   "+sum);  //Check vector length through the (random) net is approximately okay
  }
}

void draw() {
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
        buffer[j]=sc*(b<0f? b*weights[wIdx]:b*weights[wIdx+1]);
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
