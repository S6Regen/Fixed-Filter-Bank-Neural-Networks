FFBNet net; //<>// //<>//
float[] work;
void setup() {

  for (int i=1; i<20; i++) {
    work=new float[2048];
    work[1]=1f;
    net=new FFBNet(2048, i);
    net.recall(work, work);
    float sum=0;
    for (int j=0; j<work.length; j++) {
      sum+=work[j]*work[j];
    }
    println(i+"   "+sum);
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
  FFBNet(int inputLen, int depth) {
    this.vecLen=2*inputLen;// double up the input dimension to allow ResNet type behavior etc.
    this.depth=depth;
    double s=Math.sqrt(1.0/vecLen); //scaling for 1 WHT
    sc=(float)(Math.sqrt(2.0)*s*Math.pow(s, 1.0/depth)); //scaling for switch slope function, WHT and final WHT
    buffer=new float[vecLen];
    weights=new float[2*depth*vecLen];
    for (int i=0; i<weights.length; i++) {
      weights[i]=1f-2f*(float)Math.random(); // random initialization between -1 and 1
    }
  }
 //<>//
  void recall(float[] resultVec, float[] inVec) {
    adjust(inVec); // copy inVec to buffer (lower and upper halves) and adjust vector length to a constant value
    signFlip(); // Sign flip the elements of buffer according to a random but fixed pattern (same every time.)
    int wIdx=0;
    for (int i=0; i<depth; i++) { //This mixes up the frequency spectrum to get a relatively well spread out response through the filter bank
      wht();	// Fixed filter bank (Walsh Hadamard transform).  Fast fixed weighted sums! O(nlog(n)).
      for (int j=0; j<vecLen; j++, wIdx+=2) {
        float b=buffer[j];
        buffer[j]=sc*(b<0f? b*weights[wIdx]:b*weights[wIdx+1]);
      }
    }
    wht();  // Final WHT  // Smooths out switching noise from nonlinear functions etc.
    System.arraycopy(buffer, 0, resultVec, 0, resultVec.length);
  }

  // Switch slope at zero nonlinear function
  // Check the sign of the elements of buffer[] and multipy element by one slope or another depending on that.
  // Also include a scaling operation both for the function itself and the WHTs
  // To keep the vector length through the network resonable 
  void switchSlope() {
  }  

  // Walsh Hadamard Transform of buffer
  // No scaling appled (vector length after transform is greater)
  void wht() {
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

  // Random sign flip of the elements of buffer according to a fixed pattern using hashing
  void signFlip() {
    int h=123456;
    for (int i=0; i<vecLen; i++) {
      h*=0x9E3779B9;
      h+=0x6A09E667;
      // Faster than -  if(h<0) buffer[i]=-buffer[i];
      buffer[i]=Float.intBitsToFloat((h&0x80000000)^Float.floatToRawIntBits(buffer[i]));
    }
  }

  // Copy the input vector to the lower half of buffer while
  // adjusting the length of the vector to a fixed value (ie. sphere it)
  // copy to the upper half as well
  void adjust(float[] inVec) {
    float sum = 0f;
    int n=vecLen>>1;  // ie. vecLen/2
    for (int i = 0; i < n; i++) {
      sum += inVec[i] * inVec[i];
    }
    float adj = 1f/ (float) Math.sqrt((sum/n) + 1e-20f);
    for (int i=0; i<n; i++) {
      float v=adj*inVec[i];
      buffer[i]=v;   // to lower half
      buffer[i+n]=v; // to upper half
    }
  }
}
