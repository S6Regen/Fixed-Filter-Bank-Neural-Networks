FFBNet parent;
FFBNet child;
float[][] inputVecs;
float[][] targetVecs;
float[] work;

int dim=64;
int layerDepth=4;
int piecewise=4;
int mutate=25;
float precision=25f;
int iterations=300;

void setup() {
  work=new float[dim];
  inputVecs=new float[dim][dim];
  targetVecs=new float[dim][dim];
  for (int i=0; i<dim; i++) {
    inputVecs[i][i]=1f;
    for (int j=0; j<dim; j++) {
      targetVecs[i][j]=sin(0.002*(i+dim)*j);
    }
  }
  parent=new FFBNet(dim, layerDepth, piecewise);
  child=new FFBNet(dim, layerDepth, piecewise);
  size(256, 256);
}

void draw() {
  boolean lossType=(frameCount & 1)==0; // odd or even frameCount
  float parentCost=lossType? costL1(parent):costL2(parent);
  for (int iter=0; iter<iterations; iter++) {
    System.arraycopy(parent.parameters, 0, child.parameters, 0, parent.parameters.length);
    for (int i=0; i<mutate; i++) {
      int rIdx=(int)random(0, child.parameters.length);
      float v=child.parameters[rIdx];
      float m=2f*exp(random(-precision, 0f));
      if (random(-1f, 1f)<0f) m=-m;
      m+=v;
      if (m>1f) m=v;
      if (m<-1f) m=v;
      child.parameters[rIdx]=m;
    }
    float childCost=lossType? costL1(child):costL2(child);  
    if (childCost<parentCost) {
      parentCost=childCost;
      float[] t=parent.parameters;
      parent.parameters=child.parameters;
      child.parameters=t;
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

float costL2(FFBNet net) {
  float cost=0f;
  for (int i=0; i<dim; i++) {
    net.recall(work, inputVecs[i]);
    for (int j=0; j<dim; j++) {
      float d=targetVecs[i][j]-work[j];
      cost+=d*d;
    }
  }
  return cost;
}

float costL1(FFBNet net) {
  float cost=0f;
  for (int i=0; i<dim; i++) {
    net.recall(work, inputVecs[i]);
    for (int j=0; j<dim; j++) {
      float d=targetVecs[i][j]-work[j];
      cost+=abs(d);
    }
  }
  return cost;
}

// Fixed Filter Bank Neural Network 
class FFBNet { 
  int vecLen;
  int depth;
  int pieces;
  float sc;
  float[] buffer;
  float[] flips;
  float[] parameters;   
  // inputLen must be 2,4,8,16,32... (int power of 2)
  FFBNet(int inputLen, int netDepth, int piecewise) {
    vecLen=2*inputLen;// double up the input dimension to allow ResNet type behavior etc.
    depth=netDepth;
    pieces=piecewise;
    sc=(float)Math.sqrt(1.0/vecLen); //scaling for 1 WHT
    buffer=new float[vecLen];
    parameters=new float[2*vecLen*depth*pieces+vecLen];
    for (int i=0; i<parameters.length; i++) {
      parameters[i]=1f-2f*(float)Math.random(); // random initialization between -1 and 1
    }
    flips=new float[vecLen];
    for (int i=0; i<vecLen; i++) {
      flips[i]=Math.random()<0.5? -1f:1f;
    }
  }

  void recall(float[] resultVec, float[] inVec) {
    int n=vecLen>>1;  // vecLen/2 Ie. length of inVec
    // sum squared of inVec
    float sumsq=0f;
    for (int i=0; i<n; i++) {
      sumsq+=inVec[i]*inVec[i];
    }
    float adj = 0.25f*sc/ (float) Math.sqrt((sumsq/n) + 1e-20f);
    for (int i=0; i<n; i++) {
      float b=adj*inVec[i];
      buffer[i]=b*flips[i];
      buffer[i+n]=b*flips[i+n];
    }
    whtBuffer();  // initial random projection
    int pIdx=0; // parameter index
    for (int i=0; i<depth; i++) { 
      for (int j=0; j<vecLen; j++) {
        float sum=0f;
        float b=buffer[j];
        for (int k=0; k<pieces; k++) { //piece-wise linear function (but nonlinear overall!)
          sum+=parameters[pIdx++]*absFloat(b-parameters[pIdx++]);
        }
        buffer[j]=sc*sum;
      }
      whtBuffer();
    }
    for (int i=0; i<n; i++) {
      resultVec[i]=buffer[i]*parameters[pIdx++]+buffer[i+n]*parameters[pIdx++];
    }
  }

  float absFloat(float x) {
    return Float.intBitsToFloat(0x7fffffff&Float.floatToRawIntBits(x));
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
