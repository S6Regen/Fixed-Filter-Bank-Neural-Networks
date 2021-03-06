FFBNet parent; //<>// //<>// //<>// //<>// //<>// //<>// //<>//
FFBNet child;
float[][] inputVecs;
float[][] targetVecs;
float[] work;

int dim=64;
int netWidth=256;// the net width is set high (4 by input dimension)
int layerDepth=4;// as an example but really depth is perfered
int mutate=25;   // for speedy evolution
float precision=25f;
int iterations=1000;

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
  parent=new FFBNet(netWidth, layerDepth);
  child=new FFBNet(netWidth, layerDepth);
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
  float sc;
  float[] buffer;
  float[] flips;
  float[] parameters;   
  // netWidth must be 2,4,8,16,32... (int power of 2)
  FFBNet(int netWidth, int netDepth) {
    vecLen=netWidth;
    depth=netDepth;
    sc=(float)Math.sqrt(1.0/vecLen); //scaling for 1 WHT
    buffer=new float[vecLen]; 
    flips=new float[vecLen]; // Fixed pattern of random sign flips (inc. scaling factor)
    parameters=new float[2*depth*vecLen]; //for parameterized nonlinear functions
    for (int i=0; i<parameters.length; i++) {
      parameters[i]=1f-2f*(float)Math.random(); // random initialization between -1 and 1
    }
    int h=123456; // LCG seed
    for (int i=0; i<vecLen; i++) {
      h*=0x9E3779B9;  // LCG pseudorandom generator
      h+=0x6A09E667;
      flips[i]=h<0? -1f:1f;
    }
  }

  void recall(float[] resultVec, float[] inVec) {
    int n=inVec.length;
    // sum squared of inVec
    float sumsq=0f;
    for (int i=0; i<n; i++) {
      sumsq+=inVec[i]*inVec[i];
    }
    // sphering adjustment value with scaling for final WHT
    float adj = sc/ (float) Math.sqrt((sumsq/n) + 1e-20f);
    for (int i=0, j=0; i<vecLen; i++) {
      buffer[i]=adj*flips[i]*inVec[j++];
      if (j==n) j=0;
    }
    int pIdx=0; // parameter index
    float scWP=1.7f*sc; // scaling for 1 WHT and functions 
    for (int i=0; i<depth; i++) { 
      whtBuffer();  
      for (int j=0; j<vecLen; j++, pIdx+=2) {
        float b=buffer[j];
        // switch slope at zero nonlinear function
        // with scaling factor sc for WHTs, nonlinear function
        float par=b<0f?  parameters[pIdx]:parameters[pIdx+1];
        buffer[j]=scWP*b*par;
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
