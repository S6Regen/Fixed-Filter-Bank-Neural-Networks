FFBNet parent;
FFBNet child;
float[][] inputVecs;
float[][] targetVecs;
float[] work;

int dim=64;
int layerDepth=32;
int mutate=25;
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
  parent=new FFBNet(dim, layerDepth);
  child=new FFBNet(dim, layerDepth);
  size(256, 256);
}

void draw() {
  // Evolution based optimization
  // www.cs.bham.ac.uk/~jer/papers/ctsgray.pdf
  // with switching between L1 and L2 cost to escape local minimum traps.
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
  int halfVecLen;
  int depth;
  float scaleHStepFn=1.7f/(float)Math.sqrt(2.0);
  float scaleWHT;
  float[] buffer;
  float[] flips;
  float[] parameters;   
  // vecLen must be 2,4,8,16,32... (int power of 2)
  FFBNet(int vecLen, int depth) {
    this.vecLen=vecLen;
    this.depth=depth;
    halfVecLen=vecLen>>1;
    scaleWHT=(float)Math.sqrt(1.0/vecLen); //scaling for 1 WHT 
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
    float sumsq=0f;
    for (int i=0; i<vecLen; i++) {
      sumsq+=inVec[i]*inVec[i];
    }
    // sphere data ie. constant vector length and predetermined pattern random sign flip
    float adj = 1f/(float) Math.sqrt((sumsq/vecLen) + 1e-20f);
    for (int i=0; i<vecLen; i++) {
      resultVec[i]=adj*flips[i]*inVec[i];
    }
    wht(resultVec); // random sign flipping + WHT=random projection
    int pIdx=0; // parameter index 
    for (int i=0; i<depth; i++) {
      for (int j=0; j<vecLen; j++, pIdx+=2) {
        float x=resultVec[j];// nonlinear switch slope according to sign
        float par=x<0f?  parameters[pIdx]:parameters[pIdx+1];
        resultVec[j]=x*par;
      }
      hStep(resultVec);
    }
  }

  //Go through the input data pairwise.
  //Put the sum in the lower half of a new array.
  //Put the difference in the upper half of the new array.
  //Copy back with scaling.  
  void hStep(float[] vec) {
    for (int i=0; i<halfVecLen; i++) {
      float a=vec[i+i];    // access vec pairwise
      float b=vec[i+i+1];  // sequentially
      buffer[i]=a+b;       // in lower half of buffer
      buffer[i+halfVecLen]=a-b;  // in upper part of buffer
    }
    for (int i=0; i<vecLen; i++) {
      vec[i]=scaleHStepFn*buffer[i];
    }
  } 

  // Walsh Hadamard Transform of a vector
  void wht(float[] vec) {
    int i, j, hs=1;
    float a, b;
    while (hs<vecLen) {
      i=0;
      while (i<vecLen) {
        j=i+hs;
        while (i<j) {
          a=vec[i];
          b=vec[i+hs];
          vec[i]=a+b;
          vec[i+hs]=a-b;
          i+=1;
        }
        i+=hs;
      }
      hs+=hs;
    }
    for (int k=0; k<vecLen; k++) {
      vec[k]=scaleWHT*vec[k];
    }
  }
}
