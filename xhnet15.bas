#include "wht.bi"  ' Fast Walsh Hadmard Transform with no scaling factor  FFHT library
#include "vecops4.bas"
#include "rnd256.bas"
#include "file.bi"
type xhnet
	veclen as ulong
	depth as ulong
	ln2 as ulong
	weights(any) as single
	workA(any) as single
	declare sub init(veclen as ulong,depth as ulong)
	declare sub recall(result as single ptr,inVec as single ptr)
	declare sub memsize()
	declare function load(filename as string) as integer
	declare function save(filename as string) as integer
end type

sub xhnet.init(vecsize as ulong,depth as ulong)
	this.veclen=2*vecsize  ' process at double the size of the input/output vector to allow
	this.depth=depth	   ' ResNet like behavior as a possibility
	this.ln2=log(veclen)/log(2)
	memsize()
	dim as rnd256 r
	r.init()
	for i as ulong=0 to ubound(weights)
		weights(i)=r.nextsinglesym()
	next
end sub

sub xhnet.memsize()
	redim workA(veclen-1)
	redim weights(2*veclen*depth-1)
end sub

sub xhnet.recall(result as single ptr,inVec as single ptr)
	dim as single ptr wts=@weights(0),wa=@workA(0)
	dim as single sc=sqr(2.0)/sqr(veclen) 'scaling factor for WHT and switch slope at zero function
	dim as ulong hvec=veclen shr 1
	adjust(wa,inVec,sc*sc,hvec)
	copy(wa+hvec,wa,hvec)
	signflip(wa,0,veclen) 'whiten the input a little as seems advisable with pseudorandom sign
	for i as ulong=1 to depth 'flipping based on a hash 
		fht_float(wa,ln2) 'raw Walsh Hadamard transform, no scaling
		switch(wa,wts,veclen):wts+=2*veclen 'f(x)=a*x x>=0,f(x)=b*x x<0
		if i=depth then exit for
		scale(wa,wa,sc,veclen) 'scale for both WHT and switch slope at zero function
	next					   
	fht_float(wa,ln2) 'Extra WHT to reduce quatization from the switching functions
	copy(result,wa+hvec,hvec)
end sub
		
'returns 0 on success   
function xhnet.save(filename as string) as integer
   dim as integer e,f
   f=freefile()
   open filename for binary access write as #f
   e or=put( #f,,veclen)
   e or=put( #f,,depth)
   e or=put( #f,,ln2)
   e or=put( #f,,weights())
   close #f
   return e
end function
 
'returns 0 on success
function xhnet.load(filename as string) as integer
   dim as integer e,f
   f=freefile()
   open filename for binary access read as #f
   e or=get( #f,,veclen)
   e or=get( #f,,depth)
   e or=get( #f,,ln2)
   memsize()
   e or=get( #f,,weights())
   close #f
   return e
end function

sub presentData(array as single ptr,x as ulong,y as ulong,edge as ulong)
dim as ulong idx
for j as ulong=0 to edge-1
for i as ulong=0 to edge-1
	dim as single r=array[idx]:idx+=1
	dim as single g=array[idx]:idx+=1
	dim as single b=array[idx]:idx+=2
	if(r>1!) then r=1!
	if(g>1!) then g=1!
	if(b>1!) then b=1!
	if(r<-1!) then r=-1!
	if(g<-1!) then g=-1!
	if(b<-1!) then b=-1!
	r=r*127.5!+127.5
	g=g*127.5!+127.5
	b=b*127.5!+127.5
	pset (i+x,j+y),RGB(r,g,b)
next
next
end sub

const as string IMG_FILE="imgdata.dat"
const as string NET_FILE="net15.dat"
const edge=32
const threads=2
const mutations=75

dim shared as ulong size=4*edge*edge
dim shared as xhnet nets(threads)
dim shared as boolean shouldrun
dim shared as single parentCost=1!/0!
dim shared as ulong imgcount,iter
dim shared as any ptr mutex
mutex=mutexcreate()

for i as ulong=0 to ubound(nets)
  nets(i).init(size,5)
next
dim as any ptr threadList(threads-1)
dim as integer ff=freefile()
open IMG_FILE for binary access read as #ff
get #ff,,imgcount
dim shared as single imgdata(imgcount*size-1)
get #ff,,imgdata()
close #ff

sub threadsub(x as any ptr)
  dim as rnd256 r
  r.init()
  dim as xhnet ptr child=x
  dim as single vec(size-1)
  dim as ulong n=ubound(child->weights)+1
  while shouldrun
	dim as single childcost=0!
	for i as ulong=0 to imgcount-1
		child->recall(@vec(0),@imgdata(i*size))
		childcost+=errorl2(@vec(0),@imgdata(i*size),size)
	next  
	mutexlock(mutex)
		if childcost<parentcost then 
			parentcost=childcost
			copy(@nets(0).weights(0),@child->weights(0),n)
		else
			copy(@child->weights(0),@nets(0).weights(0),n)
		end if
		iter+=1
	mutexunlock(mutex)
	for i as ulong=1 to mutations 
	    dim as ulong idx=r.nextex(n)
		dim as single v=child->weights(idx)
		dim as single mut=r.nextmutation()
		mut+=v
		if mut>1! then mut=v
		if mut<-1! then mut=v
		child->weights(idx)=mut
    next
  wend
end sub

screenres 400,400,32
windowtitle(NET_FILE)

dim as single work(size-1)
dim as boolean re,ne
dim as ulong recallcount
if fileExists(NET_FILE)  then 
	nets(0).load(NET_FILE)
	for i as ulong=1 to threads-1
	  copy(@nets(i).weights(0),@nets(0).weights(0),ubound(nets(0).weights)+1)
	next
end if	
do
  var k=inkey()
  if (k="t") or (k="T") and not shouldrun then
   shouldrun=true
   re=false
   ne=false
   for i as ulong=0 to threads-1
     threadlist(i)=threadcreate(@threadsub, cptr(any ptr,@nets(i+1)))
   next  
  end if
  if (k="r") or (k="R") and not shouldrun then
    re=true
    ne=false
  end if 
  if (k="n") or (k="N") and not shouldrun then
    re=false
    ne=true
  end if 
  if (k="s") or (k="S") then
    if shouldrun then
      shouldrun=false
      for i as ulong = 0 to ubound(threadlist)
		 ThreadWait(threadlist(i))
	  next
	  nets(0).save(NET_FILE)
	end if  	 
    re=false
    ne=false
  end if 
  if k=chr(27) then 
	exit do
  end if	
  if not (re or ne or shouldrun) then
   cls
   draw string (5,20),"T to Train, R to Recall, S to Stop, Esc to Quit"
   sleep(100)
   continue do
  end if
  if shouldrun then
	cls
	draw string (20,20),"Training    Cost:"+Str(parentcost)+"   Iter:"+Str(iter)
	sleep(1000)
	continue do
  end if
  if re then
    cls
    draw string (20,20),"Recall"
     nets(0).recall(@work(0),@imgData(recallcount*size))
	 presentData(@work(0),100,100,edge)
	 sleep(2000)
	 recallcount+=1
	 if recallcount=imgcount then recallcount=0
  end if 
  if ne then
    cls
    draw string (20,20),"Recall from Noise"
    for i as ulong=0 to ubound(work)
       work(i)=2!*rnd()-1!
     next
     nets(0).recall(@work(0),@work(0))
	 presentData(@work(0),100,100,edge)
	 sleep(2000)
  end if
loop
