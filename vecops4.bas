#define MINIMUM_ADJ_ 1e-20!

'40000/s
sub absolute(result as single ptr,x as single ptr,n as ulongint)
	dim as ulong ptr resultptr=cast(ulong ptr,result)
	dim as ulong ptr xptr=cast(ulong ptr,x)
	for i as ulongint=0 to n-1
		resultptr[i]=xptr[i] and &h7fffffff
	next
end sub

sub square(result as single ptr,x as single ptr,n as ulongint)
	for i as ulongint=0 to n-1
		result[i]=x[i]*x[i]
	next
end sub

'39000/s
sub signof(result as single ptr,x as single ptr,n as ulongint)
	dim as ulong ptr resultptr=cast(ulong ptr,result)
	dim as ulong ptr xptr=cast(ulong ptr,x)
	for i as ulongint=0 to n-1
		resultptr[i]=(xptr[i] and &h80000000) or &h3f800000 ' or 1!
	next
end sub

'29000/s
sub signedsqr(result as single ptr,x as single ptr,n as ulongint)
	dim as ulong ptr resultptr=cast(ulong ptr,result)
	dim as ulong ptr xptr=cast(ulong ptr,x)
	for i as ulongint=0 to n-1
		dim as ulong f=xptr[i],s=f and &h80000000
		f and=&h7fffffff
		f +=&h3f800000
		f shr=1
		f or=s
		resultptr[i]=f
	next
end sub

'38000/s
sub signedsquare(result as single ptr,x as single ptr,n as ulongint)
	dim as ulong ptr rlptr=cast(ulong ptr,result)
	dim as ulong ptr xlptr=cast(ulong ptr,x)
	for i as ulongint=0 to n-1
		dim as ulong s=&h80000000ul and xlptr[i]
		result[i]=x[i]*x[i]
		rlptr[i] or=s
	next
end sub

'15000/s
sub truncate(result as single ptr,x as single ptr,t as single,n as ulongint)
	for i as ulongint=0 to n-1
		dim as single v=abs(x[i])-t
		if v<0! then v=0!
		if x[i]<0! then v=-v
	    result[i]=v
	next
end sub

'44000/s
sub scale(result as single ptr,x as single ptr,sc as single,n as ulongint)
	for i as ulongint=0 to n-1
		result[i]=x[i]*sc
	next
end sub

'47000/s
sub add(result as single ptr,x as single ptr,y as single ptr,n as ulongint)
	for i as ulongint=0 to n-1
		result[i]=x[i]+y[i]
	next
end sub

'46000/s
sub subtract(result as single ptr,x as single ptr,y as single ptr,n as ulongint)
	for i as ulongint=0 to n-1
		result[i]=x[i]-y[i]
	next
end sub

'43000/s
sub multiply(result as single ptr,x as single ptr,y as single ptr,n as ulongint)
	for i as ulongint=0 to n-1
		result[i]=x[i]*y[i]
	next
end sub

'8000/s
function errorl2(x as single ptr,y as single ptr,n as ulongint) as double
	dim as double e
	for i as ulongint=0 to n-1
		dim as single d=x[i]-y[i]
		e+=d*d
	next
	return e
end function

'8000/s
function errorl1(x as single ptr,y as single ptr,n as ulongint) as double
	dim as double e
	for i as ulongint=0 to n-1
		dim as single d=x[i]-y[i]
		e+=abs(d)
	next
	return e
end function

'9000/s
function sumsq(x as single ptr,n as ulongint) as double
	dim as double e
	for i as ulongint =0 to n-1
		e+=x[i]*x[i]
	next
	return e
end function

'7000/s
sub adjust (result as single ptr,x as single ptr,adjscale as single,n as ulongint)
	dim as single adj=adjscale/(sqr(sumsq(x,n)/n)+MINIMUM_ADJ_)
	scale(result,x,adj,n)
end sub
		
'16000/s size of x divible by 4
sub signflip(x as single ptr,h as ulongint,n as ulongint)
	dim as ulong ptr xlptr=cast(ulong ptr,x)
	h+=1442695040888963407ULL '&h6A09E667F3BCC908ULL Used Knuth
    h*=6364136223846793005ULL '&h9E3779B97F4A7C15ULL
	for i as ulongint=0 to n-1 step 4
		 dim as ulong s=h shr 32
		 h+=1442695040888963407ULL '&h6A09E667F3BCC908ULL
         h*=6364136223846793005ULL '&h9E3779B97F4A7C15ULL
		 xlptr[i] xor=s and &h80000000  
		 s+=s
		 xlptr[i+1] xor=s and &h80000000 
		 s+=s
		 xlptr[i+2] xor=s and &h80000000
		 s+=s
		 xlptr[i+3] xor=s and &h80000000	  
   next
end sub

sub zero(x as single ptr,n as ulongint)
	for i as ulongint=0 to n-1
		x[i]=0!
	next
end sub

sub copy(result as single ptr,x as single ptr,n as ulongint)
	for i as ulongint=0 to n-1
		result[i]=x[i]
	next
end sub
 
'25000/s
sub multiplyaddto(result as single ptr,x as single ptr,y as single ptr,n as ulongint)
	for i as ulongint=0 to n-1
		result[i]+=x[i]*y[i]
	next
end sub
/'
'includes bias
sub switchaddto(result as single ptr,x as single ptr,wt as single ptr,n as ulongint)
    dim as single ptr wt2=wt+n,wt3=wt2+n
	for i as ulongint=0 to n-1
	   dim as single v=x[i]+wt[i]
	   if v<0! then 
		  v*=wt2[i]
	   else 
		  v*=wt3[i]
	   end if
	   result[i]+=v
	next
end sub

sub sqswitchaddto(result as single ptr,x as single ptr,wt as single ptr,n as ulongint)
    dim as single ptr wt2=wt+n,wt3=wt2+n
	for i as ulongint=0 to n-1
	   dim as single v=x[i]+wt[i]
	   dim as single w=v*v
	   if v<0! then 
		  w*=wt2[i]
	   else 
		  w*=wt3[i]
	   end if
	   result[i]+=w
	next
end sub

'/
function sumsquare naked (x as single ptr,n as ulongint) as single
asm
	xorps xmm0,xmm0
	xorps xmm1,xmm1
	xorps xmm2,xmm2
	xorps xmm3,xmm3
	.align 16
sumsquarelp:
    movups xmm4,[rdi]
    movups xmm5,[rdi+16]
    movups xmm6,[rdi+32]
    movups xmm7,[rdi+48]
    sub rsi,16
    mulps xmm4,xmm4
    mulps xmm5,xmm5
    mulps xmm6,xmm6
    mulps xmm7,xmm7
    lea rdi,[rdi+64]
    addps xmm0,xmm4
    addps xmm1,xmm5
    addps xmm2,xmm6
    addps xmm3,xmm7
    jnz sumsquarelp
    haddps xmm0,xmm1
    haddps xmm2,xmm3
    haddps xmm0,xmm2
    haddps xmm0,xmm0
    haddps xmm0,xmm0
    ret 
end asm
end function

sub switchaddto naked(addto as single ptr,x as single ptr,wts as single ptr,n as ulongint)
asm
	xorps xmm0,xmm0  'zero xmm0
	.align 16
switchaddtolp:
	movups xmm12,[rsi]	'x values
	movups xmm13,[rsi+16]
	movups xmm14,[rsi+32]
	movups xmm15,[rsi+48]
	movups xmm5,[rdx]   'wt block 1
	movups xmm6,[rdx+16]
	movups xmm7,[rdx+32]
	movups xmm8,[rdx+48]
	sub rcx,16
	movaps xmm1,xmm12	'copy x values
	movaps xmm2,xmm13
	movaps xmm3,xmm14
	movaps xmm4,xmm15
	cmpltps xmm1,xmm0	'masks
	cmpltps xmm2,xmm0
	cmpltps xmm3,xmm0
	cmpltps xmm4,xmm0
	andps xmm5,xmm1 	'wt block 1 and mask
	andps xmm6,xmm2
	andps xmm7,xmm3
	andps xmm8,xmm4
	andnps xmm1,[rdx+64] 'wt block 2 and not mask
	andnps xmm2,[rdx+80]
	andnps xmm3,[rdx+96]
	andnps xmm4,[rdx+112]
    orps xmm1,xmm5
    orps xmm2,xmm6
    orps xmm3,xmm7
    orps xmm4,xmm8
    mulps xmm1,xmm12
    mulps xmm2,xmm13
    mulps xmm3,xmm14
    mulps xmm4,xmm15
    addps xmm1,[rdi]
    addps xmm2,[rdi+16]
    addps xmm3,[rdi+32]
    addps xmm4,[rdi+48]
    lea rsi,[rsi+64]
    lea rdx,[rdx+128]
    movups [rdi],xmm1
    movups [rdi+16],xmm2
    movups [rdi+32],xmm3
    movups [rdi+48],xmm4
    lea rdi,[rdi+64]
    jnz switchaddtolp
    ret
end asm
end sub

sub switch naked(x as single ptr,wts as single ptr,n as ulongint)
asm
	xorps xmm0,xmm0  'zero xmm0
	.align 16
switchlp:
	movups xmm12,[rdi]	'x values
	movups xmm13,[rdi+16]
	movups xmm14,[rdi+32]
	movups xmm15,[rdi+48]
	movups xmm5,[rsi]   'wt block 1
	movups xmm6,[rsi+16]
	movups xmm7,[rsi+32]
	movups xmm8,[rsi+48]
	sub rdx,16
	movaps xmm1,xmm12	'copy x values
	movaps xmm2,xmm13
	movaps xmm3,xmm14
	movaps xmm4,xmm15
	cmpltps xmm1,xmm0	'masks
	cmpltps xmm2,xmm0
	cmpltps xmm3,xmm0
	cmpltps xmm4,xmm0
	andps xmm5,xmm1 	'wt block 1 and mask
	andps xmm6,xmm2
	andps xmm7,xmm3
	andps xmm8,xmm4
	andnps xmm1,[rsi+64] 'wt block 2 and not mask
	andnps xmm2,[rsi+80]
	andnps xmm3,[rsi+96]
	andnps xmm4,[rsi+112]
    orps xmm1,xmm5
    orps xmm2,xmm6
    orps xmm3,xmm7
    orps xmm4,xmm8
    mulps xmm1,xmm12
    mulps xmm2,xmm13
    mulps xmm3,xmm14
    mulps xmm4,xmm15
   
    movups [rdi],xmm1
    movups [rdi+16],xmm2
    movups [rdi+32],xmm3
    movups [rdi+48],xmm4
    lea rdi,[rdi+64]
    lea rsi,[rsi+128]
    jnz switchlp
    ret
end asm
end sub

' switch slope at zero, x to the power of 1.5 
sub switchp15 naked(addto as single ptr,x as single ptr,wts as single ptr,n as ulongint)
asm
	xorps xmm0,xmm0  'zero xmm0
	mov eax,0x7fffffff
	movd xmm9,eax
	mov eax,0x3f800000
	movd xmm10,eax
	shufps xmm9,xmm9,0
	shufps xmm10,xmm10,0
	.align 16
switch15lp:
	movups xmm12,[rsi]	'x values
	movups xmm13,[rsi+16]
	movups xmm14,[rsi+32]
	movups xmm15,[rsi+48]
	movups xmm5,[rdx]   'wt block 1
	movups xmm6,[rdx+16]
	movups xmm7,[rdx+32]
	movups xmm8,[rdx+48]
	sub rcx,16
	movaps xmm1,xmm12	'copy x values
	movaps xmm2,xmm13
	movaps xmm3,xmm14
	movaps xmm4,xmm15
	cmpltps xmm1,xmm0	'masks
	cmpltps xmm2,xmm0
	cmpltps xmm3,xmm0
	cmpltps xmm4,xmm0
	andps xmm5,xmm1 	'wt block 1 and mask
	andps xmm6,xmm2
	andps xmm7,xmm3
	andps xmm8,xmm4
	andnps xmm1,[rdx+64] 'wt block 2 and not mask
	andnps xmm2,[rdx+80]
	andnps xmm3,[rdx+96]
	andnps xmm4,[rdx+112]
    orps xmm1,xmm5
    orps xmm2,xmm6
    orps xmm3,xmm7
    orps xmm4,xmm8
    mulps xmm1,xmm12
    mulps xmm2,xmm13
    mulps xmm3,xmm14
    mulps xmm4,xmm15
    pand xmm12,xmm9
    pand xmm13,xmm9
    pand xmm14,xmm9
    pand xmm15,xmm9
    paddd xmm12,xmm10
    paddd xmm13,xmm10
    paddd xmm14,xmm10
    paddd xmm15,xmm10
    psrld xmm12,1
    psrld xmm13,1
    psrld xmm14,1
    psrld xmm15,1
    mulps xmm1,xmm12
    mulps xmm2,xmm13
    mulps xmm3,xmm14
    mulps xmm4,xmm15
    
    addps xmm1,[rdi]
    addps xmm2,[rdi+16]
    addps xmm3,[rdi+32]
    addps xmm4,[rdi+48]
    lea rsi,[rsi+64]
    lea rdx,[rdx+128]
    movups [rdi],xmm1
    movups [rdi+16],xmm2
    movups [rdi+32],xmm3
    movups [rdi+48],xmm4
    lea rdi,[rdi+64]
    jnz switch15lp
    ret
end asm
end sub
/'
dim as single x(63),y(63),w(127)
for i as ulong=0 to ubound(w)
w(i)=-1
next

for i as ulong=0 to ubound(x)
x(i)=-i
next
switchp15(@y(0),@x(0),@w(0),64)

for i as ulong=0 to ubound(x)
 print i,y(i),i^1.5
next
getkey
'/

