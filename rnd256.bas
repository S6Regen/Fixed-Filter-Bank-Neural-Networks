
function hash64(h as ulongint) as ulongint
	h = ( h xor ( h shr 30 ) ) * &hBF58476D1CE4E5B9
	h = ( h xor ( h shr 27 ) ) * &h94D049BB133111EB
	return h xor ( h shr 31 )
end function

' Read the time stamp counter (x86-64) in a messed up way
function timestamp naked() as ulongint
   asm
     rdtsc
     add rdx,rax
     bswapq rdx
     or rax,rdx
     ret
   end asm
end function

function bitstosingle naked(b as ulong) as single
asm
    movd xmm0,edi
    ret
end asm
end function

type rnd256
	as ulongint s0,s1,s2,s3
	declare sub init()
	declare function next64() as ulongint
	declare function nextinc (max as ulong) as ulong
	declare function nextinc (min as long, max as long) as long
	declare function nextex (max as ulong) as ulong
	declare function nextex (min as long, max as long) as long
	declare function nextsingle() as single
	declare function nextsinglesym() as single
	declare function nextmutation() as single
end type	

function rnd256.next64() as ulongint
	dim as ulongint res=hash64(s0)
	dim as ulongint t = s1 shl 17
	s2 xor= s0
	s3 xor= s1
	s1 xor= s2
	s0 xor= s3
	s2 xor= t
	s3=(s3 shl 45) or (s3 shr 19)
	return res
end function

sub rnd256.init()
	s0=hash64(timestamp())
	s1=hash64(timestamp())
	s2=hash64(timestamp())
	s3=hash64(timestamp())
end sub

function rnd256.nextinc (max as ulong) as ulong
        dim as ulongint r = next64() and &hffffffff
        r *= CULngInt(max) + 1
        r = r shr 32
        return  r
end function

function rnd256.nextinc (min as long, max as long) as long
        dim as ulongint r = next64() and &hffffffffull
        r *= CULngInt(max-min) + 1
        r = r shr 32
        return  r+min
end function

function rnd256.nextex overload (max as ulong) as ulong
        dim as ulongint r = next64() and &hffffffffull
        r *= CULngInt(max)
        r = r shr 32
        return  r
end function

function rnd256.nextex overload (min as long, max as long) as long
        dim as ulongint r = next64() and &hffffffffull
        r *= CULngInt(max-min)
        r = r shr 32
        return  r+min
end function

function rnd256.nextsingle() as single
	return next64()*(0.99999997!/&hffffffffffffffffULL)
end function
	
function rnd256.nextsinglesym() as single
    dim as longint r=next64()
	return r*(-0.99999997!/&h8000000000000000)
end function

function rnd256.nextmutation() as single
	dim as ulong r=next64()
	r and=&hbfffffff
	return bitstosingle(r)
end function






