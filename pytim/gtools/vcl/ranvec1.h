/****************************  ranvec1.h   ********************************
* Author:        Agner Fog
* Date created:  2014-09-09
* Last modified: 2016-11-25
* Version:       1.25
* Project:       vector classes
* Description:
* Header file defining pseudo random number generators with vector output.
* Two pseudo random number generators are combined:
* 1. "Mersenne Twister for Graphic Processor" (MTGP).
*    (Saito & Matsumoto: Variants of Mersenne Twister Suitable for Graphic
*    Processors". ACM Transactions on Mathematical Software, v. 39, no. 2,
*    2013).
* 2. "Multiply-With-Carry Generator" (MWC).
*    (Press, et. al.: Numerical Recipes: The Art of Scientific Computing,
*    3rd. edition. Cambridge Univ. Press, 2007).
*
* Instructions:
* Make an object of the class Ranvec1. The constructor has a parameter "gtype"
* to indicate the desired random number generator:
* gtype = 1: MWC.  Use for smaller projects and where speed is critical
* gtype = 2: MTGP. Use for large projects with many streams
* gtype = 3: Both generators combined. Use for the most demanding projects
* 
* Multi-threaded programs must make one instance of Ranvec1 for each thread,
* with different seeds. It is not safe to access the same random number 
* generator instance in multiple threads. 
*
* The Ranvec1 object must be initialized with one or more seeds, by calling
* one of the init functions. The same seed will always generate the same 
* sequence of random numbers (with the same value of gtype). A different seed
* or set of seeds will generate a different sequence. You can use one of the
* following member functions for initialization:
*
* void init(int seed):  General initialization function
* void init(int seed1, int seed2):  Use seed1 for the MWC generator and
*        seed2 for the MTGP.
* void initByArray(int seeds[], int numSeeds):  Use an array of seeds.
*        The sequence will change if at least one of the seeds is changed.
*        If gtype = 3 then seeds[0] will be used for the MWC generator and
*        all the remaining seeds will be used for MTGP.
*
* The following member functions can be used for random number outputs:
* Scalars:
* uint32_t random32b():                 Returns an integer of 32 random bits
* int      random1i(int min, int max):  One integer in the interval min <= x <= max
* int      random1ix(int min, int max): Same, with extra precision
* uint64_t random64b():                 Returns an integer of 64 random bits
* float    random1f():                  One floating point number in the interval 0 <= x < 1
* double   random1d():                  One double in the interval 0 <= x < 1
*
* 128 bit vectors:
* Vec4ui   random128b():                Returns 128 random bits as a vector of 4 integers
* Vec4i    random4i(int min, int max):  4 integers in the interval min <= x <= max
* Vec4i    random4ix(int min, int max): Same, with extra precision
* Vec4f    random4f():                  4 floating point numbers in the interval 0 <= x < 1
* Vec2d    random2d():                  2 doubles in the interval 0 <= x < 1
*
* 256 bit vectors:
* Vec8ui   random256b():                Returns 256 random bits as a vector of 8 integers
* Vec8i    random8i(int min, int max):  8 integers in the interval min <= x <= max
* Vec8i    random8ix(int min, int max): Same, with extra precision
* Vec8f    random8f():                  8 floating point numbers in the interval 0 <= x < 1
* Vec4d    random4d():                  4 doubles in the interval 0 <= x < 1
*
* 512 bit vectors:
* Vec16ui  random512b():                Returns 512 random bits as a vector of 16 integers
* Vec16i   random16i(int min, int max): 16 integers in the interval min <= x <= max
* Vec16i   random16ix(int min, int max):Same, with extra precision
* Vec16f   random16f():                 16 floating point numbers in the interval 0 <= x < 1
* Vec8d    random8d():                  8 doubles in the interval 0 <= x < 1
*
* The 256 bit vector functions are available only if MAX_VECTOR_SIZE >= 256.
* The 512 bit vector functions are available only if MAX_VECTOR_SIZE >= 512.
*
* For detailed instructions, see VectorClass.pdf
* For theoretical explanation, see the article: "Pseudo-Random Number Generators
* for Vector Processors and Multicore processors". www.agner.org/random/theory
*
* (c) Copyright 2014-2016 GNU General Public License www.gnu.org/licenses
******************************************************************************/
#ifndef RANVEC1_H
#define RANVEC1_H  122

#include "vectorclass.h"

#ifdef VCL_NAMESPACE
namespace VCL_NAMESPACE {
#endif

/******************************************************************************
        Ranvec1base: Base class for combined random number generator
Do not use this class directly. Use only the derived class Ranvec1
******************************************************************************/

// Combined random number generator, base class
// (Total size depends on instruction set)
class Ranvec1base {
public:
    Ranvec1base(int gtype = 3);                    // Constructor
    void init(int seed);                         // Initialize with seed
    void init(int seed1, int seed2);             // Initialize with seed1 for MWC and seed2 for MTGP
    void initByArray(int32_t const seeds[], int numSeeds); // Initialize by array of seeds
    void next(uint32_t * dest);                  // Produce 16*32 = 512 random bits
protected:
    void initMWC(int32_t seed);                  // Initialize MWC with seed
    void initMTGP(int32_t seed);                 // Initialize MTGP with seed
    void initMTGPByArray(int32_t const seeds[], int numSeeds); // Initialize MTGP by array of seeds
#if INSTRSET < 8                                 // SSE2 - AVX: use 128 bit vectors
    Vec4ui next1();                              // Get 128 bits from MWC
    Vec4ui next2();                              // Get 128 bits from MTGP
#elif INSTRSET < 9                               // AVX2: use 256 bit vectors
    Vec8ui next1();                              // Get 256 bits from MWC
    Vec8ui next2();                              // Get 256 bits from MTGP
#else                                            // AVX512: use 512 bit vectors
    Vec16ui next1();                             // Get 512 bits from MWC
    Vec16ui next2();                             // Get 512 bits from MTGP
#endif

    // State buffer for MWC
    uint32_t buffer1[16];                        // State buffer for MWC
    int iw;                                      // buffer1 index

    // Constant parameters for MTGP and MWC
    enum constants {
        // Constant parameters for MTGP-11213, no. 1
        mexp    = 11213,                         // Mersenne exponent
        bsize   = (mexp + 31) / 32,              // Size of state buffer, 32-bit words
        vs      = 4,                             // Vector size, 32-bit words
        csize   = (bsize + vs - 1) / vs,         // Size of state buffer in 128, 256 or 512-bit vectors
        bo      = 16,                            // Offset at beginning and end of buffer to enable unaligned access
        mpos    = 84,                            // Middle position index
        sh1     = 12,                            // Shift count 1
        sh2     =  4,                            // Shift count 2
        tbl0    = 0x71588353,                    // Transformation matrix row 0
        tbl1    = 0xdfa887c1,                    // Transformation matrix row 1
        tbl2    = 0x4ba66c6e,                    // Transformation matrix row 2
        tbl3    = 0xa53da0ae,                    // Transformation matrix row 3
        temper0 = 0x200040bb,                    // Tempering matrix row 0
        temper1 = 0x1082c61e,                    // Tempering matrix row 1
        temper2 = 0x10021c03,                    // Tempering matrix row 2
        temper3 = 0x0003f0b9,                    // Tempering matrix row 3
        mask    = 0xfff80000,                    // Bit mask

        // Factors for MWC generators
        mwcfac0 = 4294963023,                    // Factor for each MWC generator
        mwcfac1 = 3947008974,
        mwcfac2 = 4162943475,
        mwcfac3 = 2654432763,
        mwcfac4 = 3874257210,
        mwcfac5 = 2936881968,
        mwcfac6 = 4294957665,
        mwcfac7 = 2811536238,
        shw1    = 30,                            // Shift counts for MWC tempering
        shw2    = 35,
        shw3    = 13
    };
    // Variables and state buffer for MTGP
    int gentype;                                 // Generator type
    int idx;                                     // Buffer index
    int idm;                                     // Buffer index to middle position
#if INSTRSET < 8                                 // SSE2 - AVX: use 128 bit vectors
    Vec4ui nextx;                                // Temporary vector in generation
    Vec4ui xj;                                   // x value at middle position
#elif INSTRSET < 9                               // AVX2: use 256 bit vectors
    Vec8ui nextx;                                // Temporary vector in generation
    Vec8ui xj;                                   // x value at middle position
#else                                            // AVX512: use 512 bit vectors
    Vec16ui nextx;                               // Temporary vector in generation
    Vec16ui xj;                                  // x value at middle position
#endif
    uint32_t buffer2[csize*vs + bo*3];           // State buffer for MTGP
};


// 512 bit output buffer
// Filled with 512 bits at a time,
// Returns 32, 64, 128, 256 or 512 bits at a time
// Use one instance for each output size, don't mix output sizes
class Buf512 {
public:
    Buf512(Ranvec1base * r) {                    // Constructor
        p = r; ix = 16;}
    void reset () {                              // Reset
        ix = 16;
    }
    uint32_t get32() {                           // Get integer
        if (ix > 15) fill();                     // Fill on first call
        uint32_t x = bb[ix++];                   // Get one element from buffer
        if (ix > 15) fill();                     // Fill when last element retrieved
        return x;
    }
    uint64_t get64() {                           // Get 64-bit integer
        if (ix > 15) fill();                     // Fill on first call
        uint64_t x = *(uint64_t*)(bb+ix);        // Get 64 bits from buffer
        ix += 2;
        if (ix > 15) fill();                     // Fill when last element retrieved
        return x;
    }
    Vec4ui get128() {                            // Get 128 bits
        if (ix > 15) fill();                     // Fill on first call
        Vec4ui x = Vec4ui().load(bb+ix);         // Get 128 bits from buffer
        ix += 4;
        if (ix > 15) fill();                     // Fill when last element retrieved
        return x;
    }
#if MAX_VECTOR_SIZE >= 256
    Vec8ui get256() {                            // Get 256 bits
        if (ix > 15) fill();                     // Fill on first call
        Vec8ui x = Vec8ui().load(bb+ix);         // Get 256 bits from buffer
        ix += 8;
        if (ix > 15) fill();                     // Fill when last element retrieved
        return x;
    }
#endif
#if MAX_VECTOR_SIZE >= 512
    Vec16ui get512() {                           // Get 512 bits
        if (ix) fill();                          // Fill on first call
        Vec16ui x = Vec16ui().load(bb);          // Get 512 bits from buffer
        fill();                                  // Fill because whole buffer is used
        return x;
    }
#endif
protected:
    Ranvec1base * p;                             // Pointer to generator
    uint32_t bb[16];                             // Integer buffer
    int ix;                                      // Index into buffer bb
    void fill() {                                // Refill buffer
        p->next(bb);
        ix = 0;
    }
};


/******************************************************************************
        Ranvec1: Class for combined random number generator

Make one instance of Ranvec1 for each thread.
Remember to initialize it with a seed. 
Each instance must have a different seed if you want different random sequences
******************************************************************************/

// Combined random number generator. Derived class with various output functions
// (Total size depends on INSTRSET and MAX_VECTOR_SIZE)
class Ranvec1 : public Ranvec1base {
public:
    // Constructor
    Ranvec1(int gtype = 3) : Ranvec1base(gtype), buf32(this), buf64(this), buf128(this)
#if MAX_VECTOR_SIZE >= 256
    , buf256(this)
#endif
#if MAX_VECTOR_SIZE >= 512
    , buf512(this)
#endif
    {
        randomixInterval = randomixLimit = 0;
    }
    // Initialization with seeds
    void init(int seed) {                        // Initialize with one seed
        Ranvec1base::init(seed);
        resetBuffers();
    }
    void init(int seed1, int seed2) {            // Initialize with two seeds
        Ranvec1base::init(seed1, seed2);
        resetBuffers();
    }
    void initByArray(int32_t const seeds[], int numSeeds) { // Initialize by array of seeds
        Ranvec1base::initByArray(seeds, numSeeds);
        resetBuffers();
    }

    // Output functions, scalar:
    uint32_t random32b() {                // Returns an integer of 32 random bits
        return buf32.get32();}
    int      random1i(int min, int max);  // One integer in the interval min <= x <= max
    int      random1ix(int min, int max); // Same, with extra precision
    uint64_t random64b() {                // Returns an integer of 64 random bits
        return buf64.get64();}
    float    random1f();                  // One floating point number in the interval 0 <= x < 1
    double   random1d();                  // One double in the interval 0 <= x < 1

    // Output functions, 128 bit vectors:
    Vec4ui   random128b() {               // Returns 128 random bits as a vector of 4 integers
        return buf128.get128();}
    Vec4i    random4i(int min, int max);  // 4 integers in the interval min <= x <= max
    Vec4i    random4ix(int min, int max); // Same, with extra precision
    Vec4f    random4f();                  // 4 floating point numbers in the interval 0 <= x < 1
    Vec2d    random2d();                  // 2 doubles in the interval 0 <= x < 1

    // Output functions, 256 bit vectors:
#if MAX_VECTOR_SIZE >= 256
    Vec8ui   random256b() {               // Returns 256 random bits as a vector of 8 integers
        return buf256.get256();}
    Vec8i    random8i(int min, int max);  // 8 integers in the interval min <= x <= max
    Vec8i    random8ix(int min, int max); // Same, with extra precision
    Vec8f    random8f();                  // 8 floating point numbers in the interval 0 <= x < 1
    Vec4d    random4d();                  // 4 doubles in the interval 0 <= x < 1
#endif
    // Output functions, 512 bit vectors:
#if MAX_VECTOR_SIZE >= 512
    Vec16ui  random512b() {               // Returns 512 random bits as a vector of 16 integers
        return buf512.get512();}
    Vec16i   random16i(int min, int max); // 16 integers in the interval min <= x <= max
    Vec16i   random16ix(int min, int max);// Same, with extra precision
    Vec16f   random16f();                 // 16 floating point numbers in the interval 0 <= x < 1
    Vec8d    random8d();                  // 8 doubles in the interval 0 <= x < 1
#endif

protected:
    void resetBuffers();                         // Reset all output buffers
    Buf512 buf32;                                // Buffer for 32-bit output
    Buf512 buf64;                                // Buffer for 64-bit output
    Buf512 buf128;                               // Buffer for 128-bit output
#if MAX_VECTOR_SIZE >= 256
    Buf512 buf256;                               // Buffer for 256-bit output
#endif
#if MAX_VECTOR_SIZE >= 512
    Buf512 buf512;                               // Buffer for 512-bit output
#endif
   uint32_t randomixInterval;                    // Last interval for irandomx function
   uint32_t randomixLimit;                       // Last rejection limit for irandomx function
};

#ifdef VCL_NAMESPACE
}
#endif

#endif  // RANVEC1_H
