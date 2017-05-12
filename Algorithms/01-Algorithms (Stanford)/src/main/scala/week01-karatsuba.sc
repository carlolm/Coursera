/**
  * Carlo P. Las Marias
  * carlol@gmail.com
  * 11 May 2017
  */


/**
  * Get number of digits of a number
  * @param x     the number
  * @return      integer
  */

def len(x: BigInt): Int = x.toString.length

/**
  * Split a number in half to high and low digits
  * @param x    the number
  * @param m    place to split
  * @return
  */

def split(x: BigInt, m: Int): (BigInt,BigInt) = {
  val string = x.toString
  (BigInt(string.dropRight(m)), BigInt(string.takeRight(m)))
}

/**
  * Karatsuba multiplication algorithm
  * Handling big int
  * @param num1:  BigInt number 1
  * @param num2:  BigInt number 2
  * @return       BigInt number product
  */

def karatsuba(num1: BigInt, num2: BigInt): BigInt = {

  if (num1 < 10 || num2 < 10)
    num1 * num2
  else {
    val m = math.min(len(num1), len(num2))
    val (a, b) = split(num1, m/2)
    val (c, d) = split(num2, m/2)

    val ac = karatsuba(a, c)
    val bd = karatsuba(b, d)
    val ab_cd = karatsuba(a+b, c+d)

    ac * BigInt("1" + "0" * (m/2 * 2)) + ((ab_cd - ac - bd) * BigInt("1" + "0" * (m/2))) + bd
  }
}

/**
  *   Testing
  */

var num1 = BigInt("1" * 4)
var num2 = num1
var a = karatsuba(num1, num2)
var b = num1 * num2
assert(a == b)

num1 = BigInt("123456")
num2 = BigInt("12")
a = karatsuba(num1, num2)
b = num1 * num2
assert(a == b)

num1 = BigInt("1234")
num2 = BigInt("5678901")
a = karatsuba(num1, num2)
b = num1 * num2
assert(a == b)

num1 = BigInt("12345")
num2 = BigInt("67890")
a = karatsuba(num1, num2)
b = num1 * num2
assert(a == b)

num1 = BigInt("3141592653589793238462643383279502884197169399375105820974944592")
num2 = BigInt("2718281828459045235360287471352662497757247093699959574966967627")

a = karatsuba(num1, num2)
b = num1 * num2
assert(a == b)
