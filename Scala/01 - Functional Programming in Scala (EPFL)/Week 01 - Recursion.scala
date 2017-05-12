/**
  *   Carlo P. Las Marias
  *   carlol@gmail.com
  *   14 Feb 2017
  */

package recfun

object Main {
  def main(args: Array[String]) {
    println("Pascal's Triangle")
    for (row <- 0 to 10) {
      for (col <- 0 to row)
        print(pascal(col, row) + " ")
      println()
    }
  }

  /**
   * Exercise 1
   */
    def pascal(c: Int, r: Int): Int =
      // Handle input error. Column can not exceed row
      // Return 0.  Alternative, handling:
      // throw new IllegalArgumentException("Column can not be greater than row")
      // Shouls also check if inputs are valid: Ints and defined
      if (c > r)
        0
      else
      if (c == 0 || r == c)
        1
      else
        pascal(c-1, r-1) + pascal(c, r-1)

  /**
   * Exercise 2
   */
    def balance(chars: List[Char]): Boolean = {

      def loop(acc: Int, charsInner: List[Char]): Boolean = {

        if (acc < 0)
          false
        else if (charsInner.isEmpty)
          acc == 0
        else {
          val i = charsInner.head match {
            case '(' => 1
            case ')' => -1
            case _ => 0
          }
          loop(acc + i, charsInner.tail)
        }
      }

      loop(0, chars)
    }

  /**
   * Exercise 3
   */
    def countChange(money: Int, coins: List[Int]): Int = {
      if (money == 0)
        1
      else if (coins.isEmpty || money < 0)
        0
      else
        countChange(money - coins.head, coins) + countChange(money, coins.tail)
    }
  }
