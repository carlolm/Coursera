/**
  *     Carlo P. Las Marias
  *     carlol@gmail.com
  *     20 March 2017
  */

/**
  *     Problem: Place n queens on an n x n board such
  *              that no queens are attacking each other 
  *
  *     Function takes in n and returns a Set of List[Int]
  *     representing valid solutions.
  * 
  *     For example, for n = 4,
  *     a valid solution would be List(2,0,3,1),
  *     where rows are represented by list positions
  *     and the element represents the column position:
  *                     O O X O
  *                     X O O O
  *                     O O O X
  *                     O X O O
  */

def queens(n: Int): Set[List[Int]] = {
  
  // Place k queens on the board
  def placeQueens(k: Int): Set[List[Int]] =

    // If no queens need to be placed
    // return a set with an empty list
    if (k == 0) Set(List())
    else
      for {
        // Sub-problem: place (k-1) queens which is a set
        // of partial solutions [recursion]
        queens <- placeQueens(k-1)
        // Try all column positions
        col <- 0 until n
        // Filter: only columns that are safe
        if isSafe(col, queens)
      
        // return the new column added to the partial
        // solution 'queens'
      } yield col :: queens


  // Check if a position is safe
  def isSafe(col: Int, queens: List[Int]): Boolean = {
    val row = queens.length
    // Create a list of tuples (r, c)
    // Define range of rows and zip it with queens
    val queensWithRow = (row - 1 to 0 by -1) zip queens

    queensWithRow forall {
      // check each that columns are all different
      // and no diagonal checks
      case (r,c) => col != c && math.abs(col - c) != row - r
    }

  }

  placeQueens(n)

}

// input a solution
def show(queens: List[Int]) = {
  val lines =
    // reverse to show earlier rows first
    for (col <- queens.reverse)
    // create a vector that is the size of the queens
    // fill with "* " | update places an "X " where the queen is
    // mkString prints out all elements of a collection (≈ join(''))
    yield Vector.fill(queens.length)("* ").updated(col, "X ").mkString
  "\n" + (lines mkString "\n")
}

queens(8)
queens(8).size
(queens(8) take 3 map show) mkString "\n"