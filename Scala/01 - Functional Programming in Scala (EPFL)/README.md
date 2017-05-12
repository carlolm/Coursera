**Trello Notes**: [https://trello.com/b/osMaQAgE]


https://www.youtube.com/watch?v=DzFt0YkZo8M

- `var`: can change
- `val`: constant

## Data Types
- Byte: -128 to 127
- Boolean
- Char: unsigned max value 65535
- Short: -32768 to 32767
- Int: -2147483648 to 2147483647
- Long: -9223372036854775808 to 9223372036854775807
- Float: -3.4028235E38 to 3.4028235E38
- Double: -1.7976931348623157E308 to 1.7976931348623157E308

```
val lgprime = BigInt("6516515616516546847684735")
val bigDec = BigDecimal("3.14516516515616516516516")
```

-----

**Importing Libraries**

- `import scala.math._`

- **Options**: type for representing optional values
> - `val someValue: Option[Int] = Some(1)`
> - `val noValue : Option[Int] = None`

- Option is a container: Option[T] can be either Some[T] or None

-----

**Function Declarations**

- `def lambda = { x: Int ⇒ x + 1 }`
- `def lambda2 = (x: Int) ⇒ x + 1`
- `val lambda3 = (x: Int) ⇒ x + 1`
- `val lambda4 = new Function1[Int, Int] {`
>  `def apply(v1: Int): Int = v1 + 1`
> `}`

- `def lambda5(x: Int) = x + 1`

-----

**Lists**
- immutable, linked list
- `eq` tests identity (same object?)
- `==` tests equality (content)
- access via head, headOption, and tail; use headOption vs head in case of `IndexOutOfBoundsException`, i.e. no head
- `list.foldLeft(start_val)(expression)` similar to reduce but has a starting value

---

# Week 4

```
object Number {
	def apply(n: Int) = new Number(n)
}
```

- `apply`: allows for Number(2) => Number.apply(2)

- _Pure_ object-oriented language: every value is an object, every operation is essentially an operation on an object
- If the language is based on _classes_, then type of each value is a class
- **eta-expansion**: 

# Interactions with subtyping and generics
- **Bounds**: subject parameters to subtype constraints
- **Variance**: defines how parameterized types behave under subtyping

```
assertAllPos(Empty) => Empty
assertAllPos(NonEmpty) => NonEmpty or throw exception
```

```
def assertAllPos[S <: IntSet](r: S): S = 
```

- Parameter is a subtype of IntSet
- Function returns the same subtype as was input
- `S >: T`: S is a supertype of T

```
[S >: NonEmpty <: IntSet]
```

- bound by types above NonEmpty and below IntSet (inclusive)
- **covariant**: subtyping relationship varies with the type parameter, for example: `List[NonEmpty] <: List[IntSet]`
- **Liskov Substitution Principle**: if A <: B, everything one can do with a value of type B, one can also do with a value of type A
- in Scala, arrays *are not* covariant

---

# Variance

- Lists are immutable (can be covariant), arrays are mutable (can not be covariant)

1. `C[A] <: C[B]` : C is covariant

```
class C[+A] {}
```

2. `C[A] >: C[B]` : C is contravariant

```
class C[-A] {}
```
3. neither: nonvariant

```
class C[A] {}
```

### Typing Rules for Functions
- function A is a subtype of function B if the input/output types of B also satisfy A
- If A2 <: A1 and B1 <: B2, then A1 => B1 <= A2 => B2
- Functions are _contravariant_ in their argument type(s) and _covariant_ in their result types
- **Scala Variance checks**:
 1. _Covariant_ type parameters can only appear in method results
 2. _Contravariant_ type parameters can only appear in method parameters
 3. _Invariant_ type parameters can appear anywhere

```
package scala
trait Function1[-T, +U] {
	def apply(x: T): U
}
```

---

### Decomposition
- type tests and type casts
 - `def isInstanceOf[T]: Boolean`: checks of object's type is T
 - `def asInstanceOf[T]: T`: treats this object as an instance of T, throws ClassCastException if it isn't
 - discouraged in Scala, included for interoperability with Java
- **Pattern Matching**

```
e match {
	case pattern1 => expression1
	case pattern2 => expression2
	case Sum(e1, e2) => eval(e1) + eval(e2)
	// Note that e1 and e2 are assigned as variables when matched
}
```

- exception: MatchError
- variables always begin with a lowercase letter
- same variable can not appear more than once in a patter (e.g. Sum(x,x) not allowed)
- Names of constants begin with a capital letter

---

# Lists
- immutable and recursive (vs arrays which are flat)

```
List(A, List(B, List(C, Nil)))
A :: B :: C // ':' associates to the right 
```

- ':' == prepend

---

# Week 5: Lists

### List Methods
- `xs.length`
- `xs.last`
- `xs.init`: opposite of tail, all elements except for last
- `xs take n`: first n elements of the list
- `xs drop n`: last (length - n) elements of the list
- `xs(n)`
- `xs ++ ys`: concatenation
- `xs.reverse`
- `xs updated (n,x)`: new list where xs(n) reassigned to x
- `xs indexOf x`
- `xs contains x`

## Pairs and Tuples

## Higher Order List Functions

- `map` | `filter` | `filterNot`
- `partition p` => (xs filter p, xs filterNot p)
- `takeWhile p`: longest prefix of list that satisfies p
- `dropWhile`: ! takeWhile p
- `span p` => (xs takeWhile p, xs dropwhile p)
- `reduceLeft op`
- `(xs foldleft a)(op)`: reduce with accumulator a

## Reasoning About Concat
- **Natural induction**: a property P(n) for all integers n >= b
 - P(b) = base case
 - **induction step**: if one has P(n), then one also has P(n+1)
- **Referential transparency**: can freely apply reduction steps to parts of terms, since functional programs do not have side effects
- **Structural induction**: to prove property P(xs):
 - show that P(Nil) holds (base case)
 - if P(xs) holds, then P(x::xs) also holds

---

# Week 6: Other Collections

- **Vector**: similar to Lists, sublcass of Sequences
 - more evenly balanced access patterns than List
 - <= 32 elements: 32 element array
 - >=32 elements: 32 element points to 32 element arrays (32 x 32 = 2^10)
 - additional levels as needed (*= 2^5)
 - number of access = depth of tree = log_{32}(N)
 - similar List operations, except for cons ::
  - Vector alternative: x +: xs || xs :+ x

- **Ranges**
 - val r: Range = 1 until 5 (excludes 5)
 - val s: Range = 1 to 5 (includes 5)
 - 1 to 10 by 3

## Sequence Operations
- `xs exists p`
- `xs forall p`
- `xs zip ys`: combines elements from two lists to a list of tuples
- `xs.unzip`: creates two separate lists
- `xs.flatMap f`: f creates a collection for each element and flatMap returns a concatenated result
- `xs.sum`
- `xs.product`
- `xs.max`
- `xs.min`

## For Expression
- `for (s) yield e`
 - s: sequence of generators and filters
  - s can be wrapped in {} for a multiple line sequence
 - e: expression whose value is returned by an iteration
 - example: 

  ```
  for (p <- persons if p.age < 20) yield p.name
  ```

## Sets
1. Sets are unordered
2. Sets do not have duplicate elements
3. Fundamental operation: `contains`

## Maps
- Map[Key, Value]: associates keys of type Key with values of type Value
- **Useful methods**:
 - [Map]`.toList`: transform to list of pairs
 - [List of pairs]`.toMap`


## Option Type
```
trait Option[+A]
case class Some[+A](value: A) extends Option[A]
object None extends Option[Nothing]
```

- **groupBy**: partitions a collection into a map of collections according to a discriminator function f
- **withDefaultValue**
 - def func = [mapName] withDefaultValue [default]
 - instead of mapName(val), func(val)

---



---

## Syntax Shortcuts
- `((x,y) => x * y)` => `(_*_)`


 






















