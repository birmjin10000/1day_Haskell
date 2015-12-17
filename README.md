# 하루만 배워보는 Haskell Programming

Haskell Programming을 딱 하루만 배워봅시다.

6시간 동안 Haskell을 배워서 Haskell로 실용적인 도구를 만들어봅시다.

## 사전 학습
Haskell platform을 설치합니다. 설치후 ghci를 실행합니다. Haskell의 기본 자료형은 List입니다. List는 대괄호로 표시합니다. Haskell에서 한 줄 주석은 수평선 두 개로 표시합니다.

    > [1,2,3] -- [1,2,3]

List를 만들 때는 여러 가지 편리한 방법이 있습니다.

    > [1..5] -- [1,2,3,4,5]
    > [1,4..20] -- [1,4,7,10,13,16,19]
    > [5,4..1] -- [5,4,3,2,1]
    > [10,8..1] -- [10,8,6,4,2]

List에 사용할 수 있는 기본 함수들을 살펴봅니다.

    > sum [1,2,3] -- 6
    > product [1,2,3] -- 6
    > length [1,2,3] -- 3
    > head [1,2,3] -- 1
    > tail [1,2,3] -- [2,3]
    > init [1,2,3] -- [1,2]
    > last [1,2,3] -- 3
    > take 2 [1,2,3] -- [1,2]
    > drop 2 [1,2,3] -- [3]
    > 0:[1,2,3] -- [0,1,2,3]
    > [1,2,3] ++ [4,5,6] -- [1,2,3,4,5,6]
    > maximum [1,2,3] -- 3
    > minimum [1,2,3] -- 1
    > reverse [1,2,3] -- [3,2,1]
    > concat [[1,2],[3],[4,5]] -- [1,2,3,4,5]

Haskell 에서는 모든 동작에 있어서 함수가 중심이 됩니다. 위의 예에서 볼 수 있들이 List 에 어떤 함수를 적용할 때는 "함수 List" 꼴로 적용할 함수가 먼저 나오고 그 뒤에 List가 나옵니다.
가령 sum 함수의 경우 "sum [1,2,3]" 꼴로 호출이 되었습니다. 즉, sum 함수는 List를 하나 받아서 그List의 원소들의 합을 구하는 함수 입니다. 다시 말해 sum함수의 입력은 "List 하나"이고 출력은 "값 하나" 꼴이 됩니다. 이렇듯 함수의 입력과 출력관계를 정의할 수 있는데 이를 함수의 type이라고 합니다. ghci에서는 :type 명령(혹은 :t) 을 사용하여 어떤 함수의 type을 알 수 있습니다.

    > :t sum
    sum :: (Num a, Foldable t) => t a -> a

sum의 type에서 Num이나 Foldable은 나중에 다시 다루겠습니다. 여기서는 t a -> a 부분만 보면 됩니다. t a -> a 를 해석해 보면 입력(t a)으로는 '리스트 하나'를 받고 출력(a) 으로는 '값 하나' 를 내놓는 함수가 됩니다.

참고로 모든 함수는 반드시 소문자로 시작해야 합니다. 즉, 함수이름으로 Sum 은 불가능합니다. Haskell에서는 대소문자가 문법적으로 의미가 있습니다.

    > :t take
    take :: Int -> [a] -> [a]

take 함수는 입력으로 인자를 두 개 받는데 첫번째(Int)는 Int type의 인자이고 두 번째는 List([a]) 입니다. 그리고 출력으로는 List 하나([a])를 내놓습니다.

Haskell에서 type 은 모든 것에 있습니다.

    > :t [1,2,3]
    [1,2,3] :: Num t => [t]

[1,2,3] 은 각 요소가 숫자인 List 입니다.

함수형 프로그래밍의 가장 큰 특징은 함수가 first-class citizen 이라는 것입니다. 즉, 함수가 함수의 인자로 들어갈 수도 있고 함수 실행의 결과로도 나올 수 있습니다. 이러한 함수를 고차 함수 higher-order function 이라고 합니다. 대표적인 고차함수로는 map, filter, fold 가 있습니다.

    > :t map
    map :: (a -> b) -> [a] -> [b]
    > map (*2) [1,2,3]
    [2,4,6]

map 함수는 (a -> b) 꼴 함수 하나와 [a] 꼴 List 하나를 받아서 [b] 꼴 List 를 결과로 내놓는 함수 입니다.

    > :t filter
    filter :: (a -> Bool) -> [a] -> [a]
    > filter odd [1,2,3]
    [1,3]

filter 함수는 (a -> Bool) 꼴 함수 하나와 [a] 꼴 List 하나를 받아서 [a] 꼴 List 하나를 결과로 내놓는 함수입니다. (a -> Bool) 꼴 함수는 많이 등장하는 형태이어서 특별히 Predicate 이라고 부릅니다.

    > :t foldr
    foldr :: Foldable t => (a -> b -> b) -> b -> t a -> b
    > foldr (+) 0 [1,2,3]
    6

foldr 함수는 (a -> b -> b) 꼴 함수 하나와 b 꼴 값 하나, 그리고 t a 꼴 List 하나(사실 Foldable은 List 보다 좀 더 포괄적인 개념이지만 일단 여기서는 List에 대해서만 생각하기로 합니다)를 입력으로 받아서 b 꼴 값을 하나 내놓는 함수 입니다. foldr 함수는 이름 그대로 List처럼 여러 요소를 갖고 있는 자료형을 하나의 값으로 접는(fold) 일을 합니다. 여기서 끝에 붙은 r 은 right의 뜻으로 foldr은 fold하는 방향이 오른쪽에서 왼쪽으로 진행됩니다. fold가 왼쪽에서 오른쪽으로 진행되는 foldl 함수도 물론 있습니다.

    > :t foldl
    foldl :: Foldable t => (b -> a -> b) -> b -> t a -> b

foldr, foldl 함수는 for-loop 나 재귀를 더욱 추상화한 것입니다. 특히 foldr 함수는 다른 고차 함수들을 만들 수 있는 함수이므로 매우 중요합니다. 예를 들어 filter 함수를 재귀적으로 다음처럼 만들 수 있습니다.

    > :{
    Prelude| let my_filter f [] = []
    Prelude|     my_filter f (x:xs) = if (f x)
    Prelude|                          then x:(my_filter f xs)
    Prelude|                          else my_filter f xs
    Prelude| :}

위 코드에서 새로 나온 것들이 몇 개 있습니다. 우선, ghci 에서 여러 줄에 걸쳐 코드를 작성하려면 :{ 로 시작하고 :} 로 끝내면 됩니다. 이 때, :{ 와 :} 가 있는 줄에는 다른 것은 쓰지 않아야 합니다.
ghci에서 여러 줄에 걸쳐 함수를 정의하는 것은 사실 불편합니다. 그래서 이제부터는 여러 소스 파일을 작성하고 이를 ghci에서 불러와서 사용하겠습니다. Haskell 소스파일은 확장자가 .hs 로 끝납니다. 그리고 이렇게 작성한 파일을 ghci에서 불러올 때는 :load 명령 또는 단축명령 :l 을 사용합니다. 소스파일에서 함수 정의할 때는 let을 쓰지 않습니다.

    {-
      my_filter.hs
      Haskell에서 여러 줄 주석은 {- 로 시작하고 -} 로 끝납니다.
    -}
    my_filter:: (a -> Bool) -> [a] -> [a]
    my_filter f [] = []
    my_filter f (x:xs) = if (f x)
                         then x:(my_filter f xs)
                         else my_filter f xs

    > :l my_filter.hs
    [1 of 1] Compiling Main             ( my_filter.hs, interpreted )
    Ok, modules loaded: Main.

Haskell source 파일을 작성할 때는 off-side rule을 지켜야 합니다. 이는 축구의 off-side rule과 똑같은 맥락이며 들여쓰기를 할 때 계층을 맞추어주어야 합니다. 예를 들어 다음과 같이 작성하면 파일을 불러올 때 오류가 납니다.

    compareLength::String->String->Ordering
     compareLength x y = length x `compare` length y

그 이유는 같은 compareLength 함수의 type signature와 함수 정의부는 정의의 계층이 같기에 서로 들여쓰기 계층이 맞아야 하는데, 위에서는 compareLength함수 정의부가 그것의 type signature와 들여쓰기 깊이가 다르기 때문입니다. 참고로 들여쓰기를 지키는 대신 { 와 } 를 써서 명시적으로 묶어줄 수도 있습니다. Off-side rule을 가지는 프로그래밍 언어는 이외에도 Python, F# 등이 있습니다.

foldr 함수와 foldl 함수는 각각 foldr1, foldl1 이라는 자매 함수가 있는데, 이 함수들은 기본값(base)을 받지 않습니다. 즉, List에서 첫번째로 fold하는 원소를 기본값으로 삼습니다.

    > foldr1 (+) [1,2,3,4] -- 10
    > foldl1 (++) ["I","Love","You"] -- "ILoveYou"

그 다음, (x:xs) 와 같은 것을 pattern matiching이라고 합니다. 다음 코드를 보세요.

    > let (a:as) = [1,2,3]
    > a
    1
    > as
    [2,3]

즉, List [1,2,3] 을 (a:as) 꼴 패턴에 대응하여 각각 a 와 as 의 값을 정하는 것입니다.
다음의 코드에서 my\_filter 함수의 정의부가 두 번 등장하는 것도 pattern matching입니다.

    my_filter f [] = []
    my_filter f (x:xs) = if (f x)
                         then x:(my_filter f xs)
                         else my_filter f xs

다음으로 if..then..else 구문이 나왔습니다. 이는 자명하므로 설명하지 않겠습니다.

우리가 filter 함수를 재귀적으로 구현했는데, foldr 함수는 재귀를 보다 추상화한 함수이기 때문에 재귀적으로 구현할 수 있는 코드는 foldr 로도 구현할 수 있습니다. 이제 filter 함수를 foldr 로 구현해보겠습니다.

    > let my_filter f xs = foldr (\x base -> if (f x) then x:base else base) [] xs

재귀를 명시적으로 쓰지 않고도 filter 함수를 구현할 수 있었습니다. 그 이유는 foldr 이 재귀를 추상화한 함수이기 때문입니다.

사전 학습은 여기까지입니다. 다음 세 개의 숙제를 세미나 참석 전까지 제출해주시기 바랍니다. 숙제제출은 세미나 수료 요건 중 하나입니다.

숙제1) foldr 함수를 이용해서 sum 함수를 직접 구현해보세요.

    > let my_sum:: Num a => [a] -> a; my_sum xs = ?

숙제2) foldr 함수를 이용해서 map 함수를 직접 구현해보세요.

    > let my_map:: (a -> b) -> [a] -> [b]; my_map f xs = ?

숙제3) foldr 함수를 재귀적으로 직접 구현해보세요.

    my_foldr:: (a -> b -> b) -> b -> [a] -> b
    my_foldr f base [] = ?
    my_foldr f base (x:xs) = ?

## 첫 1시간
먼저 숙제를 함께 복기하겠습니다.

연습 1) foldl 함수를 재귀적으로 구현해보세요.

List에 대해 더 알아봅시다.
zip 함수가 있습니다. zip 함수는 이름 그대로 바지 지퍼처럼 두 개의 List의 각 원소들을 1:1 로 묶어줍니다.

    > :t zip
    zip :: [a] -> [b] -> [(a, b)]
    > zip [1,2,3] ['a','b','c']
    [(1,'a'),(2,'b'),(3,'c')]

zip 함수의 실행 결과에서 새로운 자료형이 등장합니다. 바로 Tuple 입니다. Haskell에서 List는 homogeneous 자료형입니다. [1,'a',"xyz"] 같은 것은 불가능합니다. 반면에 Tuple은 heterogenous 자료형입니다.

Tuple은 위의 예처럼 원소 두 개짜리 뿐만 아니라 (1,'a',True) 같은 세 개, 네 개 짜리도 가능합니다. 원소가 두 개짜리인 Tuple을 특별히 Pair라고 부르는데 다음과 같은 함수를 사용할 수 있습니다.

    > fst (1,'a')
    1
    > snd (1,'a')
    'a'

zipWith 란 함수도 있습니다. 이 함수는 Tuple로 만드는 대신 주어진 함수를 가지고 두 개 List의 각 원소에 대한 연산을 수행합니다.

    > zipWith (+) [1,2,3] [10,20,30,40]
    [11,22,33]

연습2) zipWith 를 재귀적으로 구현해 보세요.

    zipWith f [] _ = []
    zipWith f _ [] = []
    zipWith f (x:xs) (y:ys) = ?

List를 반복적으로 편리하게 만들어 주는 함수들이 있습니다.

    > replicate 3 1
    [1,1,1]
    > take 3 (repeat [1,2])
    [[1,2],[1,2],[1,2]]
    > take 3 (cycle [1,2])
    [1,2,1]

그런데 repeat와 cycle 함수를 보면 take 함수를 써서 일부 결과물만 가져오고 있습니다. 그 이유는 repeat 와 cycle 함수는 무한수열을 만들기 때문입니다. Haskell에서는 이렇게 무한수열을 아주 편하게 사용할 수 있는데 그 이유는 Haskell에서는 lazy evaluation 이 기본이기 때문입니다. Lazy evaluation에서는 필요할 때까지 expression을 평가하지 않습니다. List를 선언할 때 무한수열 형태로 선언할 수도 있습니다. List의 끝을 정해주지 않으면 무한수열이 됩니다.

    > take 3 [1..]
    [1,2,3]
    > take 5 [1,3..]
    [1,3,5,7,9]

Lazy evalution에서는 값이 필요할 때까지 expression은 expression그대로 남아 있습니다. 예를 들어 [1..] 라는 expression은 어떠한 값도 아닌 그냥 expression입니다. 그러다가 take 3 함수가 오면 그제서야 맨 앞 3개 요소를 값으로 평가합니다. 즉 1:2:3:[4..] 가 되어버립니다. [4..] 부분은 여전히 expression으로 남아있게 됩니다.

무한수열을 만드는 또 다른 함수는 iterate 입니다.

    > take 5 (iterate (\x -> x^2) 2)
    [2,4,16,256,65536]
    > take 5 (iterate (map (*2)) [1,2,3])
    [[1,2,3],[2,4,6],[4,8,12],[8,16,24],[16,32,48]]

위 코드에서 (\x -> x^2) 은 Lambda expression이라고 부르는 것으로 익명 함수를 편하게 정의할 수 있게 합니다.

연습3) iterate 함수를 재귀적으로 구현해 보세요.

    > let iterate f x = ?

연습4) Haskell의 lazy evaluation 덕분에 fibonacci 수열을 매우 간단하게 만들 수 있습니다. 다음 코드를 완성하세요.

    > let fib = 1:1:zipWith (+) ? ?

fold 함수가 여러 개의 값을 하나로 줄여버리는데 반해 scan 함수는 값을 계속 누적해 나갑니다. scanl 과 scanr 함수가 있습니다.

    > scanl (+) 0 [1..10]
    [0,1,3,6,10,15,21,28,36,45,55]
    > scanr (+) 0 [1..10]
    [55,54,52,49,45,40,34,27,19,10,0]

연습5) iterate 함수를 scanl을 써서 구현해 보세요.

    > let iterate f x = scanl ? ? ?

연습6) fibonacci 수열을 scanl을 써서 만들어보세요.

    > let fib = 1:scanl (+) ? ?

연습7) scanl을 foldl을 써서 만들어 보세요.

    > let scanl f base xs = foldl ? ? ?

List를 만드는 또 다른 방법으로는 List comprehension이 있습니다.

    > [x | x <- [1..10], odd x]
    [1,3,5,7,9]
    > [x*y | x <- [1..3], y <- [10,11]]
    [10,11,20,22,30,33]

List comprehension을 이용하여 isPrime 함수를 만들겠습니다.

    > isPrime n = 2 == length [d | d <- [1..n], n `mod` d == 0]
    > zip [1..] $ map isPrime [1..10]
    [(1,False),(2,True),(3,True),(4,False),(5,True),(6,False),(7,True),(8,False),(9,False),(10,False)]

새로운 문법이 나왔습니다. mod 함수는 modulo 연산자입니다. mod 7 2 의 결과는 1 입니다. 그런데 mod 함수와 같은 이항연산자는 보통 중위표기로 쓰는 것이 읽기 편합니다. 그래서 Haskell에서는 이항연산자를 중위표기법으로 쓸 때는 backtick 으로 감싸줍니다. `mod` 이런 식으로.
$ 연산자는 우선 순위가 가장 낮은 연산자 입니다. $ 연산자는 괄호를 쓰는 불편함을 덜기 위해 있습니다. 즉, 위의 코드에서 zip [1..] (map isPrime [1..10]) 라고 써야 할 코드가 $ 연산자를 이용해서 zip [1..] $ map isPrime [1..10] 으로 작성될 수 있었습니다.

이제 isPrime 함수를 써서 다음처럼 소수의 목록을 구할 수 있습니다.

    > let prime = filter isPrime [1..]
    > take 10 prime
    [2,3,5,7,11,13,17,19,23,29]

연습8) 방금 만든 prime 함수는 사실 비효율적입니다. iterate 함수와 다음의 sieve 함수를 이용하여 에라토스테네스의 체를 이용한 보다 빠른 소수생성 함수를 만드세요.

    > let sieve (p:xs) = [x|x<-xs, x `mod` p /= 0]
    > let prime = ?

## 두 번째 시간

"Hello, world!" 와 같은 문자열을 Haskell 에서는 String이라고 부릅니다. 이 String에 대한 작업을 수행하는 함수 중 하나로 words, unwords 가 있습니다.

    > :t words
    words :: String -> [String]
    > words "Hello, world!"
    ["Hello,","world!"]
    > unwords ["Jack","said","hello"]
    "Jack said hello"

그런데 "Hello, world!" 의 type은 String이 아니라 [Char] 로 나옵니다.

    > :t "Hello, world!"
    "Hello, world!" :: [Char]

그 이유는 Haskell에서 String은 별도의 type이 아니라 [Char]의 또 다른 이름입니다. 즉, String은 다음처럼 선언되어 있습니다.

    type String = [Char]

이러한 것을 type synonym이라고 부릅니다.

이번에는 실제로 새로운 type을 정의해 보겠습니다.

    > data Gender = Male | Female derivng (Show, Eq)

이렇게 하면 Gender 라는 새로운 type이 생겼고 해당 type의 값은 Male 또는 Female입니다. type을 만들 때는 반드시 대문자로 시작하여야 하고 type의 값도 반드시 대문자로 시작해야 합니다.
deriving이라는 새로운 문법이 나왔는데, 이는 typeclass란 것과 관련있습니다. deriving Show 라는 것은 Gender라는 type이 Show라는 typeclass에 속해있다는 것을 말하는 것으로 이렇게 써야 Male과 Female 이라는 값을 문자열로 출력 가능합니다. Eq의 경우 Gender type을 비교를 할 수 있는 type으로 만들기 위해 필요합니다. 여기서는 typeclass는 Java의 interface와 비슷하다고 생각하고 넘어갑니다. 뒤에서 더 다루겠습니다.

이제 새로 만든 Gender type을 이용하는 함수를 하나 만들겠습니다.

    sayHello:: Gender -> String
    sayHello gender
        | gender == Male = "Good morning, sir."
        | otherwise = "Good morning, ma'am."

새로운 문법이 나왔는데, 수직선(|)을 사용하여 조건 분기하는 이러한 문법을 guard 라고 부릅니다. guard 문법의 otherwise 부분은 if..then..else 구문의 else에 해당합니다.

또 다른 자료형을 만들어보겠습니다. 이번에 이진트리를 만들겠습니다.

    data BinTree a = Empty | Fork a (BinTree a) (BinTree a) deriving Show
    myTree = Fork 'a' (Fork 'b' Empty Empty) (Fork 'c' Empty (Fork 'd' Empty Empty))
    myTree2 = Fork 1 (Fork 2 Empty Empty) (Fork 3 Empty (Fork 4 Empty Empty))

BinTree 자료형에서 a 는 type parameter입니다. a 의 type에 의해 전체 Tree의 type이 결정됩니다.

    > :t myTree
    myTree :: Tree Char
    > :t myTree2
    myTree2 :: Num a => Tree a

새로운 문법이 나왔는데, => 부분은 Typeclass constraint라고 부르는 부분으로 type parameter 'a'가 어느 Typeclass에 속하는지를 밝히는 것입니다.

우리가 만든 이진 트리 자료형도 List에서 쓰던 map 같은 함수를 쓸 수 있으면 좋겠습니다. 가령 다음처럼.

    > import Data.Char
    > treeMap toUpper myTree
    Fork 'A' (Fork 'B' Empty Empty) (Fork 'C' Empty (Fork 'D' Empty Empty))

위의 코드에서 toUpper 함수는 소문자를 대문자로 바꾸어 주는 함수로 Data.Char 모듈에 있기 때문에 사용하려고 해당모듈을 import 하였습니다.

그런데 어떤 자료형에 map 같은 함수를 쓰는 것은 매우 쉽게 생각할 수 있고 또 자주 필요한 일입니다. 그래서 이처럼 어떤 자료형의 각 원소들의 값을 한꺼번에 바꿀 수 있는 자료형을 별도의 typeclass로 정의하고 있습니다. Fuctor라고 불리는 것이 바로 그것입니다.

    class Functor f where
        fmap :: (a -> b) -> f a -> f b

위의 코드는 Functor typeclass의 정의입니다. Typeclass 를 정의할 때는 위처럼 class 라는 키워드를 통해 합니다. 위의 코드에서 보이듯이 Functor typeclass 이기 위해서는 단 하나의 조건만 있으면 되는데, 바로 fmap 함수가 해당 자료형에 대하여 정의되어 있으면 됩니다. 우리는 이미 Functor 인 자료형을 하나 배웠습니다. 바로 List 입니다. List 에 대해 동작하는 map 함수의 type을 다시 확인해 봅시다.

    map :: (a -> b) -> [a] -> [b]

fmap 함수의 type에서 f 에 해당하는 부분을 List 표기로 바꾸면 그대로 map 함수의 type이 됨을 볼 수 있습니다. 어떤 자료형이 특정 typeclass이기 위해서는 어떤 자료형을 해당 typeclass의 instance로 선언하면 됩니다. List는 어떤 식으로 Functor의 instance로 선언되어 있는지 확인합시다.

    instance Functor [] where
        fmap = map

이를 통해 List에 대해서는 fmap 함수가 map 함수와 똑같이 동작함을 알 수 있습니다.

연습9) 우리가 만든 이진트리를 Functor로 만들어보세요.

    instance Functor BinTree where
        fmap f Empty = Empty
        fmap f (Fork a l r) = ?

이번에는 노드를 여러 개 가질 수 있는 Tree를 만들어보겠습니다.

    data RoseTree a = Branch a [RoseTree a] deriving Show

연습10) RoseTree를 Functor로 만들어보세요.

    instance Functor RoseTree where
        fmap f (Branch a ts) = ?

Tree 자료형은 map 뿐만 아니라 fold 하는 것도 자연스러운 자료형입니다. 이진 트리에 대하여 fold함수를 정의해 보겠습니다.

    foldBinTree f base Empty = base
    foldBinTree f base (Fork a l r) = f a v
        where v = foldBinTree f i l
              i = foldBinTree f base r

새로운 문법인 where 가 나왔습니다. where 는 중간값이 필요할 때 사용하는 구문입니다.

이번에는 RoseTree에 대한 fold함수를 정의해 보겠습니다.

    type Forest a = [RoseTree a]
    foldtree:: (a -> b -> c) -> ([c] -> b) -> RoseTree a -> c
    foldtree f g (Branch a ts) = f a v
        where v = foldforest f g ts
    foldforest:: (a -> b -> c) -> ([c] -> b) -> Forest a -> b
    foldforest f g ts = ?

연습11) 위의 foldforest 함수를 완성해 보세요.


## 세 번째 시간
List와 Tree 자료형은 모두 Folding이 자연스러운 자료형입니다. 이렇듯 Folding이 되는 자료형이 자주 생기기 때문에 Haskell에서는 Foldable이란 typeclass가 있습니다. Foldable typeclass의 정의를 보겠습니다.

    class Foldable t where
        foldMap :: Monoid m => (a -> m) -> t a -> m
        foldr :: (a -> b -> b) -> b -> t a -> b

어떤 자료형이 Foldable이기 위해서는 foldMap 함수나 foldr 함수 둘 중 하나만 구현하면 됩니다. 그런데 foldMap 함수를 보니 Monoid 라는 typeclass constraints가 붙어 있습니다. 그래서 Monoid에 대해 알아보겠습니다. Monoid typeclass는 Data.Monoid 모듈에 정의되어 있습니다.

    class Monoid m where
        mempty :: m
        mappend :: m -> m -> m
        mconcat :: [m] -> m
        mconcat = foldr mappend mempty

Monoid는 한 마디로 말해서 두 개가 하나가 될 수 있는 자료형을 뜻합니다. mappend 함수의 type이 이를 잘 설명해 주는 데 m -> m -> m 은 어떤 값 두 개를 받아서 하나를 내놓는 함수를 뜻합니다.
Monoid이기 위해서는 두 가지 요건이 있으면 되는데 하나는 항등원(mempty)이 있으면 되고, 다른 하나는 결합법칙이 성립하는 이항연산자(mappend)가 있으면 됩니다. mconcat 함수는 이 두개가 있으면 자동으로 얻을 수 있는 함수 입니다. 예를 들어 List는 Monoid입니다. List는 항등원 [] 가 있고,  결합법칙이 성립하는 이항연산자 ++ 이 있습니다.

    instance Monoid [a] where
        mempty = []
        mappend = (++)

Monoid는 triple(T, **\* **, e) 이라고도 정의하는데, 어떤 type T에 대하여 결합법칙을 만족하는 이항연산자 **\* **가 있고 항등원 *e*가 있음을 뜻합니다.

두 개를 하나로 만드는 연산을 반복해서 수행하다 보면 결국 여러 개의 값이 단 하나의 값으로 줄어들게 됩니다. 이 점이 바로 Monoid가 Foldable typeclass의 foldMap 함수에 등장하는 이유입니다.

이제 Foldable을 배웠으니까 과거처럼 Tree를 fold하는 함수를 직접 만들필요 없이 Tree를 Foldable의 instance로 만들면 Tree를 fold할 수 있게 됩니다. 먼저 이진 트리를 Foldable의 instance로 만들겠습니다.

    instance Foldable BinTree where
        foldMap f Empty = mempty
        foldMap f (Fork a l r) = f a `mappend` (foldMap f l) `mappend` (foldMap f r)

위의 구현을 보면 함수 f의 type은 a -> m 입니다. 즉, 함수 f의 실행결과는 Monoid가 나오므로 이를 mappend 함수에 적용시킬 수 있는 것입니다.

연습12) RoseTree를 Foldable의 instance로 만들어 보세요.

    instance Foldable RoseTree where
        foldMap f (Branch a ts) = ?

이제 다시 List에 관한 함수들을 마저 살펴보겠습니다. List에 대한 함수들은 Data.List 모듈에 있습니다.

    > import Data.List
    > takeWhile (<3) [1..5] -- [1,2]
    > dropWhile (<3) [1..5] -- [3,4,5]
    > group [1,2,2,3,3,2] -- [[1],[2,2],[3,3],[2]]
    > maximum [1,3,2] -- 3
    > minimum [3,1,2] -- 1
    > elem 1 [1,2,3] -- True
    > notElem 4 [1,2,3] -- True
    > nub [1,2,2,3,3,2] -- [1,2,3]
    > [1,2,3] !! 1 -- 2
    > inits [1,2,3] -- [[],[1],[1,2],[1,2,3]]
    > tails [1,2,3] -- [[1,2,3],[2,3],[3],[]]
    > splitAt 2 [1,2,3] -- ([1,2],[3])
    > sort [1,4,3,2,5] -- [1,2,3,4,5]
    > partition (>3) [1,4,3,2,5] -- ([4,5],[1,3,2])
    > span (>3) [5,1,4,3,2] -- ([5],[1,4,3,2])
    > break (>3) [1,4,3,2,5] -- ([1],[4,3,2,5])

연습13) max 함수와 min함수는 각각 이름 그대로 다음처럼 동작합니다.

    max 2 5 -- 5
    min 2 5 -- 2

max 함수를 이용하여 maximum 함수를 구현해 보세요. 마찬가지로 min 함수를 이용하여 minimum 함수도 구현해 보세요.

연습14) partition 함수를 구현해 보세요.

    partition :: (a -> Bool) -> [a] -> ([a], [a])
    partition p xs = ?

차집합, 합집합, 교집합의 기능을 수행하는 함수도 있습니다.

    > [1,2,3,4,5] \\ [2,4] -- [1,3,5]
    > union [1,2,3] [2,4] -- [1,2,3,4]
    > intersect [1,2,3] [2,4] -- [2]

sortOn 함수는 어떤 식으로 sort 를 할 지 정해줄 수 있습니다.

    > sortOn length [[1,2],[3],[4],[5,6,7],[8,9]] -- [[3],[4],[1,2],[8,9],[5,6,7]]

find 계열 함수들을 살펴봅시다.

    > :t find
    find :: Foldable t => (a -> Bool) -> t a -> Maybe a
    > find (=='a') "abcde"
    Just 'a'
    > find (=='f') "abcde"
    Nothing

    > :t findIndex
    findIndex :: (a -> Bool) -> [a] -> Maybe Int
    > findIndex (>7) [5..9]
    Just 3
    > findIndex (>10) [5..9]
    Nothing

    > :t elemIndex
    elemIndex :: Eq a => a -> [a] -> Maybe Int
    > elemIndex 'a' "abcde"
    Just 0
    > elemIndex 'f' "abcde"
    Nothing

이 함수들의 type에는 공통적으로 Maybe가 나옵니다. Maybe는 값이 있거나 없는 경우에 사용합니다. 보통 값이 없는 경우에 null check을 많이 합니다. 하지만 null check을 하는 것은 무척 오류가 생기기 쉽습니다. 오죽하면 null 을 처음으로 도입한 Tony Hoare가 자신이 null을 만든 것은 Billion Dollar Mistake라는 고백을 하기도 했습니다. Maybe와 같은 type은 이러한 것으로부터 자유롭습니다.

    data Maybe a = Nothing | Just a

이번 시간 마지막 내용으로 함수의 합성에 대해 알아보겠습니다.

Haskell에서 함수가 수학에서의 함수가 뜻하는 바와 똑같듯이 Haskell에서의 함수의 합성은 수학에서의 함수의 합성과 똑같습니다. 즉, 수학에서 두 개의 함수 f: x -> y 와 g: y -> z 가 있을 때 이 둘의 합성 함수는 g ○ f: x -> z 가 되듯이 Haskell에서 두 개의 함수 f:: a -> b 와 g:: b -> c의 합성 함수 g . f:: a -> c 가 됩니다. Haskell에서 함수 합성 연산자는 . (dot) 입니다.

    > :t (.)
    (.) :: (b -> c) -> (a -> b) -> a -> c
    > import Data.Char
    > :t chr
    chr :: Int -> Char
    > :t maximum
    maximum :: (Ord a, Foldable t) => t a -> a
    > :t chr . maxmimum
    > chr . maximum :: Foldable t => t Int -> Char

합성함수에 값을 적용했을 때 어떻게 나오는지 살펴봅시다.

    > map (negate . abs) [5, -3, -6, 7, -3, 2, -19, 24]
    [-5, -3, -6, -7, -3, -2, -19, -24]
    > map (negate.sum.tail) [[1..5],[3..6],[1..7]]
    [-14, -15, -27]

그런데 합성할 함수가 (+) 나 max 처럼 인자를 두 개 받는 함수이면 어떻게 해야 할까요? 그럴때는 partial application을 이용합니다. Partial application이 무엇인지 먼저 살펴봅시다.

    > :t (+)
    (+) :: Num a => a -> a -> a
    > let add5 = (+) 5
    > :t add5
    add5 :: Num a => a -> a
    > add5 7
    12

위에서 보듯 partial application 이란 인자 n개를 받는 함수가 있을 때 이 함수에 n보다 적은 갯수의 인자만을 먼저 일부 적용하는 것을 말합니다. 위에서 보듯 (+) 함수는 인자를 두 개 받는 함수인데 이에 인자 하나를 먼저 partial apply 한 결과인 add5 함수는 인자를 하나만 받는 함수가 되었습니다. Haskell에서는 함수의 partial application이 이처럼 언어차원에서 바로 지원이 되는데, 그 이유는 Haskell의 모든 함수는 curried function이기 때문입니다. Currying이라는 새로운 용어가 또 나왔습니다.

Currying이란 인자 n개를 받는 함수를 인자 1개를 받는 함수로 만드는 일을 말합니다. Haskell의 모든 함수는 curried function이라고 했습니다. 즉, (+) 함수는 사실 인자 두 개를 받아서 결과 하나를 내놓는 함수가 아니라 인자 하나를 받아서 "인자하나를 받아 결과를 내놓는 함수"를 결과로 내놓는 함수인 셈입니다. (+) 함수의 type을 이에 맞게 다시 써 보면 다음과 같습니다.

    (+):: a -> (a -> a)

이제 모든 함수가 curried function이기 때문에 function composition을 하는데 장애물은 없습니다. 다음처럼 partial application을 쓰면 됩니다.

    > (sum . replicate 5 . max 6.7) 8.9
    44.5

참고로 Currying이란 말은 미국의 수학자이자 논리학자 Haskell Curry의 이름에서 따 왔습니다. 우리가 배우고 있는 Haskell 프로그래밍 언어도 이 사람의 이름을 가져다 쓴 것입니다.

연습15) Data.List 모듈에 있는 nub 함수는 중복을 없애는 함수입니다. 그런데 이 함수는 시간복잡도가 O(N^2) 로 느린 함수입니다. 원소간 순서를 알 수 있는 List의 경우 이 보다 더 빠른 O(NlogN) 시간복잡도로 중복을 없앨 수 있습니다. map, head, group, sort 함수와 합수 합성을 적절히 이용하여 다음 함수를 만들어보세요. (참고로 영어 단어 nub은 essence를 뜻합니다)

    rmDuplicate::(Ord a) => [a] -> [a]
    rmDuplicate xs = ?

## 네 번째 시간

이번 시간에는 지금까지 배운 것들을 이용한 문제 풀이 연습을 해 보겠습니다.

연습16) 4백만 보다 작은 Fibonacci 숫자들 중 짝수들의 합을 구하는 함수를 만들어보세요. (projecteuler.net 문제2)

연습17) 세 자리 숫자의 곱으로 만들어지는 Palindrome 수 중에서 가장 큰 수를 구하는 함수를 만들어보세요. (projecteuler.net 문제4)

연습18) 피타고라스 triplet은 다음 두 가지 조건을 만족하는 자연수 세 개 입니다.

  >1) a < b < c

  >2) a^2 + b^2 = c^2

피타고라스 triplet중 a+b+c=1,000인 triplet은 딱 하나 있습니다. 이 triplet을 구하는 함수를 만들어보세요. (projecteuler.net 문제9)

다음 두 문제를 풀기 위해서는 몇 가지 더 알아야 할 내용이 있습니다. lines 함수는 String을 받아서 newline character를 구분자 삼아 List로 바꾸는 일을 합니다.

    > lines "abc\nxyz"
    ["abc","xyz"]

read 함수는 String을 특정 타입으로 바꿀 때 씁니다. 여기서는 Int로 바꾸었습니다.

    > read "52"::Int
    52
    > read "5.8"::Float
    5.8

파일을 읽고 쓰는 IO 처리는 Haskell에서는 do block안에서 합니다.

    main = do
        contents <- readFile "triangle1.txt"
        let triangle = map (map (\x -> read x::Int)) . map words . lines $ contents
        print triangle

이 코드를 t.hs 파일에 저장하고 ghc --make t.hs 로 컴파일하면 실행파일이 만들어집니다. 또는 ghc t 만 해도 됩니다.

연습19) 다음과 같은 삼각형꼴 숫자 배열에서 위에서 아래로 가는 경로 중 그 합이 가장 작은 경우는 23입니다.
<pre>
        <b>3</b>
       <b>7</b> 4
      2 <b>4</b> 6
     8 5 <b>9</b> 3
</pre>
다음 삼각형꼴 숫자배열에서 가장 작은 경로의 합을 구하는 함수를 만들어보세요. (projecteuler.net 문제18)
<pre>
<a href="triangle1.txt">triangle1.txt</a>
</pre>

연습20) 19번에서 만든 함수로 다음 삼각형꼴 숫자배열에서 가장 작은 경로의 합을 구해보세요. 실행시간이 너무 오래 걸린다면 효율적인 알고리즘을 고민해서 다시 작성해 보세요. (projecteuler.net 문제67)
<pre>
<a href="triangle2.txt">triangle2.txt</a>
</pre>

연습21) 4를 자연수의 덧셈으로 만들 수 있는 방법은 다음처럼 4개가 있습니다.

    3+1
    2+2
    2+1+1
    1+1+1+1

어떤 수 n을 자연수의 덧셈으로 만들 수 있는 방법의 가짓 수를 구하는 함수를 만들어보세요. 그 함수를 이용하여 100의 경우의 가짓수를 구해보세요.

## 다섯 번째 시간

Data.List 모듈에서 다루지 않은 함수 중 concatMap이 있습니다. 이 함수는 다음 처럼 동작합니다.

    > concatMap (\x -> replicate x x) [1,2,3]
    [1,2,2,3,3,3]

이름에서 드러나듯 concat 과 map 의 기능을 합친 것처럼 동작합니다.

Unix 계열 OS에 있는 wc utility를 Haskell로 한 번 만들어 봅시다.

먼저 command line utility이므로 console에 뭔가를 써야 합니다. 이 용도의 함수 중 하나는 이미 앞에서 나왔는데, 바로 print입니다. 비슷한 종류의 함수들로 putStr, putStrLn 이 있습니다.

    > print 9
    9
    > putStr "haha"
    Haha> putStrLn "hoho"
    hoho

이들 함수의 type을 확인해 보면 모두 출력이 IO () 인 것을 볼 수 있다. 이는 이들 함수가 IO에 뭔가를 기록하지만 함수 자체의 반환값은 () 로 아무것도 없음을 뜻합니다. () 은 void 라고 생각하시면 됩니다.

우리가 만들 wc utility의 기능을 생각해봅니다. 먼저 다음 세가지 옵션을 지원해야 합니다. 만약에 아무런 옵션도 주어지지 않으면, 이 세 가지 값을 모두 계산해서 보여주어야 합니다.

    -c - 문자 숫자 세기
    -w - 단어 숫자 세기
    -l - 줄 수 세기

그리고 wc utility는 여러 개의 파일들을 입력받아 각각 파일별 계산 결과와 모든 것의 합을 출력할 수 있어야 합니다. 만일 입력파일이 주어지지 않으면 stdin 으로부터의 입력에 대해 계산을 합니다.

이제 wc 함수의 type을 생각해 봅시다. wc 함수는 위의 여러 가지 옵션들과 함께 파일 경로 목록을 받아 뭔가 계산을 한 다음에 IO에 뭔가를 기록할 것입니다. 이를 바탕으로 wc 함수의 type을 써보면 다음과 같이 될 것입니다.

    wc:: Options -> [FilePath] -> IO ()

먼저 입력파일이 없을 때의 처리를 어떻게 할 것인지 생각해 봅시다. 다음과 같은 꼴이 되면 될 것 같습니다.

    wc options [] = do
      text <- getContents
      let count = getCount text
      printCount options count

먼저 getContents 함수를 통해 stdin으로부터의 입력을 가지고 오고, getCount 함수에서 문자수, 단어수, 줄수 등을 계산합니다. 그리고 마지막으로 주어진 options에 따라 형식을 갖추어 출력을 합니다.

다음으로 입력파일이 하나만 있을 때의 처리를 생각해 봅시다. 다음처럼 간단하게 하면 될 것 같습니다.

    wc options [FilePath] = do
      text <- readFile FilePath
      let count = getCount text
      printCount options count

마지막으로 입력 파일이 여러 개 있을 때의 처리를 생각해 봅시다.

## 여섯 번째 시간

## 더 읽을 거리
####범주론 Category Theory
Haskell에는 Monoid, Functor와 같은 익숙하지 않은 용어가 등장하는데, 이는 Haskell의 설계에 추상대수학의 한 분야인 Category Theory의 개념들을 일부 넣었기 때문입니다. 하지만 Haskell을 더 잘 알기 위해 Category Theory를 알아야 하는가? 라고 묻는다면 대답은 "절대 그렇지 않다" 입니다. Category Theory는 무척 방대한 학문이고 Haskell이 이로부터 가져온 개념들은 아주 아주 일부일 뿐입니다. Functor와 Monoid같은 용어가 Category Theory에서 왔다는 정도만 알면 충분합니다.
그러므로 여기서 소개하는 Category Theory는 가볍게 읽어보고 넘어가시기 바랍니다.

우리가 중고등학교 시절 수학책에서 가장 먼저 나오는 단원이 집합입니다. 그만큼 집합이 수학에서 중요하다는 소리인데, 범주론 소개 역시 집합에서 출발하겠습니다.

Set

Magma - Set + binary operation

Semigroup - Magma + associative binary operation(결합법칙을 만족하는 이항연산자)

Monoid - Semigroup + Identity(항등원)

Group - Monoid + Inverse(역원)

Abelian group - Group의 이항연산자가 communicative law(교환법칙) 까지 만족할 때

Ring - Abelian group + 두 번째 associative binary operation

이러한 것들을 Algebraic structure(대수적 구조) 라고 부릅니다. 여기 소개된 것 말고도 Field, Vector space 등 훨씬 더 많습니다.

Group의 예를 들어보겠습니다. 0부터 5까지 6개의 자연수를 원소로 갖는 집합 ℤ<sub>6</sub>과 이항 연산자 additon of modular 6 를 묶어서 <ℤ<sub>6</sub>, +<sub>6</sub>> 라고 할 때 이는 Group입니다.

Category theory란 이러한 Algebraic structure들 간의 관계를 연구하는 수학의 한 분야입니다. 몇 가지 개념을 더하면 모든 Algebraic structure는 Category라는 것으로 승급될 수 있습니다. Functor란 Category들 간의 관계를 뜻하는데, A 라는 Category와 B라는 Category간에는 C 라는 Functor 관계가 있다 정도의 개념으로 이해하면 됩니다. Haskell에서 List가 Functor 인 것도 List의 모든 원소들이 다른 종류의 것들로 바뀔 수 있기에 그렇다고 생각하시면 됩니다.

## License
Eclipse Public License
