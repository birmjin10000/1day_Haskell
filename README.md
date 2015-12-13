# 하루만 배워보는 Haskell Programming

Haskell Programming을 딱 하루만 배워봅시자.

6시간 동안 Haskell을 배워서 Haskell로 실용적인 도구를 만들어봅시다.

## 사전 학습
Haskell platform을 설치합니다. 설치후 ghci를 실행합니다. Haskell의 기본 자료형은 List입니다. List는 대괄호로 표시합니다. Haskell에서 한 줄 주석은 수직선 두 개로 표시합니다.

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

함수형 프로그래밍의 가장 큰 특징은 함수가 first-class citizen 이라는 것입니다. 즉, 함수가 함수의 인자로 들어갈 수도 있고 함수 실행의 결과물로도 나올 수 있습니다. 이러한 함수를 고차 함수 higher-order function 이라고 합니다. 대표적인 고차함수로는 map, filter, fold 가 있습니다.

    > :t map
    map :: (a -> b) -> [a] -> [b]
    > map (*2) [1,2,3]
    [2,4,6]

map 함수는 (a -> b) 꼴 함수 하나와 [a] 꼴 List 하나를 받아서 [b] 꼴 List 함수를 결과로 내놓는 함수 입니다.

    > :t filter
    filter :: (a -> Bool) -> [a] -> [a]
    > filter odd [1,2,3]
    [1,3]

filter 함수는 (a -> Bool) 꼴 함수 하나와 [a] 꼴 List 하나를 받아서 [a] 꼴 List 하나를 결과로 내놓는 함수입니다. (a -> Bool) 꼴 함수는 많이 등장하는 형태이어서 특별히 Predicate 이라고 부릅니다.

    > :t foldr
    foldr :: Foldable t => (a -> b -> b) -> b -> t a -> b
    > foldr (+) 0 [1,2,3]
    6

foldr 함수는 (a -> b -> b) 꼴 함수 하나와 b 꼴 값 하나, 그리고 t a 꼴 List 하나(사실 Foldable은 List 보다 좀 더 포괄적인 개념이지만 일단 여기서는 List에 대해서만 생각하기로 합니다)를 입력으로 받아서 b 꼴 값을 하나 내놓는 함수 입니다. foldr 함수는 이름 그대로 List와 값은 여러 요소를 갖고 있는 자료형을 하나의 값으로 접는(fold) 일을 합니다. 여기서 끝에 붙은 r 은 right의 뜻으로 foldr은 fold하는 방향이 오른쪽에서 왼쪽으로 진행됩니다. fold가 왼쪽에서 오른쪽으로 진행되는 foldl 함수도 물론 있습니다.

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

    > :l my_filter.hs
    [1 of 1] Compiling Main             ( my_filter.hs, interpreted )
    Ok, modules loaded: Main.

그 다음, (x:xs) 와 같은 것을 pattern matiching이라고 합니다. 다음 코드를 보세요.

    > let (a:as) = [1,2,3]
    > a
    1
    > as
    [2,3]

즉, List [1,2,3] 을 (a:as) 꼴 패턴에 대응하여 각각 a 와 as 의 값을 정하는 것입니다.
다음으로 if..then..else 구문이 나왔습니다. 이는 자명하므로 설명하지 않겠습니다.

우리가 filter 함수를 재귀적으로 구현했는데, foldr 함수는 재귀를 보다 추상화한 함수이기 때문에 재귀적으로 구현할 수 있는 코드는 foldr 로도 구현할 수 있습니다. 이제 filter 함수를 foldr 로 구현해보겠습니다.

    > let my_filter f xs = foldr (\x base -> if (f x) then x:base else base) [] xs

재귀를 명시적으로 쓰지 않고도 filter 함수를 구현할 수 있었습니다. 그 이유는 foldr 이 재귀를 추상화한 함수이기 때문입니다.

사전 학습이 여기까지입니다. 다음 세 개의 숙제를 세미나 참석 전까지 제출해주시기 바랍니다. 숙제제출은 세미나 수료 요건 중 하나입니다.

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
zip 함수가 있습니다. zip 함수는 이름 그대로 바지 지퍼처럼 두 개의 List의 각 원소들을 1:1 대로 묶어줍니다.

    > :t zip
    zip :: [a] -> [b] -> [(a, b)]
    > zip [1,2,3] ['a','b','c']
    [(1,'a'),(2,'b'),(3,'c')]

zip 함수의 실행 결과에서 새로운 자료형이 등장합니다. 바로 Tuple 입니다. Haskell에서 List는 homogeneous 자료형입니다. [1,'a',"xyz"] 같은 것은 불가능합니다. 반면에 Tuple은 heterogenous 자료형입니다.

Tuple은 위의 예처럼 원소 두 개짜리 뿐만 아니라 (1,'a',True) 같은 세 개, 네 개 짜리도 가능합니다. 원소가 두 개짜리인 Tuple을 Pair라고 부르는데 다음과 같은 함수를 사용할 수 있습니다.

    > fst (1,'a')
    1
    > snd (1,'a')
    'a'

zipWith 란 함수도 있습니다. 이 함수는 Tuple로 만드는 대신 주어진 함수를 가지고 두 개 List의 각 원소에 대한 연산을 수행합니다.

    > zipWith (+) [1,2,3] [10,20,30]
    [11,22,33]

List를 반복적으로 편리하게 만들어 주는 함수들이 있습니다.

    > replicate 3 1
    [1,1,1]
    > take 3 (repeat [1,2])
    [[1,2],[1,2],[1,2]]
    > take 3 (cycle [1,2])
    [1,2,1]

그런데 repeat와 cycle 함수를 보면 take 함수를 써서 일부 결과물만 가져오고 있습니다. 그 이유는 repeat 와 cycle 함수는 무한수열을 만들기 때문입니다. Haskell에서는 이렇게 무한수열을 아주 편하게 사용할 수 있는데 그 이유는 Haskell이 lazy evaluation 이 기본이기 때문입니다. Lazy evaluation에서는 필요할 때까지 expression을 평가하지 않습니다. List를 선언할 때 무한수열 형태로 선언할 수도 있습니다. List의 끝을 정해주지 않으면 무한수열이 됩니다.

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

연습2) iterate 함수를 재귀적으로 구현해 보세요.

    > let iterate f x = ?

연습3) Haskell의 lazy evaluation 덕분에 fibonacci 수열을 매우 간단하게 만들 수 있습니다. 다음 코드를 완성하세요.

    > let fib = 1:1:zipWith (+) ? ?

fold 함수가 여러 개의 값을 하나로 줄여버리는데 반해 scan 함수는 값을 계속 누적해 나갑니다. scanl 과 scanr 함수가 있습니다.

    > scanl (+) 0 [1..10]
    [0,1,3,6,10,15,21,28,36,45,55]
    > scanr (+) 0 [1..10]
    [55,54,52,49,45,40,34,27,19,10,0]

연습4) iterate 함수를 scanl을 써서 구현해 보세요.

    > let iterate f x = scanl ? ? ?

연습5) fibonacci 수열을 scanl을 써서 만들어보세요.

    > let fib = 1:scanl (+) ? ?

연습6) scanl을 foldl을 써서 만들어 보세요.

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

연습7) 방금 만든 prime 함수는 사실 비효율적입니다. iterate 함수와 다음의 sieve 함수를 이용하여 에라토스테네스의 체를 이용한 보다 빠른 소수생성 함수를 만드세요.

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

이렇게 하면 Gender 라는 새로운 type이 생겼고 해당 type의 값은 Male 또는 Female입니다. type을 만들 때는 반드시 대문자로 시작하여야 하고 type의 값도 반드시 대문자로 시작해야 합니다. deriving이라는 새로운 문법이 나왔는데, 이는 typeclass란 것과 관련있습니다. deriving Show 라는 것은 Gender라는 type이 Show라는 typeclass에 속해있다는 것을 말하는 것으로 이렇게 써야 Male과 Female 이라는 값을 문자열로 출력 가능합니다. Eq의 경우 비교를 할 수 있는 값으로 만들기 위해 필요합니다. 여기서는 typeclass는 Java의 interface와 비슷하다고 생각하고 넘어갑니다. 뒤에서 더 다루겠습니다.

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

fmap 함수의 type에서 f 에 해당하는 부분을 List로 바꾸면 그대로 map 함수의 type이 됨을 볼 수 있다. 어떤 자료형이 특정 typeclass이기 위해서는 어떤 자료형을 해당 typeclass의 instance로 선언하면 된다. List는 어떤 식으로 Functor의 instance로 선언되어 있는지 확인하자.

    instance Functor [] where
        fmap = map

이를 통해 List에 대해서는 fmap 함수가 map 함수와 똑같이 동작함을 알 수 있습니다.

연습8) 우리가 만든 이진트리를 Functor로 만들어보세요.

    instance Functor BinTree where
        fmap f Empty = Empty
        fmap f (Fork a l r) = ?

이번에는 노드를 여러 개 가질 수 있는 Tree를 만들어보겠습니다.

    data RoseTree a = Branch a [RoseTree a] deriving Show

연습9) RoseTree를 Functor로 만들어보세요.

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
    foldtree:: (a -> b -> b) -> ([b] -> b) -> b -> RoseTree a -> b
    foldtree f g base (Branch a ts) = f a v
        where v = foldforest f g base ts
    foldforest:: (a -> b -> b) -> ([b] -> b) -> b -> Forest a -> b
    foldforest f g base ts = ?
     
연습문제10) 위의 foldforest 함수를 완성해 보세요.


## 세 번째 시간
List와 Tree 자료형은 모두 Folding이 자연스러운 자료형입니다. 이렇듯 Folding이 되는 자료형이 자주 생기기 때문에 Haskell에서는 Foldable이란 typeclass가 있습니다. Foldable typeclass의 정의를 보겠습니다.

    class Foldable t where
        foldMap :: Monoid m => (a -> m) -> t a -> m
        foldr :: (a -> b -> b) -> b -> t a -> b

어떤 자료형이 Foldable이기 위해서는 foldMap 함수나 foldr 함수 둘 중 하나만 구현하면 됩니다. 그런데 foldMap 함수를 보니 Monoid 라는 typeclass constraints가 붙어 있스니다. 그래서 Monoid에 대해 알아보겠습니다.


## 네 번째 시간 

## License
Eclipse Public License
