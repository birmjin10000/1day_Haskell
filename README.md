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

Haskell 에서는 모든 동작에 있어서 함수가 중심이 됩니다. 위의 예에서 볼 수 있들이 List 에 어떤 함수를 적용할 때는 "함수 List" 꼴로 적용할 함수가 먼저 나오고 그 뒤에 List가 나옵니다.
가령 sum 함수의 경우 "sum [1,2,3]" 꼴로 호출이 되었습니다. 즉, sum 함수는 List를 하나 받아서 그List의 원소들의 합을 구하는 함수 입니다. 즉 sum함수의 입력은 "List 하나"이고 출력은 "값 하나" 꼴이 됩니다. 이렇듯 함수의 입력과 출력관계를 정의할 수 있는데 이를 함수의 type이라고 합니다. ghci에서는 :type 명령(혹은 :t) 을 사용하여 어떤 함수의 type을 알 수 있습니다.

    > :t sum
    sum :: (Num a, Foldable t) => t a -> a

sum의 type에서 Num이나 Foldable은 나중에 다시 다루겠습니다. 여기서는 t a -> a 부분만 보면 됩니다. t a -> a 를 해석해 보면 입력(t a)으로는 '리스트 하나'를 받고 출력(a) 으로는 '값 하나' 를 내놓는 함수가 됩니다.

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

그런데 repeat와 cycle 함수를 보면 take 함수를 써서 일부 결과물만 가져오고 있습니다. 그 이유는 repeat 와 cycle 함수는 무한수열을 만들기 때문입니다. Haskell에서는 이렇게 무한수열을 아주 편하게 사용할 수 있는데 그 이유는 Haskell이 lazy evaluation 이 기본이기 때문입니다. Lazy evaluation에서는 필요할 때까지 expression을 평가하지 않습니다. 무한수열을 만드는 또 다른 함수는 iterate 입니다.

    > take 5 (iterate (\x -> x^2) 2)
    [2,4,16,256,65536]

위 코드에서 (\x -> x^2) 은 Lambda expression이라고 부르는 것으로 익명 함수를 편하게 정의할 수 있게 합니다.

연습2) Haskell의 lazy evaluation 덕분에 fibonacci 수열을 매우 간단하게 만들 수 있습니다. 다음 코드를 완성하세요.

    > let fib = 1:1:zipWith (+) ? ?

fold 함수가 여러 개의 값을 하나로 줄여버리는데 반해 scan 함수는 값을 계속 누적해 나갑니다. scanl 과 scanr 함수가 있습니다.

    > scanl (+) 0 [1..10]
    [0,1,3,6,10,15,21,28,36,45,55]

연습3) 아까zipWith 함수를 써서 만든 fibonacci 수열을 이번에는 scanl을 써서 만들어보세요.

    > let fib = 1:scanl (+) ? ?


## 두 번째 시간

## 세 번째 시간

## 

## License
Eclipse Public License
