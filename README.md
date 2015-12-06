# 하루만 배워보는 Haskell Programming

Haskell Programming을 딱 하루만 배워보자.

6시간 동안 Haskell을 배워서 Haskell로 실용적인 도구를 만들어보자.

## 사전 학습
Haskell platform을 설치한다.
가장 기본 자료형인 List에 대해 알아보자. List는 대괄호로 표시한다.
> [1,2,3]
[1,2,3]

이제 List에 사용할 수 있는 기본 함수들을 살펴보자.
> sum [1,2,3]
6
> product [1,2,3]
6
> length [1,2,3]
3
> head [1,2,3]
1
> tail [1,2,3]
[2,3]
> init [1,2,3]
[1,2]
> last [1,2,3]
3
> take 2 [1,2,3]
[1,2]
> drop 2 [1,2,3]
[3]
> 0:[1,2,3]
[0,1,2,3]
> [1,2,3] ++ [4,5,6]
[1,2,3,4,5,6]
> maximum [1,2,3]
3
> minimum [1,2,3]
1
> reverse [1,2,3]
[3,2,1]

Haskell 에서는 모든 동작에 있어서 함수가 중심이 됩니다. 위의 예에서 볼 수 있들이 List 에어떤 함수를 적용할 때는 "함수 List" 꼴로 적용할 함수가 먼저 나오고 그 뒤에 List가 나옵니다.
가령 sum 함수의 경우 "sum [1,2,3]" 꼴로 호출이 되었습니다. 즉, sum 함수는 List를 하나 받아서 그List의 원소들의 합을 구하는 함수 입니다. 즉 sum함수의 입력은 "List 하나"이고 출력은 "값 하나" 꼴이 됩니다. 이렇듯 함수의 입력과 출력관계를 정의할 수 있는데 이를 함수의 type이라고 합니다. ghci에서는 :type 명령(혹은 :t) 을 사용하여 어떤 함수의 type을 알 수 있습니다.
>:t sum
sum :: (Num a, Foldable t) => t a -> a
sum의 type에서 Num이나 Foldable은 나중에 다시 다루겠습니다. 여기서는 t a -> a 부분만 보면 됩니다. t a -> a 를 해석해 보면 입력(t a)으로는 '리스트 하나'를 받고 출력(a) 으로는 '값 하나' 를 내놓는 함수가 됩니다.
> t take
take :: Int -> [a] -> [a]
take 함수는 입력으로 인자를 두 개 받는데 첫번째(Int)는 Int type의 인자이고 두 번째는 List([a]) 입니다. 그리고 출력으로는 List 하나([a])를 내놓습니다.


## 첫 1시간

## 두 번째 시간

## 세 번째 시간

## 

## License
Eclipse Public License
