from collections import deque
# create a list of prime numbers
prime_numbers = [2, 3, 5, 7,6]
prime_numbersss = [4,6,8,10]
p = deque(prime_numbers)

# reverse the order of list elements
for i in range(len(prime_numbersss)):
    p.popleft()
prime_numbersss.reverse()
for elem in prime_numbersss:
    p.appendleft(elem)

print(p)
print(p.popleft())
print('Reversed List:', prime_numbers)

# Output: Reversed List: [7, 5, 3, 2]