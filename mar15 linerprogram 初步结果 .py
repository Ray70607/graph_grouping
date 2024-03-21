import random

def generate_array(length, equal_distribution=True):
    if length <= 0:
        return []

    random_array = []
    if equal_distribution:
        half_length = length // 2
        random_array = ['A'] * half_length + ['B'] * (length - half_length)
        random.shuffle(random_array)
    else:
        for _ in range(length):
            random_char = random.choice(['A', 'B'])
            random_array.append(random_char)

    return random_array

# Example usage:
array_length = 6
equal_distribution = 1
result = generate_array(array_length, equal_distribution)
#result=['B', 'A', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'A']

def calculate_f(result):
    array_length = len(result)
    f = [0] * array_length
    
    for i in range(array_length):
        for j in range(i):
            d=0
            if result[i]==result[j]:
                d=1
            else:
                d=-1
            f[i]+=d/(i-j)**2
        for j in range(i+1,array_length):
            d=0
            if result[i]==result[j]:
                d=1
            else:
                d=-1
            d*=-1
            f[i]+=d/(i-j)**2
        
    return f


def get_short_array_difference(f):
    result = []
    array_length = len(f)
    for i in range(array_length - 1):
        result.append(f[i] - f[i + 1])
    return result


def max_abs_index(arr):
    max_abs = float('-inf')  # Initialize max absolute value to negative infinity
    max_abs_index = None     # Initialize index of max absolute value to None
    
    for i in range(len(arr)):
        abs_value = abs(arr[i])
        if abs_value > max_abs:
            max_abs = abs_value
            max_abs_index = i
            
    return max_abs_index
def max_idx(arr):
    if not arr:
            return None  # Return None if the array is empty
    
    max_val = arr[0]  # Initialize max_val to the first element
    max_index = 0  # Initialize max_index to 0
    
    for i in range(1, len(arr)):
            if arr[i] > max_val:
                max_val = arr[i]
                max_index = i
                
    return max_index

while(1):
    f=calculate_f(result)
    f2=get_short_array_difference(f)
    index = max_idx(f2)
    
    print(result)
    print(f)
    print(f2)
    print("Index of element with the largest  value:", index)
    
    #f2[index]=0
    if(f2[index]<2):
        break
    else:
        result[index],result[index+1]=result[index+1],result[index]
        
print(result)