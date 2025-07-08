"""
Example file to demonstrate python.analysis.typeCheckingMode effects
"""

def add_numbers(a, b):
    """Add two numbers together."""
    return a + b

def process_data(data):
    """Process some data."""
    result = []
    for item in data:
        result.append(item * 2)
    return result

# Type issues that different modes will catch differently:

# 1. This will cause issues in strict mode (no type hints)
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

# 2. This will cause type mismatch in basic+ modes  
def main():
    # Passing string where number expected
    result1 = add_numbers("hello", 5)  # Type error
    
    # Calling function with wrong type
    average = calculate_average("not_a_list")  # Type error
    
    # Using result incorrectly
    final = result1 + 10  # This might work but is type-unsafe
    
    print(f"Results: {result1}, {average}, {final}")

if __name__ == "__main__":
    main()
