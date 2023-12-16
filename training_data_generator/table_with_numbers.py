import random

def generate_table(rows):
    data = []
    for _ in range(rows):
        num1 = random.randint(-5, 5)

        num2 = random.randint(-10, 10)

        num3 = random.randint(0, 5)

        data.append([num1, num2, num3, 1 if num1 + num2 * num3 > 0 else 0])

    return data

def save_to_file(data, filename):
    with open(filename, 'w') as file:
        for row in data:
            file.write(','.join(map(str, row)) + '\n')

if __name__ == "__main__":
    table_data = generate_table(2000)
    save_to_file(table_data, 'training_data/add_numbers.csv')


