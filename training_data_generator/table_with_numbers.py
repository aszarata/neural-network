import random

def generate_table(rows):
    data = []
    for _ in range(rows):
        num1 = random.randint(1, 100)

        num2 = random.randint(1, 100)

        data.append([num1, num2, num1 + num2])

    return data

def save_to_file(data, filename):
    with open(filename, 'w') as file:
        for row in data:
            file.write(','.join(map(str, row)) + '\n')

if __name__ == "__main__":
    table_data = generate_table(20000)
    save_to_file(table_data, 'training_data/add_numbers.csv')


