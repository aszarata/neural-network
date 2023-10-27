import random

def generate_table(rows):
    data = []
    for _ in range(rows):
        num1 = random.randint(1, 100)

        num2 = random.randint(1, 100)

        data.append([num1, num2, 2*num1 + 2*num2 + 5])

    return data

def save_to_file(data, filename):
    with open(filename, 'w') as file:
        for row in data:
            file.write(','.join(map(str, row)) + '\n')

if __name__ == "__main__":
    table_data = generate_table(1000)  # Change 5 to the number of rows you want
    save_to_file(table_data, 'generated_table.csv')  # Change 'generated_table.csv' to your desired filename
