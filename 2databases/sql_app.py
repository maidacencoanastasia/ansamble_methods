import mysql.connector

# Замените значения переменных соответствующими данными вашей базы данных
host = "127.0.0.1" #port 3306
user = "root"
password = "root"
database = "medic"

# Создание подключения
try:
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    print("Connection established")
except mysql.connector.Error as err:
    print("Something went wrong: {}".format(err))
# Создание объекта курсора
cursor = conn.cursor()
'''
doctors_data = [
    ("John Doe", "1980-01-15", "M", "john.doe@email.com", "+1234567890", "john_doe", "password123"),
    ("Jane Smith", "1985-05-20", "F", "jane.smith@email.com", "+9876543210", "jane_smith", "securepass"),
    # Add more rows as needed
]
# SQL statement for inserting data into the doctors table
insert_query = "INSERT INTO doctors (fullName, dateOfBirth, gender, email, phone, login, password) VALUES (%s, %s, %s, %s, %s, %s, %s)"

# Insert data into the doctors table
cursor.executemany(insert_query, doctors_data)

# Commit the changes to the database
conn.commit()
'''
# Пример выполнения SQL-запроса
cursor.execute("SELECT * FROM doctors")

# Получение результатов запроса
results = cursor.fetchall()

# Вывод результатов
for row in results:
    print(row)

# Закрытие курсора и соединения
cursor.close()
conn.close()
