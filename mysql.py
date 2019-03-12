import pymysql
t = ["2625", "7242", "7747", "4246", "4743", "1022", "4345", "2324", "2524", "7062", "1713", "8070", "1343"]
db = pymysql.connect(host="localhost",user="root", password="zhenmafan", db="chess")
cursor = db.cursor()

cursor.execute("DROP TABLE IF EXISTS chessModel")

sql1 = """CREATE TABLE chessModel (
			STEP  CHAR(4) NOT NULL
		)"""
cursor.execute(sql1)

for i in t:
	sql2 = "INSERT INTO chessModel(STEP) VALUES (\'%s\')" % (i)
	print(sql2)
	try:
		cursor.execute(sql2)
		db.commit()
	except:
		db.rollback()


#data = cursor.fetchone()
#print ("Database version : %s " % data)

db.close()