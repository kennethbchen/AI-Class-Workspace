import pandas


df = pandas.read_csv("data/video_games_sales.csv")

names = df["Name"]

print(names)