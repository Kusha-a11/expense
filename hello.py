import methods as m
import json
import copy


#######################################################

# Получить имя пользователя и получить данные через API
# Убрать бесполезные данные

#username = "sudesh2911"
username = "tyrange"
#username = "tyrange"
all_games_List = m.getGames(username)
m.filterList(all_games_List, username)

#######################################################

# Импортировать файл с дебютами в формате JSON
# Построить деревья решений для партий белыми и чёрными

openings = json.load(open('openings2.json'))
WhiteTree = m.buildOpeningTree(openings)
BlackTree = copy.deepcopy(WhiteTree)

#######################################################

# Вывести самые популярные дебюты

opening_freq_white = copy.deepcopy(openings)
opening_freq_black = copy.deepcopy(openings)
for x in opening_freq_white:
    opening_freq_white[x] = 0
    opening_freq_black[x] = 0

# Передать все партии в дерево решений

m.convertPGN(all_games_List, WhiteTree, BlackTree, opening_freq_white, opening_freq_black)

# Вывести частоту всех дебютов

for k,v in opening_freq_white.items():
    if v != 0:
        pass
        print(k,v)

for k,v in opening_freq_black.items():
    if v != 0:
        print(k,v)

#######################################################

# Возвращает, сколько раз вы были в этой позиции
# p — PGN, преобразованный в список ходов

p = "e2e4 d7d5".split(" ")
w = WhiteTree.traverse(p, WhiteTree.root)
b = BlackTree.traverse(p, BlackTree.root)

print(w.attributes)
print(b.attributes)

#######################################################