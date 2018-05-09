import csv

forms_prod = []
forms_uat = []
forms_supfa460 = []

with open('PRODN.csv', 'r') as f1:
    next(f1)
    reader1 = csv.reader(f1)
    for row1 in reader1:
        forms_prod.append(row1[0])

    # unique itemss
    forms_prod = list(set(forms_prod))
    # sort items, sort() has no return
    forms_prod.sort()

with open('UAT.csv', 'r') as f2:
    next(f2)
    reader2 = csv.reader(f2)
    for row2 in reader2:
        forms_uat.append(row2[0])

    # unique itemss
    forms_uat = list(set(forms_uat))
    forms_uat.sort()

with open('SUPFA460.csv', 'r') as f3:
    next(f3)
    reader3 = csv.reader(f3)
    for row3 in reader3:
        forms_supfa460.append(row3[0])

    # unique itemss
    forms_supfa460 = list(set(forms_supfa460))
    forms_supfa460.sort()



print("Forms missing in SCB PRODUCT:\n", list(set(forms_supfa460) - set(forms_prod)))
print("Forms missing in SCB UAT:\n", list(set(forms_supfa460) - set(forms_uat)))
print("SCB UAT equals SCB PRODUCT: ", set(forms_prod) == set(forms_uat))


