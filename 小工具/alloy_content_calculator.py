from numpy import linalg,array


target = array([3.3,0.7,2.1,1])
print('------------------根据提示按顺序输入各个材料的中的元素含量------------------')
print('------------------------------------------------------------------------')
print('------------含量为百分数，不要输入%，如2%则输入2，不包含需要输入0------------')
print('------------------------------------------------------------------------')
print('--------不同含量之间用空格隔开，输入完成后按空格进入下一元素含量的输入----------')
print('------------------------------------------------------------------------')

v = []

Fe = input('请按顺序输入各个材料中的铁含量：').split(' ')
for i in range(len(Fe)):
    Fe[i] = float(Fe[i])
v.append(Fe)
Mn = input('请按顺序输入各个材料中的锰含量：').split(' ')
for i in range(len(Mn)):
    Mn[i] = float(Mn[i])
v.append(Mn)
Si = input('请按顺序输入各个材料中的硅含量：').split(' ')
for i in range(len(Si)):
    Si[i] = float(Si[i])
v.append(Si)
v.append([1,1,1,1])
r = array(v)

result = linalg.solve(r,target)
print('------------------------------------------------------------------------')
print('------------------------------------------------------------------------')
print('计算完成')
for i in range(len(result)):
    print('第%s种材料的质量占总质量的%.4f%%' % (i+1,result[i]*100))
print('------------------------------------------------------------------------')
print('------------------------------------------------------------------------')

input('按任意键退出程序')