︠c323ff11-66c7-4133-a99b-c6ec8e0a5c68︠
def memo(f):
    cache = {}
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError:
            return f(*args)
    return _f
︡3c64a186-e49d-4335-a309-67b29c0d9c07︡
︠ed602ca9-d872-4779-abcd-92b24d0a1983︠
%md
The split function returns all ways to insert one
plus sign into a binary integer
︡d06e5480-10a3-42a6-bca0-6521acd1cda3︡{"html":"<p>The split function returns all ways to insert one\nplus sign into a binary integer</p>\n"}︡
︠23eeeceb-a2de-47de-a123-f5b794f934fa︠
def split(string,key = 0):
    result = []
    for i in range(1,len(string)):
        result.append([string[:i],string[i:]])
    return result
︡795efdce-25c2-4fcc-9042-e329ae32ebee︡
︠c646c87a-acbc-4ec5-b952-d743c301dc0c︠
for v in split(12.binary()):
    print v
︡fe6bc3da-5cd7-4c9f-a664-876c176641b9︡{"stdout":"['1', '100']\n['11', '00']\n['110', '0']\n"}︡
︠4a0eae5d-fb42-42e3-b879-8574b682afb2︠

︠ab4463be-ec0b-4098-973e-916aabcd4542︠
@memo
def possibleSums(n):
    result = [n]
    if n == 0 or n==1:
        pass
    else:
        for v in split(n.binary()):
            w  = [Integer(a,base=2) for a in v]
            for i in possibleSums(w[0]):
                for j in possibleSums(w[1]):
                    if i+j not in result:
                        result.append(i+j)
    return result
︡7ec244d9-6292-4068-8843-6acbfa429ec1︡
︠4ed22b99-04c2-4ca2-99ef-0a79c58a069c︠
possibleSums(232)
︡e9134212-fbb4-4010-9775-1b946547aa81︡{"stdout":"[232, 105, 42, 11, 4, 5, 7, 12, 6, 8, 22, 15, 9, 14, 27, 53, 43, 13, 23, 16, 18, 29, 58, 116]\n"}︡
︠3537051b-1714-405e-85f6-c8206deae130︠
def minSteps(n):
    """
    OUTPUT:
    c , the minimal steps to transform n to 1
    v , a list of binary strings of length (c+1) that realizes the transformation.
    """
    if n == 1: return (0,[1])
    else:
        seq = [n.binary()]
        v =[m for m in possibleSums(n) if m != n]
        w  = [(a,minSteps(a)) for a in v]
        # print w
        t = min(w, key = lambda x:x[1][0])
        seq += t[1][1]
        if len(seq) == t[1][0]+2:
            return (t[1][0]+ 1, seq)
        else:
            print 'sequence length does not match'
︡b91d011f-e458-464e-bd3c-8b470930cebb︡
︠897a560b-316b-4bdb-8cb8-7e991e2edddf︠
minSteps(232)
︡348cabfb-725d-4973-9b71-8bb5e785151f︡{"stdout":"(2, ['11101000', '100', 1])\n"}︡
︠f1773660-f9ca-4a26-9996-dc4a7b1f94c0︠
for i in range(2,300):
    a = minSteps(Integer(i))
    if a[0] > 2:
        print i,a
︡f7cd65e5-535c-418e-8031-5eb179bf4428︡
︠4c1b550a-267f-41d5-81ec-3843d36c8419︠
for i in range(15,100):
    a = minSteps(Integer(i))
    print a[1]
︡4337a577-1c63-4504-b562-113c3a3cfe24︡{"stdout":"['1111', '1000', 1]\n['10000', 1]\n['10001', '10', 1]\n['10010', '10', 1]\n['10011', '100', 1]\n['10100', '10', 1]\n['10101', '100', 1]\n['10110', '100', 1]\n['10111', '1000', 1]\n['11000', '10', 1]\n['11001', '100', 1]\n['11010', '100', 1]\n['11011', '100', 1]\n['11100', '100', 1]\n['11101', '100', 1]\n['11110', '1000', 1]\n['11111', '10000', 1]\n['100000', 1]\n['100001', '10', 1]\n['100010', '10', 1]\n['100011', '100', 1]\n['100100', '10', 1]\n['100101', '100', 1]\n['100110', '100', 1]\n['100111', '1000', 1]\n['101000', '10', 1]\n['101001', '100', 1]\n['101010', '100', 1]\n['101011', '100', 1]\n['101100', '100', 1]\n['101101', '100', 1]\n['101110', '1000', 1]\n['101111', '10000', 1]\n['110000', '10', 1]\n['110001', '100', 1]\n['110010', '100', 1]\n['110011', '100', 1]\n['110100', '100', 1]\n['110101', '100', 1]\n['110110', '1000', 1]\n['110111', '1000', 1]\n['111000', '100', 1]\n['111001', '100', 1]\n['111010', '100', 1]\n['111011', '1000', 1]\n['111100', '100', 1]\n['111101', '1000', 1]\n['111110', '10000', 1]\n['111111', '100000', 1]\n['1000000', 1]\n['1000001', '10', 1]\n['1000010', '10', 1]\n['1000011', '100', 1]\n['1000100', '10', 1]\n['1000101', '100', 1]\n['1000110', '100', 1]\n['1000111', '1000', 1]\n['1001000', '10', 1]\n['1001001', '100', 1]\n['1001010', '100', 1]\n['1001011', '100', 1]\n['1001100', '100', 1]\n['1001101', '100', 1]\n['1001110', '1000', 1]\n['1001111', '10000', 1]\n['1010000', '10', 1]\n['1010001', '100', 1]\n['1010010', '100', 1]\n['1010011', '100', 1]\n['1010100', '100', 1]\n['1010101', '100', 1]\n['1010110', '1000', 1]\n['1010111', '1000', 1]\n['1011000', '100', 1]\n['1011001', '100', 1]\n['1011010', '100', 1]\n['1011011', '1000', 1]\n['1011100', '100', 1]\n['1011101', '1000', 1]"}︡
︠49b38da3-6bc9-4679-be1e-f8ab4fd11517︠
