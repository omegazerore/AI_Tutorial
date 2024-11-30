"""
In cryptography, there is a very simple encryption method where each character in the plaintext is shifted by a certain integer
𝐾
K to produce the characters in the ciphertext. Both plaintext and ciphertext characters are within the printable ASCII range. For example, if
𝐾
=
2
K=2, then "apple" would become "crrng" after encryption. Decryption simply reverses this process.

The problem is to take a ciphertext string as input and output the plaintext by following the decryption method described above.

As for the value of
𝐾
K in this task, you’ll need to deduce it based on the Sample Input and Sample Output. It’s quite simple!
"""

print(ord("*"))
print(ord("1"))

print(ord("C"))
print(ord("J"))
# AttributeError: 'str' object has no attribute 'ord'

K = 7

"""
ord -> char
"""

K = 7

# encoded = input()
encoded = "1JKJ'pz'{ol'{yhklthyr'vm'{ol'Jvu{yvs'Kh{h'Jvywvyh{pvu5"

decoded = ""

for str_ in encoded:
    decoded += chr(ord(str_)-K)
    print(decoded)

# print(decoded)