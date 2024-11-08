from collections import Counter


class Node:
    def __init__(self, symbol, freq, left=None, right=None) -> None:
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


class HuffmanTree:
    def __init__(self) -> None:
        self.codes = {}
        self.tree = None

    def build(self, text):
        if len(text.strip()) == 0:
            return None

        queue = []
        for key, value in Counter(text).items():
            queue.append(Node(key, value))

        # Отсортируем очередь по приоритетеу
        queue.sort(key=lambda x: x.freq, reverse=True)
        # Начинаем строить дерево
        while len(queue) > 1:
            # Достанем 2 наиболее приоритетных элемента
            left = queue.pop()
            right = queue.pop()
            freq = left.freq + right.freq

            # Создаем узел
            queue.append(Node(None, freq, left, right))

            # Обновим порядок
            queue.sort(key=lambda x: x.freq, reverse=True)

        self.tree = queue[0]
        self.update_codes(self.tree)

        return self.encode(text)

    def update_codes(self, root, path=''):
        if root is None:
            return

        if root.is_leaf:
            self.codes[root.symbol] = path if path else '1'

        self.update_codes(root.left, path+'0')
        self.update_codes(root.right, path+'1')

    def encode(self, text: str):
        result = ''
        encoded = self.codes.keys()
        for char in text:
            if char not in encoded:
                raise ValueError(f'Символ {char} отсутствует в таблице')

            result += self.codes[char]

        return result

    def decode(self, code):
        index = -1
        text = ''
        while index < len(code) - 1:
            index, char = self._decode(self.tree, index, code)
            text += char

        return text

    def _decode(self, root, index, code):
        if root is None:
            return index

        if root.is_leaf:
            return index, root.symbol

        index += 1
        root = root.left if code[index] == '0' else root.right
        return self._decode(root, index, code)
