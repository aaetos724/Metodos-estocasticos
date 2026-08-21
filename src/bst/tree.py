"""Binary Search Tree keyed on a numeric ID, storing arbitrary records.

Ported from a coursework notebook (student registry keyed on
`matricula`). Refactored here as a small generic-ish class instead of
one hardcoded to students: `insert`/`search`/`delete` operate on
(key, record) pairs, so the same tree works for any record type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Node:
    key: int
    record: Any
    left: "Node | None" = None
    right: "Node | None" = None


class BinarySearchTree:
    def __init__(self):
        self.root: Node | None = None

    def insert(self, key: int, record: Any) -> None:
        self.root = self._insert(self.root, key, record)

    def _insert(self, node: Node | None, key: int, record: Any) -> Node:
        if node is None:
            return Node(key, record)
        if key < node.key:
            node.left = self._insert(node.left, key, record)
        elif key > node.key:
            node.right = self._insert(node.right, key, record)
        else:
            node.record = record  # key exists: update in place
        return node

    def search(self, key: int) -> Node | None:
        node = self.root
        while node is not None and node.key != key:
            node = node.left if key < node.key else node.right
        return node

    def delete(self, key: int) -> None:
        self.root = self._delete(self.root, key)

    def _delete(self, node: Node | None, key: int) -> Node | None:
        if node is None:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            successor = node.right
            while successor.left:
                successor = successor.left
            node.key = successor.key
            node.record = successor.record
            node.right = self._delete(node.right, successor.key)
        return node

    def inorder(self) -> list[Node]:
        return self._inorder(self.root)

    def _inorder(self, node: Node | None) -> list[Node]:
        if node is None:
            return []
        return self._inorder(node.left) + [node] + self._inorder(node.right)

    def preorder(self) -> list[Node]:
        return self._preorder(self.root)

    def _preorder(self, node: Node | None) -> list[Node]:
        if node is None:
            return []
        return [node] + self._preorder(node.left) + self._preorder(node.right)

    def postorder(self) -> list[Node]:
        return self._postorder(self.root)

    def _postorder(self, node: Node | None) -> list[Node]:
        if node is None:
            return []
        return self._postorder(node.left) + self._postorder(node.right) + [node]

    def count(self) -> int:
        return self._count(self.root)

    def _count(self, node: Node | None) -> int:
        if node is None:
            return 0
        return 1 + self._count(node.left) + self._count(node.right)

    def height(self) -> int:
        return self._height(self.root)

    def _height(self, node: Node | None) -> int:
        if node is None:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))

    def __iter__(self) -> Iterator[Node]:
        return iter(self.inorder())

    def __len__(self) -> int:
        return self.count()
