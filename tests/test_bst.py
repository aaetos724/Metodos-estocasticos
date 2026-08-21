from src.bst.tree import BinarySearchTree


def _sample_tree():
    tree = BinarySearchTree()
    for key in [50, 30, 70, 20, 40, 60, 80]:
        tree.insert(key, f"record-{key}")
    return tree


def test_insert_and_inorder_is_sorted():
    tree = _sample_tree()
    keys = [node.key for node in tree.inorder()]
    assert keys == sorted(keys)


def test_search_finds_existing_key():
    tree = _sample_tree()
    node = tree.search(40)
    assert node is not None
    assert node.record == "record-40"


def test_search_missing_key_returns_none():
    tree = _sample_tree()
    assert tree.search(999) is None


def test_count_and_height():
    tree = _sample_tree()
    assert tree.count() == 7
    assert tree.height() == 3  # balanced by construction


def test_delete_leaf_node():
    tree = _sample_tree()
    tree.delete(20)
    assert tree.search(20) is None
    assert tree.count() == 6


def test_delete_node_with_two_children_keeps_bst_property():
    tree = _sample_tree()
    tree.delete(30)  # has children 20 and 40
    keys = [node.key for node in tree.inorder()]
    assert keys == sorted(keys)
    assert tree.search(30) is None
    assert tree.count() == 6


def test_delete_root_with_two_children():
    tree = _sample_tree()
    tree.delete(50)
    keys = [node.key for node in tree.inorder()]
    assert keys == sorted(keys)
    assert tree.search(50) is None


def test_preorder_and_postorder_lengths_match_count():
    tree = _sample_tree()
    assert len(tree.preorder()) == tree.count()
    assert len(tree.postorder()) == tree.count()


def test_len_and_iter():
    tree = _sample_tree()
    assert len(tree) == 7
    assert [n.key for n in tree] == sorted(n.key for n in tree)


def test_empty_tree():
    tree = BinarySearchTree()
    assert tree.count() == 0
    assert tree.height() == 0
    assert tree.search(1) is None
