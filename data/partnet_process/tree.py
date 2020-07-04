import json

class TreeNode(object):
    def __init__(self, text, id, depth, parent=None, children=None):
        self.text = text
        self.id = id
        self.depth = depth
        self.children = children
        self.parent = parent
        self.objs = None

    @staticmethod
    def from_json(path):
        with open(path, 'r') as fp:
            hier = json.load(fp)

        root = construct_tree(hier[0], depth=0)
        return root

    def __repr__(self):
        child_id = None if self.children is None else [x.id for x in self.children]
        par_id = None if self.parent is None else self.parent.id
        return "text: {}, ID: {}, depth: {}, parent: {}, children: {}".format(self.text, self.id, self.depth, par_id, child_id)

    def print_hier(self, depth=0):
        prefix = "    " * (depth - 1) + "|---" * min(depth, 1)
        s = "text: {} ID: {}".format(self.text, self.id)
        if self.objs is not None:
            s += " objs: {}".format(self.objs)
        print(prefix + s)
        if self.children is not None:
            for child in self.children:
                child.print_hier(depth + 1)

    def query_node_by_id(self, id):
        """query node by given id"""
        if self.id == id:
            return self
        if self.children is not None:
            for child in self.children:
                res = child.query_node_by_id(id)
                if res is not None:
                    return res
        return None

    def query_parent_id(self, depth=1):
        """query its parent id at given depth"""
        if self.depth == depth:
            return self.id
        if self.parent is not None:
            return self.parent.query_parent_id(depth)
        return None

    def query_id_by_depth(self, depth):
        if self.depth == depth:
            return [self.id]
        elif self.depth > depth:
            return []
        else:
            res = []
            if self.children is not None:
                for child in self.children:
                    child_res = child.query_id_by_depth(depth)
                    res.extend(child_res)
            return res

    def collect_objs(self):
        if self.objs is not None:
            return self.objs
        else:
            res = []
            if self.children is not None:
                for child in self.children:
                    child_res = child.collect_objs()
                    res.extend(child_res)
            return res


def construct_tree(item, depth=0):
    text = item['text']
    id = item['id']
    node = TreeNode(text, id, depth)
    if 'objs' in item.keys():
        node.objs = item['objs']
    if 'children' in item.keys():
        children = []
        for child in item['children']:
            child_node = construct_tree(child, depth + 1)
            child_node.parent = node

            children.append(child_node)
    else:
        children = None
    node.children = children

    return node


if __name__ == '__main__':
    path = "/home/megaBeast/Desktop/partnet_data/PartNet/38037/result.json"
    root = TreeNode.from_json(path)
    root.print_hier()

    node = root.query_node_by_id(5)
    print(node)

    print(node.collect_objs())

    print(root.query_id_by_depth(depth=1))

    par_d1 = node.query_parent_id(depth=1)
    print(par_d1)
