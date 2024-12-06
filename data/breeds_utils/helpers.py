# ImageNet label mappings
def get_label_mapping(dataset_name, ranges):
    if dataset_name == "imagenet":
        label_mapping = None
    elif dataset_name == "restricted_imagenet":

        def label_mapping(class_to_idx):
            return restricted_label_mapping(class_to_idx, ranges=ranges)

    elif dataset_name == "custom_imagenet":

        def label_mapping(class_to_idx):
            return custom_label_mapping(class_to_idx, ranges=ranges)

    else:
        raise ValueError("No such dataset_name %s" % dataset_name)

    return label_mapping


def restricted_label_mapping(class_to_idx, ranges):
    range_sets = [set(range(s, e + 1)) for s, e in ranges]

    # add wildcard
    # range_sets.append(set(range(0, 1002)))
    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(range_sets):
            if idx in range_set:
                mapping[class_name] = new_idx
        # assert class_name in mapping
    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping


def custom_label_mapping(class_to_idx, ranges):

    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(ranges):
            if idx in range_set:
                mapping[class_name] = new_idx

    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping
