import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from argparse import Action, ArgumentParser

class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple: The expanded list or tuple from the string.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)

def poly2obb_np(rbox):
    res = np.empty([*rbox.shape[:-1], 5])
    rboxes = rbox.reshape(-1, 4, 2).astype(np.float32) 
    for i, rbox in enumerate(rboxes):
        (x, y), (w, h), angle = cv2.minAreaRect(rbox[None])
        if w >= h:
            angle = -angle
        else:
            w, h = h, w
            angle = -90 - angle
        theta = angle / 180 * np.pi
        res[i, 0] = x 
        res[i, 1] = y
        res[i, 2] = w
        res[i, 3] = h
        res[i, 4] = theta

    return res

def collect_results(results_dir, class_ignore=[]):
    results_per_image = {}
    for result_file in glob(os.path.join(results_dir, "*.txt")):
        result_name = os.path.splitext(os.path.basename(result_file))[0]
        class_name = result_name
        if class_name in class_ignore:
            continue
        
        with open(result_file, "r") as fr:
            lines = fr.readlines()
        for line in lines:
            line = line.strip()
            linelist= line.split(" ")
            image_id = linelist[0]
            prob = float(linelist[1])
            box = list(map(float, linelist[2:10]))
            box.append(prob)
            box = np.asarray(box, dtype=np.float32)
            if image_id in results_per_image:
                results = results_per_image[image_id]
                if class_name in results:
                    results[class_name] = np.concatenate((results[class_name], box[None]), axis=0)
                else:
                    results[class_name] = box[None]
            else:
                results_per_image[image_id] = {class_name: box[None]}
    return results_per_image



