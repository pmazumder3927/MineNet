#v0.1
import os
import sys
import nbtlib
from nbtlib import File
from nbtlib.tag import *
from nbtlib import parse_nbt
#from nbtlib import serialize_tag
from nbtlib import CompoundSchema
from nbtlib import schema
import numpy as np

globalPalette = []
schemX = 4
schemY = 8
schemZ = 4

Structure = schema('Structure', {
    'DataVersion': Int,
    'author': String,
    'size': List[Int],
    'palette': List[schema('State', {
        'Name': String,
        'Properties': Compound,
    })],
    'blocks': List[schema('Block', {
        'state': Int,
        'pos': List[Int],
        'nbt': Compound,
    })],
    'entities': List[schema('Entity', {
        'pos': List[Double],
        'blockPos': List[Int],
        'nbt': Compound,
    })],
})


class StructureFile(File, schema('StructureFileSchema', {'': Structure})):
    def __init__(self, structure_data=None):
        super().__init__({'': structure_data or {}})
        self.gzipped = True
    @classmethod
    def load(cls, filename, gzipped=True):
        return super().load(filename, gzipped)


#def generate_structure(author, size, palette,):

def load_structure_data(structurePath, data):
    nbt_file = nbtlib.load(structurePath)
    nbt_data = nbt_file.root[data]


def load_structure_blocks(structurePaths):

    structPaths = os.listdir(structurePaths)
    structures = []
    for path in structPaths:
        structures.append(nbtlib.load(structurePaths + '/' + path))

    for structure in range(len(structures)):
        nbt_file = structures[structure]
        nbt_palette = nbt_file.root['palette']
        #print(nbt_file)
        for key in nbt_palette:
            if key not in globalPalette:
                globalPalette.append(key)


        outputArr = np.zeros((len(structures), schemX, schemY, schemZ))
    for i in range(len(structures)):
        nbt_file = structures[i]
        nbt_data = nbt_file.root['blocks']
        nbt_palette = nbt_file.root['palette']
        converted_blocks = np.zeros((schemX, schemY, schemZ))
        for block in nbt_data:
            converted_blocks[block['pos'][0],
                             block['pos'][1],
                             block['pos'][2]] = convert_palette(block['state'], nbt_palette, globalPalette)

        outputArr[i] = converted_blocks
        """
    f = open('palettes/globalPalette.txt', 'w+')
    print(globalPalette)
    for element in globalPalette:
        f.writelines(element['Name'] + '\n')"""
    return outputArr


def convert_palette(block_state, original_palette, new_palette):
    return new_palette.index(original_palette[block_state])


def create_nbt_from_3d(blocks, epoch):
    globalDict = (nbtlib.tag.List)(globalPalette)
    blockArr = []
    for structure in blocks:
        for i in range(schemX):
            for j in range(schemY):
                for k in range(schemZ):
                    block = {
                        'state': structure[i, j, k],
                        'pos': [i, j, k]
                    }
                    blockArr.append(block)

        new_structure = Structure({
            'DataVersion': 1139,
            'author': 'danny',
            'size': [schemX, schemY, schemZ],
            'palette': globalDict,
            'blocks': blockArr,
            'entities': [],
        })

        structure_file = StructureFile(new_structure)
        structure_file.save("schem{}.nbt".format(epoch))



def load_test_set(filepath):
    nbt_file = nbtlib.load(filepath)
    return nbt_file.root['blocks']

#structfiles = ['input_nbt_files/birchcottage.nbt', 'input_nbt_files/cottage.nbt', 'input_nbt_files/workshop.nbt']

#create_nbt_from_3d(load_structure_blocks(structfiles), 'output_nbt_files/schem')