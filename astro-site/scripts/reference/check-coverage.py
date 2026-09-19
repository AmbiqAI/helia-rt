"""Compare the public manifest and extracted members with Doxygen independently."""
import json
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

xml_dir, manifest_path, model_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
model = json.loads(model_path.read_text())
compounds = {}
declarations = {}
for entry in ET.parse(xml_dir / 'index.xml').getroot().findall('compound'):
    if entry.attrib['kind'] in ('file', 'namespace', 'group'):
        root = ET.parse(xml_dir / (entry.attrib['refid'] + '.xml')).getroot().find('compounddef')
        for member in root.findall('./sectiondef/memberdef'):
            declarations[member.attrib['id']] = member
    if entry.attrib['kind'] in ('class', 'struct'):
        root = ET.parse(xml_dir / (entry.attrib['refid'] + '.xml')).getroot().find('compounddef')
        compounds[root.findtext('compoundname')] = root
symbols = {symbol['id']: symbol for module in model['modules'] for symbol in module['symbols']}
checked = 0
for entry in manifest:
    compound = compounds.get(entry['symbol'])
    if compound is None:
        symbol = symbols[entry['symbol']]
        candidates = [member for member in declarations.values() if member.findtext('name') == entry['name'] and member.attrib['kind'] == {'function': 'function', 'macro': 'define', 'enum': 'enum'}.get(symbol['kind'], symbol['kind'])]
        if len(candidates) != 1:
            raise SystemExit(f"Expected one XML declaration for {entry['symbol']}; found {len(candidates)}")
        if symbol['kind'] == 'enum':
            expected_values = [member.findtext('name') for member in candidates[0].findall('enumvalue')]
            actual_values = [member['name'] for member in symbol['members']]
            if expected_values != actual_values:
                raise SystemExit(f"Enumerator coverage mismatch for {entry['symbol']}")
            checked += len(expected_values)
        continue
    expected = []
    for section in compound.findall('sectiondef'):
        if 'private' in section.attrib['kind'] or 'package' in section.attrib['kind']:
            continue
        for member in section.findall('memberdef'):
            if member.attrib.get('prot') not in ('private', 'package'):
                expected.append(member.findtext('name'))
    actual = [member['name'] for member in symbols[entry['symbol']].get('members', [])]
    if entry['name'] == 'MicroMutableOpResolver':
        parent = next(module for module in model['modules'] if module['name'] == entry['name'])
        actual.extend(symbol['name'] for module in parent.get('submodules', []) for symbol in module['symbols'])
    if Counter(expected) != Counter(actual):
        raise SystemExit(f"API coverage mismatch for {entry['symbol']}: XML={Counter(expected)}, extracted={Counter(actual)}")
    checked += len(expected)
print(f'Independent XML coverage: {checked} public/protected member declarations preserved.')
