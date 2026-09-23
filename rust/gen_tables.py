"""Regenerate src/tables.rs from evaluation.py. Run from the repo root:

    python3 rust/gen_tables.py

The Rust classical player (src/classical.rs) is a port of eval_pos(); the
piece-square tables are copied from the Python by this script rather than by
hand, so the two cannot drift on a typo. test_rl.py checks the scores agree.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluation as e

def rows(table, per=8):
  return '\n'.join('    ' + ', '.join(str(v) for v in table[k:k + per]) + ','
                   for k in range(0, len(table), per))

parts = ['// Generated from evaluation.py by rust/gen_tables.py -- do not edit.\n']
for name, tables in (('MIDGAME', e.midgame_piece_pos), ('ENDGAME', e.endgame_piece_pos)):
  parts.append(f'pub const {name}: [[i32; 64]; 6] = [')
  for t in tables:
    parts.append('  [\n' + rows(t) + '\n  ],')
  parts.append('];\n')
parts.append('pub const CENTER_DISTANCE: [i32; 64] = [\n' + rows(e.center_distance) + '\n];\n')
parts.append(f'pub const PASSED_MG: [i32; 8] = {e.passed_mg};\npub const PASSED_EG: [i32; 8] = {e.passed_eg};\n')
parts.append(f'pub const PHASE_VALS: [i32; 6] = {e.phase_vals};\npub const MATERIAL: [i32; 6] = {e.material_vals};\n')
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'tables.rs')
open(out, 'w').write('\n'.join(parts))
print('wrote', out)
