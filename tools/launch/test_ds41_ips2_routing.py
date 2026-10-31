#!/usr/bin/env python3
"""The DS41 main fabric's second rail must reach g6m4 as --ips2, not --pf-ips2."""

import ast
from pathlib import Path
import unittest


SRC = Path(__file__).with_name('supervisor.py')


def boot_rail_assignment(source: str) -> ast.AST:
    module = ast.parse(source)
    supervisor = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'Supervisor')
    boot = next(node for node in supervisor.body if isinstance(node, ast.FunctionDef) and node.name == 'boot')
    assigns = [node for node in ast.walk(boot) if isinstance(node, ast.Assign) and
               any(isinstance(target, ast.Name) and target.id == 'ips2' for target in node.targets)]
    if len(assigns) != 1:
        raise AssertionError(f'expected one boot ips2 assignment, got {len(assigns)}')
    argv = [node for node in ast.walk(boot) if isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == 'argv' for target in node.targets)]
    if not argv or not any(isinstance(node, ast.Name) and node.id == 'ips2' for node in ast.walk(argv[0].value)):
        raise AssertionError('engine argv no longer passes the selected second-rail flag')
    return assigns[0].value


def routed_flag(source: str, model_name: str) -> str:
    expr = boot_rail_assignment(source)
    class Model:
        name = model_name
    class Models:
        @staticmethod
        def of(_):
            return Model()
    return eval(compile(ast.Expression(expr), str(SRC), 'eval'), {'models': Models, 'o': object()})


class TestDs41Ips2Routing(unittest.TestCase):
    def test_main_fabric_second_rail_for_ds41(self):
        src = SRC.read_text()
        self.assertEqual(routed_flag(src, 'ds41'), '--ips2')
        self.assertEqual(routed_flag(src, 'glm53'), '--ips2')
        self.assertEqual(routed_flag(src, 'glm53f'), '--pf-ips2')

    def test_old_ds41_pf_only_route_fails(self):
        src = SRC.read_text()
        old = "models.of(o).name == 'glm53'"
        new = "models.of(o).name in ('glm53', 'ds41')"
        self.assertEqual(src.count(new), 1)
        self.assertEqual(routed_flag(src.replace(new, old, 1), 'ds41'), '--pf-ips2')


if __name__ == '__main__':
    unittest.main()
