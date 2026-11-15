#!/usr/bin/env python3
"""Сборка BACKEND.yaml из модульного дерева specifications/backend.

Модули:
    openapi.yaml          — info, servers, tags, security, шаблоны SDK
    common/*.yaml         — parameters, errors(+responses, headers), security
    schemas/*.yaml        — компонентные схемы по доменным контекстам D1
    paths/*.yaml          — операции по доменным контекстам (медиатор tag)

Сборщик:
  * объединяет компоненты (conflict = ошибка, а не «побеждает последний»);
  * переписывает $ref вида "<file>.yaml#/Name" в "#/components/<section>/Name";
  * разворачивает $ref на подпуть чужого файла (<file>.yaml#/A/properties/B)
    во встроенную схему, потому что внутренние $ref не переносимы между файлами;
  * проверяет, что все $ref разрешаются, все operationId уникальны, каждая
    операция описана в тегах и в info.description.

Запуск:  python specifications/scripts/build_backend_spec.py [--check]
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1] / "backend"
OUTPUT = Path(__file__).resolve().parents[1] / "BACKEND.yaml"

SECTION_PREFIX = {
    "schemas/": "schemas",
    "common/parameters": "parameters",
    "common/errors": "schemas",
    "common/_responses": "responses",
    "common/security": "securitySchemes",
}

REF_RE = re.compile(r"^([\w./_-]+\.yaml)#/(.+)$")


class SpecError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SpecError(f"{path}: ожидался YAML-объект")
    return data


def _module_path(name: str) -> Path:
    """Файл модуля по имени из $ref (schemas/x.yaml, common/x.yaml)."""
    candidate = (ROOT / name).resolve()
    if not candidate.is_file():
        raise SpecError(f"модуль не найден: {name}")
    return candidate


def _section_for(module: Path) -> str:
    rel = module.resolve().relative_to(ROOT.resolve()).as_posix()
    if rel.startswith("schemas/"):
        return "schemas"
    if rel.startswith("common/"):
        stem = rel[len("common/"): -len(".yaml")]
        mapping = {
            "parameters": "parameters",
            "errors": "schemas",
            "_responses": "responses",
            "security": "securitySchemes",
            "_headers": "headers",
        }
        # _responses.yaml содержит и responses, и headers — обрабатывается отдельно
        return mapping[stem]
    raise SpecError(f"неизвестный модуль компонентов: {rel}")


def _register(
    components: dict[str, dict[str, Any]], module: Path, section: str, name: str, value: Any
) -> str:
    bucket = components.setdefault(section, {})
    if name in bucket and bucket[name] != value:
        raise SpecError(
            f"{module.name}: конфликт определения {section}/{name} "
            "— компонент объявлен в двух модулях с разным содержимым"
        )
    bucket[name] = value
    return f"#/components/{section}/{name}"


def _lookup(module: Path, name: str, section: str, registry: dict[Path, dict[str, Any]]) -> tuple[str, Any]:
    """Компонент модуля в нужной секции: явно указанной или найденной по имени."""
    data = registry[module]
    if name in (data.get(section) or {}):
        return section, data[section][name]
    for candidate in ("schemas", "parameters", "responses", "headers", "securitySchemes"):
        if name in (data.get(candidate) or {}):
            return candidate, data[candidate][name]
    raise SpecError(f"{module.name}: компонент {section}/{name} не объявлен")


def _walk(module: Path, pointer: list[str], registry: dict[Path, dict[str, Any]]) -> Any:
    """Узел модуля по указателю; имя схемы можно писать без секции (schemas/device.yaml#/Name)."""
    data = registry[module]
    if pointer[0] in data:
        path = pointer
    else:
        section, _ = _lookup(module, pointer[0], "schemas", registry)
        path = [section, *pointer]
    node: Any = data
    for index, key in enumerate(path):
        try:
            node = node[key]
        except (KeyError, TypeError) as exc:  # noqa: PERF203 - сообщение важнее цикла
            raise SpecError(
                f"{module.name}: не разрешается указатель #/{'/'.join(pointer)} "
                f"(трудность на '{key}')"
            ) from exc
    return copy.deepcopy(node)


def _inline_from_module(
    module: Path, pointer: list[str], registry: dict[Path, dict[str, Any]]
) -> Any:
    return _walk(module, pointer, registry)


def _rewrite_refs(
    node: Any,
    components: dict[str, dict[str, Any]],
    registry: dict[Path, dict[str, Any]],
    origin: str,
) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            match = REF_RE.match(ref)
            if not match:
                raise SpecError(f"{origin}: внешний $ref без файла: {ref}")
            # Все модули ссылаются на файлы от корня specifications/backend,
            # поэтому $ref не зависит от каталога, где лежит модуль.
            target = _module_path(match.group(1))
            if target not in registry:
                raise SpecError(f"{origin}: $ref на несобранный модуль {match.group(1)}")
            pointer = match.group(2).split("/")
            if len(pointer) == 1:
                section, content = _lookup(target, pointer[0], _section_for(target), registry)
                location = _register(
                    components,
                    target,
                    section,
                    pointer[0],
                    _rewrite_refs(content, components, registry, f"{target.name}#/{pointer[0]}"),
                )
                return {**node, "$ref": location}
            return _rewrite_refs(
                _inline_from_module(target, pointer, registry),
                components,
                registry,
                origin,
            )
        return {
            key: _rewrite_refs(value, components, registry, origin)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_rewrite_refs(item, components, registry, origin) for item in node]
    return node


def _rewrite_refs_from_module(
    module: Path,
    name: str,
    section: str,
    registry: dict[Path, dict[str, Any]],
    components: dict[str, dict[str, Any]],
) -> Any:
    content = _walk(module, [name], registry)
    return _rewrite_refs(content, components, registry, f"{module.name}#/{name}")


def _collect_refs(node: Any, found: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                found.append(value)
            else:
                _collect_refs(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, found)


def _resolve_pointer(document: dict[str, Any], ref: str) -> bool:
    if not ref.startswith("#/"):
        return False
    node: Any = document
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def build() -> tuple[dict[str, Any], list[str]]:
    root = _load(ROOT / "openapi.yaml")
    registry: dict[Path, dict[str, Any]] = {}
    modules = sorted((ROOT / "schemas").glob("*.yaml")) + sorted((ROOT / "common").glob("*.yaml"))
    for module in modules:
        registry[module.resolve()] = _load(module)
    path_modules: dict[Path, dict[str, Any]] = {}
    for module in sorted((ROOT / "paths").glob("*.yaml")):
        path_modules[module.resolve()] = _load(module)
        registry[module.resolve()] = path_modules[module.resolve()]

    components: dict[str, dict[str, Any]] = {}
    for module in modules:
        resolved = module.resolve()
        data = registry[resolved]
        for section in ("schemas", "parameters", "securitySchemes", "responses", "headers"):
            for name, value in (data.get(section) or {}).items():
                _register(
                    components,
                    module,
                    section,
                    name,
                    _rewrite_refs(value, components, registry, module.name),
                )
        if not any(
            key in data for key in ("schemas", "parameters", "responses", "headers", "securitySchemes")
        ):
            raise SpecError(f"{module.name}: модуль без компонентов")

    paths: dict[str, Any] = {}
    problems: list[str] = []
    operation_ids: dict[str, str] = {}
    documented_tags = {tag["name"] for tag in root.get("tags", [])}
    for module, data in path_modules.items():
        for route, item in (data.get("paths") or {}).items():
            if route in paths:
                raise SpecError(f"{module.name}: маршрут {route} объявлен дважды")
            paths[route] = _rewrite_refs(item, components, registry, module.name)
    for route, item in paths.items():
        for method, operation in item.items():
            if method.startswith("x-"):
                continue
            origin = f"{method.upper()} {route}"
            op_id = operation.get("operationId")
            if not op_id:
                problems.append(f"{origin}: отсутствует operationId")
            elif op_id in operation_ids:
                problems.append(
                    f"{origin}: operationId '{op_id}' дублирует {operation_ids[op_id]}"
                )
            else:
                operation_ids[op_id] = origin
            tags = operation.get("tags") or []
            if not tags:
                problems.append(f"{origin}: операция без tag")
            for tag in tags:
                if tag not in documented_tags:
                    problems.append(f"{origin}: тег '{tag}' не описан в openapi.yaml")
            if not operation.get("summary"):
                problems.append(f"{origin}: отсутствует summary")

    document = {
        key: value for key, value in root.items() if key not in {"paths", "components"}
    }
    document["paths"] = paths
    document["components"] = components

    unresolved: list[str] = []
    refs: list[str] = []
    _collect_refs(document, refs)
    for ref in refs:
        if not _resolve_pointer(document, ref):
            unresolved.append(ref)
    problems.extend(f"неразрешимый $ref: {ref}" for ref in sorted(set(unresolved)))
    return document, problems


class _Dumper(yaml.SafeDumper):
    pass


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _represent_str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="только проверить, не писать файл")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        document, problems = build()
    except SpecError as error:
        print(f"ОШИБКА: {error}", file=sys.stderr)
        return 1

    if problems:
        for problem in problems:
            print(f"ОШИБКА: {problem}", file=sys.stderr)
        return 1

    text = yaml.dump(
        document,
        Dumper=_Dumper,
        allow_unicode=True,
        sort_keys=False,
        width=100,
        default_flow_style=False,
    )
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print("BACKEND.yaml устарел: запустите сборщик без --check", file=sys.stderr)
            return 1
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    if not args.quiet:
        operations = sum(
            1 for item in document["paths"].values() for m in item if not m.startswith("x-")
        )
        print(
            f"BACKEND.yaml: {len(document['paths'])} путей, {operations} операций, "
            f"{len(document['components'].get('schemas', {}))} схем"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
