from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_qa import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DB_DIR,
    DEFAULT_DOC_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    answer_question,
    build_index,
    format_sources,
)


def path_arg(value: str) -> Path:
    return Path(value).expanduser()


def add_index_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db-dir",
        type=path_arg,
        default=DEFAULT_DB_DIR,
        help=f"本地向量索引目录，默认: {DEFAULT_DB_DIR}",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"索引名称，默认: {DEFAULT_COLLECTION_NAME}",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"向量模型，默认: {DEFAULT_EMBEDDING_MODEL}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把文本语料做成向量索引，并基于检索结果进行问答。"
    )
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build", help="读取语料并建立/重建向量索引")
    build.add_argument(
        "--source",
        type=path_arg,
        default=DEFAULT_DOC_PATH,
        help=f"语料文件路径，默认: {DEFAULT_DOC_PATH}",
    )
    build.add_argument("--chunk-size", type=int, default=900, help="每个文本块的最大字符数")
    build.add_argument("--overlap", type=int, default=120, help="相邻文本块的重叠字符数")
    build.add_argument("--batch-size", type=int, default=64, help="写入索引的批量大小")
    build.add_argument(
        "--no-rebuild",
        action="store_true",
        help="如果索引已存在，则复用现有索引",
    )
    add_index_options(build)

    ask = subparsers.add_parser("ask", help="提出一个问题并返回答案")
    ask.add_argument("question", nargs="+", help="问题文本")
    ask.add_argument("--top-k", type=int, default=5, help="检索片段数量")
    ask.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help=f"生成模型，默认: {DEFAULT_LLM_MODEL}")
    ask.add_argument("--no-llm", action="store_true", help="只检索片段，不调用 Gemini 生成答案")
    ask.add_argument("--hide-sources", action="store_true", help="不打印来源片段")
    add_index_options(ask)

    chat = subparsers.add_parser("chat", help="进入连续问答模式")
    chat.add_argument("--top-k", type=int, default=5, help="每轮检索片段数量")
    chat.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help=f"生成模型，默认: {DEFAULT_LLM_MODEL}")
    chat.add_argument("--no-llm", action="store_true", help="只检索片段，不调用 Gemini 生成答案")
    chat.add_argument("--hide-sources", action="store_true", help="不打印来源片段")
    add_index_options(chat)

    return parser


def print_answer(result, show_sources: bool) -> None:
    mode = "Gemini 生成" if result.used_llm else "仅向量检索"
    print(f"\n答案 ({mode}):\n{result.answer}")
    if show_sources:
        print("\n来源片段:")
        print(format_sources(result.hits))


def handle_build(args: argparse.Namespace) -> int:
    stats = build_index(
        doc_path=args.source,
        db_dir=args.db_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        rebuild=not args.no_rebuild,
    )
    action = "已重建" if stats.rebuilt else "已复用"
    print(
        f"\n{action}索引: {stats.chunk_count} 个文本块\n"
        f"语料: {stats.source_path}\n"
        f"索引: {stats.db_dir}\n"
        f"索引名称: {stats.collection_name}"
    )
    return 0


def handle_ask(args: argparse.Namespace) -> int:
    question = " ".join(args.question)
    result = answer_question(
        question=question,
        db_dir=args.db_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
        use_llm=not args.no_llm,
    )
    print_answer(result, show_sources=not args.hide_sources)
    return 0


def handle_chat(args: argparse.Namespace) -> int:
    print("进入问答模式。输入 exit、quit 或 q 退出。")
    while True:
        try:
            question = input("\n问题> ").strip()
        except EOFError:
            print()
            return 0

        if question.lower() in {"exit", "quit", "q"}:
            return 0
        if not question:
            continue

        try:
            result = answer_question(
                question=question,
                db_dir=args.db_dir,
                collection_name=args.collection,
                embedding_model=args.embedding_model,
                llm_model=args.llm_model,
                top_k=args.top_k,
                use_llm=not args.no_llm,
            )
            print_answer(result, show_sources=not args.hide_sources)
        except RuntimeError as exc:
            print(f"错误: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "build":
            return handle_build(args)
        if args.command == "ask":
            return handle_ask(args)
        if args.command == "chat":
            return handle_chat(args)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
