"""Integration test for the fine-tuning window extraction pipeline.

Exercises ``extract_fasta_windows_for_sample`` against the **real** jaguar
reference (DNA Zoo ``DNAZOO_Panthera_onca_HiC`` / ``Panthera_onca_HiC``, ~2.5 GB
compressed) and the real per-sample jaguar VCF that ships under
``data/raw/``. The reference download is cached in
``data/raw/reference/`` so re-runs do not re-pull a multi-gigabyte file.

Speed/scope trade-off:
    The full jaguar reference + VCF would take minutes to load even on a
    fast machine because ``finetune_windows`` reads the entire FASTA into
    memory. To keep the integration loop tight while still exercising real
    data, this test (1) extracts a single scaffold (``HiC_scaffold_1``)
    from the cached reference into a small per-test FASTA and (2) subsets
    the source VCF to the first ``MAX_VCF_RECORDS`` PASS records on that
    scaffold. Together these reduce the working set from ~2.5 GB to a few
    MB without changing the public API path under test.

Contig-naming caveat:
    The VCF uses the original HiC assembly contig names (``HiC_scaffold_1``
    etc.) while RefSeq may relabel headers (e.g. ``NC_077XXX.1 ...
    HiC_scaffold_1 ...``). The subset FASTA writer therefore matches a
    requested scaffold against either the header ID or any whitespace-
    separated description token, and re-emits the contig under the
    VCF-facing name so downstream string-equality matching in
    ``extract_locus_windows_from_vcf`` succeeds.
"""

from __future__ import annotations

import gzip
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from urllib.request import Request, build_opener

import pytest

from jaguar_geo_assign.data.felid_assemblies import (
    APPROVED_FELID_ASSEMBLIES,
    build_refseq_fasta_url,
)
from jaguar_geo_assign.data.finetune_windows import (
    DOWNSTREAM_BASES,
    UPSTREAM_BASES,
    WINDOW_SIZE,
    iter_locus_windows_from_vcf,
    load_reference_index,
    write_locus_windows_jsonl,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VCF_PATH = (
    _REPO_ROOT
    / "data"
    / "raw"
    / "jaguar.57samples.allChr.snps.hardFilter.bi.maf.ld.masked.hwe.recode.vcf"
)
_REFERENCE_CACHE_DIR = _REPO_ROOT / "data" / "raw" / "reference"

_JAGUAR_IDENTIFIER = "DNAZOO_Panthera_onca_HiC"
_JAGUAR_ASSEMBLY = "Panthera_onca_HiC"
_TARGET_CONTIG = "HiC_scaffold_1"
_SAMPLES_UNDER_TEST = ("LegadoSP", "bPon001")
MAX_VCF_RECORDS = 50


def _jaguar_assembly_name() -> str:
    """Cross-check the hard-coded assembly name against the registry source of truth.

    Drift between this test's pinned ``_JAGUAR_ASSEMBLY`` and
    ``APPROVED_FELID_ASSEMBLIES`` would silently produce a 404 at download
    time; this lookup surfaces that drift up front via ``pytest.fail`` with
    the missing identifier instead of letting an opaque ``StopIteration``
    escape the helper.
    """
    try:
        assembly = next(a for a in APPROVED_FELID_ASSEMBLIES if a.identifier == _JAGUAR_IDENTIFIER)
    except StopIteration:
        pytest.fail(
            f"Identifier {_JAGUAR_IDENTIFIER!r} not found in APPROVED_FELID_ASSEMBLIES; "
            "the registry has drifted from this test's pinned jaguar build."
        )
    return assembly.assembly_name


def _download_jaguar_reference_if_needed() -> Path:
    """Return the path to the cached gzipped jaguar reference, downloading on cache miss.

    Cached under ``data/raw/reference/`` so the ~2.5 GB transfer happens at
    most once per checkout. Atomic rename via a ``.partial`` suffix prevents
    a half-written file from being treated as a valid cache hit on retry.
    """
    assembly_name = _jaguar_assembly_name()
    assert assembly_name == _JAGUAR_ASSEMBLY, (
        f"registry assembly name {assembly_name!r} drifted from test pin {_JAGUAR_ASSEMBLY!r}"
    )
    assembly = next(a for a in APPROVED_FELID_ASSEMBLIES if a.identifier == _JAGUAR_IDENTIFIER)
    _REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = _REFERENCE_CACHE_DIR / f"{_JAGUAR_IDENTIFIER}.fna.gz"
    if target.exists() and target.stat().st_size > 0:
        return target
    url = build_refseq_fasta_url(_JAGUAR_IDENTIFIER, _JAGUAR_ASSEMBLY, assembly.url_override)
    partial = target.with_suffix(target.suffix + ".partial")
    opener = build_opener()
    with opener.open(Request(url), timeout=600) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.replace(target)
    return target


def _iter_fasta_records(handle: Iterator[str]) -> Iterator[tuple[str, str, list[str]]]:
    """Yield ``(header_id, header_line, sequence_lines)`` for every FASTA record.

    Implemented as a streaming generator because the jaguar reference does
    not fit comfortably alongside the test process if accumulated as a dict;
    callers consume only the contig they care about and discard the rest.
    """
    header_id: str | None = None
    header_line: str | None = None
    seq_lines: list[str] = []
    for raw in handle:
        line = raw.rstrip("\n")
        if line.startswith(">"):
            if header_id is not None:
                yield header_id, header_line or "", seq_lines
            header_line = line[1:]
            header_id = header_line.split()[0]
            seq_lines = []
        elif header_id is not None:
            seq_lines.append(line)
    if header_id is not None:
        yield header_id, header_line or "", seq_lines


def _write_subset_reference_for_contig(
    cached_fasta: Path, target_contig: str, destination: Path
) -> None:
    """Copy a single contig out of ``cached_fasta`` under the VCF-facing name.

    Header matching tolerates both raw (``>HiC_scaffold_1``) and
    RefSeq-relabelled (``>NC_077XXX.1 ... HiC_scaffold_1 ...``) layouts so
    the test does not depend on which form NCBI happens to ship today.
    """
    with gzip.open(cached_fasta, "rt", encoding="ascii") as handle:
        for header_id, header_line, seq_lines in _iter_fasta_records(handle):
            tokens = header_line.split()
            if header_id == target_contig or target_contig in tokens:
                destination.write_text(
                    f">{target_contig}\n" + "\n".join(seq_lines) + "\n", encoding="ascii"
                )
                return
    raise AssertionError(
        f"Contig {target_contig!r} not found in cached reference {cached_fasta} "
        "(neither as header ID nor as a whitespace-separated description token)"
    )


def _write_subset_vcf(
    source_vcf: Path,
    destination: Path,
    *,
    target_contig: str,
    samples: tuple[str, ...],
    max_records: int,
    reference_token: str,
) -> int:
    """Project ``source_vcf`` to ``samples`` x ``target_contig`` x first ``max_records`` PASS rows.

    Header rewriting contract (required by the strict data-contract guards
    in :mod:`jaguar_geo_assign.data.finetune_windows`):
        * ``##reference=`` lines are rewritten to ``reference_token`` so
          the build-token check sees a deterministic value matching the
          subset FASTA filename.
        * ``##contig=<ID=...>`` lines are filtered to retain only the
          target contig, since the subset FASTA only contains that contig.
        * Non-requested sample columns are dropped from ``#CHROM``.

    Returns the number of data records written.
    """
    sample_indices: list[int] = []
    written = 0
    saw_reference_header = False
    with (
        source_vcf.open("rt", encoding="utf-8") as src,
        destination.open("wt", encoding="utf-8") as dst,
    ):
        for line in src:
            if line.startswith("##reference"):
                dst.write(f"##reference={reference_token}\n")
                saw_reference_header = True
                continue
            if line.startswith("##contig=<ID="):
                contig_id = line.split("ID=", 1)[1].split(",", 1)[0].rstrip(">\n")
                if contig_id != target_contig:
                    continue
                dst.write(line)
                continue
            if line.startswith("##"):
                dst.write(line)
                continue
            if line.startswith("#CHROM"):
                if not saw_reference_header:
                    dst.write(f"##reference={reference_token}\n")
                columns = line.rstrip("\n").split("\t")
                missing = [s for s in samples if s not in columns[9:]]
                if missing:
                    raise AssertionError(f"Samples missing from source VCF: {missing}")
                sample_indices = [columns.index(s) for s in samples]
                kept = columns[:9] + list(samples)
                dst.write("\t".join(kept) + "\n")
                continue
            if not sample_indices:
                raise AssertionError("Source VCF lacks #CHROM header before first data record")
            fields = line.rstrip("\n").split("\t")
            if fields[0] != target_contig or fields[6] not in {"PASS", "."}:
                continue
            kept_fields = fields[:9] + [fields[i] for i in sample_indices]
            dst.write("\t".join(kept_fields) + "\n")
            written += 1
            if written >= max_records:
                break
    return written


@pytest.fixture(scope="module")
def jaguar_reference_fasta() -> Path:
    """Module-scoped cache hook: download (or reuse) the jaguar reference once per session."""
    if not _VCF_PATH.exists():
        pytest.skip(f"Real jaguar VCF not found at {_VCF_PATH}; cannot run integration test")
    return _download_jaguar_reference_if_needed()


@pytest.mark.integration
def test_finetune_windows_pipeline_on_real_jaguar_data(
    jaguar_reference_fasta: Path, tmp_path: Path
) -> None:
    """End-to-end: real FASTA + real VCF subset → 512bp windows + JSONL on disk.

    Validation goals (each assertion guards a distinct failure mode):
        * **Window size**: any drift from 512 bp would silently break
          DNABERT-2's positional embedding contract.
        * **Coordinate math**: window_start/window_end must round-trip to
          the VCF locus position via ``UPSTREAM_BASES`` / ``DOWNSTREAM_BASES``.
        * **Allele substitution**: the center base must equal the emitted
          allele on real heterozygotes/homozygotes (not just synthetic ones).
        * **Heterozygote doubling**: every het locus that passes the filter
          gauntlet must produce exactly two windows (ref then alt) sharing
          identical flanks - the headline behavioural difference vs.
          consensus.py.
        * **JSONL fidelity**: line count and field set must match the
          in-memory dataclass, since the on-disk format is what the
          training loader will consume.
    """
    # Subset FASTA filename embeds the target contig token so it doubles
    # as the build-token evidence consumed by ``load_reference_index``.
    expected_tokens = (_JAGUAR_ASSEMBLY, _TARGET_CONTIG)
    subset_fasta = tmp_path / f"{_JAGUAR_ASSEMBLY}_{_TARGET_CONTIG}.fa"
    _write_subset_reference_for_contig(jaguar_reference_fasta, _TARGET_CONTIG, subset_fasta)
    contig_seq_length = sum(
        len(line) for line in subset_fasta.read_text(encoding="ascii").splitlines()[1:]
    )
    assert contig_seq_length > WINDOW_SIZE, (
        f"Subset contig {_TARGET_CONTIG} is too short ({contig_seq_length} bp) "
        f"to host any {WINDOW_SIZE}bp window"
    )

    subset_vcf = tmp_path / "subset.vcf"
    written = _write_subset_vcf(
        _VCF_PATH,
        subset_vcf,
        target_contig=_TARGET_CONTIG,
        samples=_SAMPLES_UNDER_TEST,
        max_records=MAX_VCF_RECORDS,
        reference_token=f"{_JAGUAR_ASSEMBLY}_{_TARGET_CONTIG}",
    )
    assert written > 0, (
        f"No PASS records on {_TARGET_CONTIG} found in {_VCF_PATH}; "
        "the source VCF or contig naming may have changed"
    )

    # Production-style usage: load the reference index *once*, then thread
    # it through one streaming-iterator call per sample. Validates the
    # scaling refactor (no per-sample re-read of the FASTA).
    reference = load_reference_index(subset_fasta, positive_reference_tokens=expected_tokens)
    output_jsonl = tmp_path / "windows.jsonl"
    windows = []
    for sample_id in _SAMPLES_UNDER_TEST:
        per_sample_jsonl = tmp_path / f"{sample_id}.jsonl"
        per_sample_windows = list(
            iter_locus_windows_from_vcf(
                sample_id=sample_id,
                sample_vcf=subset_vcf,
                reference=reference,
                expected_reference_tokens=expected_tokens,
            )
        )
        write_locus_windows_jsonl(per_sample_windows, per_sample_jsonl)
        windows.extend(per_sample_windows)
        assert per_sample_jsonl.exists(), f"JSONL not written for {sample_id}"

    assert windows, (
        "Pipeline produced zero windows; check that the VCF subset is non-empty "
        "and that the subset FASTA's contig name matches the VCF CHROM column"
    )

    contig_seq = subset_fasta.read_text(encoding="ascii").split("\n", 1)[1].replace("\n", "")
    het_locus_to_alleles: dict[tuple[str, str, int], list[str]] = {}
    for window in windows:
        assert len(window.sequence) == WINDOW_SIZE, (
            f"Window {window.contig}:{window.locus_pos} for {window.sample_id} "
            f"is {len(window.sequence)} bp, expected {WINDOW_SIZE}"
        )
        assert window.window_end - window.window_start == WINDOW_SIZE
        assert window.window_start == (window.locus_pos - 1) - UPSTREAM_BASES
        assert window.window_end == window.locus_pos + DOWNSTREAM_BASES
        assert window.contig == _TARGET_CONTIG
        assert window.sequence.isupper()
        assert set(window.sequence) <= set("ACGTN"), (
            f"Window sequence contains unexpected character at "
            f"{window.contig}:{window.locus_pos}: {set(window.sequence) - set('ACGTN')}"
        )
        assert window.sequence[UPSTREAM_BASES] == window.alt_allele.upper(), (
            f"Allele substitution drift at {window.contig}:{window.locus_pos} "
            f"({window.sample_id}): center={window.sequence[UPSTREAM_BASES]!r} "
            f"alt={window.alt_allele!r}"
        )
        expected_upstream = contig_seq[window.window_start : window.locus_pos - 1].upper()
        expected_downstream = contig_seq[window.locus_pos : window.window_end].upper()
        assert window.sequence[:UPSTREAM_BASES] == expected_upstream
        assert window.sequence[UPSTREAM_BASES + 1 :] == expected_downstream
        if window.is_heterozygous:
            key = (window.sample_id, window.contig, window.locus_pos)
            het_locus_to_alleles.setdefault(key, []).append(window.alt_allele)

    assert het_locus_to_alleles, (
        "Real jaguar VCF subset contained no biallelic heterozygotes; "
        "increase MAX_VCF_RECORDS or pick a different scaffold"
    )
    for key, alleles in het_locus_to_alleles.items():
        assert len(alleles) == 2, (
            f"Heterozygote at {key} produced {len(alleles)} windows, expected exactly 2 "
            "(one per allele)"
        )
        assert len(set(alleles)) == 2, (
            f"Heterozygote at {key} emitted duplicate alleles {alleles}; ref/alt copy "
            "logic must produce two distinct bases"
        )

    write_locus_windows_jsonl(
        iter_locus_windows_from_vcf(
            sample_id=_SAMPLES_UNDER_TEST[0],
            sample_vcf=subset_vcf,
            reference=reference,
            expected_reference_tokens=expected_tokens,
        ),
        output_jsonl,
    )
    jsonl_lines = output_jsonl.read_text(encoding="utf-8").splitlines()
    sample0_window_count = sum(1 for w in windows if w.sample_id == _SAMPLES_UNDER_TEST[0])
    assert len(jsonl_lines) == sample0_window_count, (
        f"JSONL line count {len(jsonl_lines)} != in-memory window count "
        f"{sample0_window_count} for {_SAMPLES_UNDER_TEST[0]}"
    )
    expected_fields = {
        "sample_id",
        "contig",
        "locus_pos",
        "window_start",
        "window_end",
        "sequence",
        "ref_allele",
        "alt_allele",
        "is_heterozygous",
        "genotype",
        "filter_status",
    }
    first_record = json.loads(jsonl_lines[0])
    assert set(first_record.keys()) == expected_fields, (
        f"JSONL schema drift: missing={expected_fields - set(first_record.keys())} "
        f"extra={set(first_record.keys()) - expected_fields}"
    )
    assert len(first_record["sequence"]) == WINDOW_SIZE
