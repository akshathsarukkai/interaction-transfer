"""Acquire, patch, build and run the published d-chain sampler.

Provenance
----------
Code    ``https://github.com/skoplev/d-chain``, GPL-3.0, HEAD ``72b2445``
        (2022-08-21). Four files; the whole model is ``dchain.cpp`` (1,045
        lines) and the whole scoring step is ``post/interpretMCMC.R``.
Paper   Koplev S, Longden J, Ferkinghoff-Borg J, Blicher Bjerregard M, Cox TR,
        Erler JT, Pedersen JT, Voellmy F, Sommer MOA, Linding R. Cell Reports
        20(12):2784-2791, 2017. doi:10.1016/j.celrep.2017.08.095.

**The published sampler itself is what this module runs.** Not a reimplementation
of it, not a reduced equivalent: the C++ that produced the deposited screen,
compiled and executed on simulated data. That is what makes the null a test of
*this estimator* rather than of a model of it.

The source is not vendored into this repository. It is GPL-3.0 and this project
is not; more practically, the same rule already applies to the Mendeley deposit
(``data/raw/`` is gitignored and digest-verified on every ingestion), and a
fetched-and-verified copy is a stronger provenance claim than a copied one.
:func:`prepare` fetches it, checks the digest, applies :data:`PATCHES`, and
compiles.

Upstream copyright and licence
------------------------------
The d-chain source is copyright its authors and licensed **GPL-3.0**. It is not
vendored here: :func:`prepare` fetches it. Two things in *this* file are
nonetheless taken from it verbatim and are therefore covered by the upstream
licence rather than by this repository's:

* the ``old`` match strings in :data:`PATCHES` -- roughly 27 lines of the
  authors' C++, quoted exactly because a patch that does not match its target
  exactly is a patch that silently does nothing; and
* the ``new`` strings paired with them, which are edits of those same lines.

They are quoted rather than described so that every edit is checkable against
the source it edits, which is the property the whole falsification rests on.
Obtain the upstream source, and its licence, from
<https://github.com/skoplev/d-chain>. See THIRD_PARTY_DATA.md.

What the patch changes, and why none of it is the model
-------------------------------------------------------
Every edit is asserted, and the audit trail lives in **two** places, which is
worth saying plainly because the second holds the riskiest edits.
:data:`PATCHES` lists eight as exact (old, new) pairs, each asserted to match
exactly once. The remaining seven -- the sufficient-statistic call sites, and the
only edits that touch lines *inside* the Metropolis blocks -- are rewritten by
the regex :data:`_SUFSTAT_CALL` and asserted to hit exactly
:data:`_SUFSTAT_CALL_N` times. A reader auditing ``PATCHES`` alone would miss
exactly the seven that matter most.

There are three kinds of edit, and the third is checkable to the byte:

1. **The Boost dependency is removed.** ``dchain.cpp`` uses Boost only for
   ``program_options`` (command line) and ``filesystem`` (path joining). Both
   are replaced by a small shim over the standard library. No line inside the
   sampler is touched.
2. **A ``--seed`` option is added.** The published program default-constructs
   ``default_random_engine``, so every run of it produces the *same* chain --
   there is no seed and no way to run a second one. An ensemble needs an
   estimator seed, and the convergence question needs more than one chain.
   ``--seed 0`` restores the default-constructed engine exactly.
3. **Sufficient statistics are computed once instead of ~1.6e5 times per
   iteration** (at 100 drugs: 40,100 in the beta sweep, 80,900 in the per-drug
   sweep, 40,000 in the per-pair sweep). ``calcSufficientStat()`` takes ``beta_plate`` and ``beta_run``
   but every line that would use them is commented out in its body, and neither
   vector is ever updated, so its return value is constant for the whole run.
   The original recomputes it inside every Metropolis proposal.

Edit 3 is the only one that could conceivably change a number, so it is checked
rather than argued: :func:`verify_equivalence` runs the patched and unpatched
programs on the same input at ``--seed 0`` and requires **byte-identical**
output files. If the check fails, the build is refused.

What is *not* patched, and is therefore inherited exactly: the likelihood, the
Student-t marginal with its ``Gamma(0.6, 0.02)`` variance prior, the
``(K, h, alpha)`` priors, the proposal distributions, the four-case selector
scheme, the absence of any prior on the selectors, the sweep order, and the
storage rule ``iter > burn && iter % subsample == 0``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DCHAIN_REPO = "https://github.com/skoplev/d-chain"
DCHAIN_COMMIT = "72b2445786daa13c3df41aa4b2312b84a7f79266"
DCHAIN_LICENSE = "GPL-3.0"
DCHAIN_PAPER = "10.1016/j.celrep.2017.08.095"
_RAW = f"https://raw.githubusercontent.com/skoplev/d-chain/{DCHAIN_COMMIT}"

#: Digests observed on the acquisition date. All four files are recorded even
#: though only two are used, so a future reader can re-derive everything in
#: ``docs/dchain_reconstruction.md`` without guessing which files existed.
DCHAIN_FILES: dict[str, tuple[str, int]] = {
    "dchain.cpp": ("a3f1def66ff00dcb6fa5f70848b9a77652d790ebda343efff7b24f5d036ebc91", 31762),
    "README.md": ("4a50d0c57e024dc17866f49496e590d0cb931660fd65299bbceac1f2d38e283f", 315),
    "post/interpretMCMC.R": ("ab6f2cfab5de78161e1fdbf46696089b651c998bffba87bbbc6c01a1f960360d", 7108),
    "data/viability_data.csv": ("dec60c26989e788556857e514a1462590af22bad98d7a36b6b0b3a51ff5ac68c", 3989),
}

DEFAULT_DIR = Path("third_party/dchain")

#: Published MCMC defaults, from ``dchain.cpp``'s ``main()``. The deposit was
#: produced with these: ``iter > 100000 && iter % 200 == 0`` over 500,000
#: iterations retains exactly 1,999 samples, and every ``|lambda|`` in the
#: deposited tables is an exact multiple of 1/1999. (The paper says 2,000.)
PUBLISHED_MCMC = {"iterations": 500_000, "burn": 100_000, "subsample": 200,
                  "init_phase": 20_000}
PUBLISHED_N_SAMPLES = 1999

_BOOST_SHIM = r'''#include <filesystem>
// --- std-library shim replacing boost::program_options and boost::filesystem --
// The only uses of boost in this file are the command line and path joining.
// Neither touches the posterior. Everything below is scaffolding; the sampler
// itself is unmodified.
namespace po {
  struct error : std::runtime_error { error(const std::string&s):std::runtime_error(s){} };
  struct value_semantic { virtual ~value_semantic(){} virtual void set(const std::string&) const=0; bool req=false; bool has_def=false; };
  template<class T> struct typed_value : value_semantic {
    T* store; T defv;
    typed_value(T* s):store(s){}
    typed_value* required(){ req=true; return this; }
    typed_value* default_value(const T& d){ defv=d; has_def=true; if(store)*store=d; return this; }
    void set(const std::string& s) const override;
  };
  template<> inline void typed_value<std::string>::set(const std::string& s) const { if(store)*store=s; }
  template<> inline void typed_value<int>::set(const std::string& s) const { if(store)*store=std::stoi(s); }
  template<class T> typed_value<T>* value(T* s){ return new typed_value<T>(s); }
  struct opt { std::string longn, shortn, desc; value_semantic* v; };
  struct options_description {
    std::string title; std::vector<opt> opts;
    options_description(const std::string& t=""):title(t){}
    struct adder {
      options_description* d;
      adder& operator()(const char* names, value_semantic* v, const char* desc){
        std::string n(names); auto c=n.find(','); opt o;
        o.longn = (c==std::string::npos)? n : n.substr(0,c);
        o.shortn = (c==std::string::npos)? "" : n.substr(c+1);
        o.desc=desc; o.v=v; d->opts.push_back(o); return *this; }
      adder& operator()(const char* names, const char* desc){ return (*this)(names,(value_semantic*)0,desc); }
    };
    adder add_options(){ return adder{this}; }
  };
  inline std::ostream& operator<<(std::ostream& os, const options_description& d){
    for(size_t i=0;i<d.opts.size();++i) os<<"  --"<<d.opts[i].longn<<"\t"<<d.opts[i].desc<<"\n";
    return os; }
  struct positional_options_description { std::string name; int n;
    void add(const char* s, int k){ name=s; n=k; } };
  struct variables_map { std::map<std::string,int> seen;
    int count(const std::string& k) const { auto it=seen.find(k); return it==seen.end()?0:it->second; } };
  struct parsed_options { std::vector<std::pair<std::string,std::string>> kv; std::vector<std::string> pos; };
  struct command_line_parser {
    int argc; char const** argv; options_description* d=0; positional_options_description* p=0;
    command_line_parser(int c, char const* a[]):argc(c),argv(a){}
    command_line_parser& options(options_description& o){ d=&o; return *this; }
    command_line_parser& positional(positional_options_description& q){ p=&q; return *this; }
    parsed_options run(){
      parsed_options out;
      for(int i=1;i<argc;i++){
        std::string a(argv[i]);
        if(a.rfind("--",0)==0){ std::string k=a.substr(2); std::string v;
          auto e=k.find('='); if(e!=std::string::npos){ v=k.substr(e+1); k=k.substr(0,e); }
          else if(i+1<argc && std::string(argv[i+1]).rfind("-",0)!=0){ v=argv[++i]; }
          out.kv.push_back(std::make_pair(k,v)); }
        else if(a.rfind("-",0)==0 && a.size()>1){ std::string s=a.substr(1); std::string v;
          if(i+1<argc && std::string(argv[i+1]).rfind("-",0)!=0) v=argv[++i];
          std::string k=s; if(d) for(size_t j=0;j<d->opts.size();++j) if(d->opts[j].shortn==s) k=d->opts[j].longn;
          out.kv.push_back(std::make_pair(k,v)); }
        else out.pos.push_back(a);
      }
      last_desc=d; last_pos=p; return out; }
    static options_description* last_desc; static positional_options_description* last_pos;
  };
  inline options_description* command_line_parser::last_desc=0;
  inline positional_options_description* command_line_parser::last_pos=0;
  inline void store(const parsed_options& po_, variables_map& vm){
    options_description* d=command_line_parser::last_desc;
    for(size_t i=0;i<po_.kv.size();++i){ vm.seen[po_.kv[i].first]++;
      if(d) for(size_t j=0;j<d->opts.size();++j)
        if(d->opts[j].longn==po_.kv[i].first && d->opts[j].v) d->opts[j].v->set(po_.kv[i].second); }
    if(command_line_parser::last_pos && !po_.pos.empty()){
      const std::string& pn=command_line_parser::last_pos->name; vm.seen[pn]++;
      if(d) for(size_t j=0;j<d->opts.size();++j)
        if(d->opts[j].longn==pn && d->opts[j].v) d->opts[j].v->set(po_.pos[0]); } }
  inline void notify(variables_map&){}
}
// --- end shim ---------------------------------------------------------------'''

#: (label, old, new, expected occurrences). Applied in order; each must match
#: exactly the stated number of times or :func:`patch_source` raises.
PATCHES: tuple[tuple[str, str, str, int], ...] = (
    ("boost/includes",
     "#include <boost/program_options.hpp>\n#include <boost/filesystem.hpp>",
     _BOOST_SHIM, 1),
    ("boost/namespaces",
     "namespace po = boost::program_options;\nnamespace fs = boost::filesystem;",
     "namespace fs = std::filesystem;", 1),
    ("seed/field",
     "\tint init_phase;  // relies on init_lambda to be true.\n\tstring strain;",
     "\tint init_phase;  // relies on init_lambda to be true.\n"
     "\tint seed;  // 0 = the default-constructed engine, i.e. published behaviour\n"
     "\tstring strain;", 1),
    ("seed/option",
     '\t\t\t("help,h", "Help screen")',
     '\t\t\t("seed,r",\n\t\t\t\tpo::value<int>(&options.seed)->default_value(0),\n'
     '\t\t\t\t"Seed for the MCMC random engine. 0 = published default.")\n'
     '\t\t\t("help,h", "Help screen")', 1),
    ("seed/echo",
     '\tcout << "\\tinit_phase: " << options.init_phase << endl;',
     '\tcout << "\\tinit_phase: " << options.init_phase << endl;\n'
     '\tcout << "\\tseed: " << options.seed << endl;', 1),
    ("seed/use",
     "\t// Random number generator\n\tdefault_random_engine generator;",
     "\t// Random number generator. The published program default-constructs\n"
     "\t// this, so every run of it is the same chain; --seed exposes it.\n"
     "\tdefault_random_engine generator;\n"
     "\tif (options.seed != 0) {\n\t\tgenerator.seed((unsigned)options.seed);\n\t}", 1),
    ("sufstat/decl",
     "\tvector<ExperimentSet> obsA;  // obsA[a][e][r],  name,experiment,repeat\n"
     "\tobsA.resize(drugs.size());\n\n"
     "\tvector<ExperimentSet> obsA0;  // obsA0[a][e][r], although only single experiment\n"
     "\tobsA0.resize(drugs.size());\n\n"
     "\tvector<vector<ExperimentSet> > obsAB;  // obsAB[a][b][e][r]\n"
     "\tobsAB.resize(drugs.size());",
     "\t// PERFORMANCE ONLY, checked byte-for-byte by verify_equivalence().\n"
     "\t// calcSufficientStat() ignores beta_plate and beta_run -- both offsets\n"
     "\t// are commented out in its body and neither vector is ever updated --\n"
     "\t// so a sufficient statistic is constant for the whole run.\n"
     "\tvector<vector<SufficientStat> > obsA;\n"
     "\tobsA.resize(drugs.size());\n\n"
     "\tvector<vector<SufficientStat> > obsA0;\n"
     "\tobsA0.resize(drugs.size());\n\n"
     "\tvector<vector<vector<SufficientStat> > > obsAB;  // obsAB[a][b][e]\n"
     "\tobsAB.resize(drugs.size());", 1),
    ("sufstat/dummy",
     "\t// Separate data into experiment specific observations.\n"
     "\t// Initialize data structures",
     "\t// Separate data into experiment specific observations.\n"
     "\tvector<double> beta_dummy;  // the two ignored arguments\n"
     "\t// Initialize data structures", 1),
    ("sufstat/fill",
     '\t\tif (first_obs.experiment == "A") {\n'
     "\t\t\tobsA[b].push_back(replip->second);\n"
     '\t\t} else if (first_obs.experiment == "A0") {\n'
     "\t\t\tobsA0[b].push_back(replip->second);\n"
     '\t\t} else if (first_obs.experiment == "AB") {\n'
     "\t\t\tint a = drug_index[replip->second[0].pretreatment];\n"
     "\t\t\tobsAB[a][b].push_back(replip->second);",
     '\t\tif (first_obs.experiment == "A") {\n'
     "\t\t\tobsA[b].push_back(calcSufficientStat(replip->second, beta_dummy, beta_dummy));\n"
     '\t\t} else if (first_obs.experiment == "A0") {\n'
     "\t\t\tobsA0[b].push_back(calcSufficientStat(replip->second, beta_dummy, beta_dummy));\n"
     '\t\t} else if (first_obs.experiment == "AB") {\n'
     "\t\t\tint a = drug_index[replip->second[0].pretreatment];\n"
     "\t\t\tobsAB[a][b].push_back(calcSufficientStat(replip->second, beta_dummy, beta_dummy));", 1),
)

#: The seven call sites where a sufficient statistic is recomputed. Rewritten by
#: regex because they differ only in which array is indexed.
_SUFSTAT_CALL = (
    r"SufficientStat stat = calcSufficientStat\("
    r"(obsA0\[a\]|obsA\[a\]|obsAB\[a\]\[b\]|obsAB\[b\]\[a\])\[i\], beta_plate, beta_run\);")
_SUFSTAT_CALL_N = 7


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_source(dest: Path = DEFAULT_DIR, force: bool = False) -> dict[str, Path]:
    """Download the four d-chain files, verifying each digest."""
    out: dict[str, Path] = {}
    for name, (digest, size) in DCHAIN_FILES.items():
        path = dest / name
        if path.exists() and not force and sha256_of(path.read_bytes()) == digest:
            out[name] = path
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{_RAW}/{name}", timeout=120) as resp:
            payload = resp.read()
        got = sha256_of(payload)
        if got != digest:
            raise RuntimeError(
                f"{name}: sha256 mismatch\n  expected {digest}\n  got      {got}\n"
                f"The pinned commit {DCHAIN_COMMIT} should be immutable. Do not "
                f"proceed with a sampler whose provenance no longer matches.")
        if len(payload) != size:
            raise RuntimeError(f"{name}: {len(payload)} bytes, expected {size}")
        path.write_bytes(payload)
        out[name] = path
    (dest / "PROVENANCE.json").write_text(json.dumps({
        "repo": DCHAIN_REPO, "commit": DCHAIN_COMMIT, "license": DCHAIN_LICENSE,
        "paper_doi": DCHAIN_PAPER,
        "files": {k: {"sha256": v[0], "size": v[1]} for k, v in DCHAIN_FILES.items()},
        "note": ("Fetched, not vendored. The patch applied before compilation is "
                 "dchain_null.dchain.PATCHES; see that module's docstring."),
    }, indent=2) + "\n")
    return out


def patch_source(src: str) -> str:
    """Apply :data:`PATCHES` and the sufficient-statistic rewrite, checking counts."""
    import re
    out = src
    for label, old, new, n in PATCHES:
        found = out.count(old)
        if found != n:
            raise RuntimeError(
                f"patch {label!r} matched {found} times, expected {n}. The "
                f"upstream file is not what this patch was written against; "
                f"re-derive it rather than loosening the match.")
        out = out.replace(old, new)
    out, k = re.subn(_SUFSTAT_CALL,
                     lambda m: f"const SufficientStat &stat = {m.group(1)}[i];", out)
    if k != _SUFSTAT_CALL_N:
        raise RuntimeError(f"sufficient-statistic rewrite hit {k} call sites, "
                           f"expected {_SUFSTAT_CALL_N}")
    return out


def _compiler() -> str:
    for cc in ("clang++", "g++", "c++"):
        if shutil.which(cc):
            return cc
    raise RuntimeError("no C++ compiler found; need clang++, g++ or c++")


def build(dest: Path = DEFAULT_DIR, verify: bool = True) -> Path:
    """Compile the patched sampler, and the unpatched one if verifying."""
    files = fetch_source(dest)
    src = files["dchain.cpp"].read_text()
    build_dir = dest / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    patched = build_dir / "dchain_patched.cpp"
    patched.write_text(patch_source(src))
    binary = build_dir / "dchain"
    cc = _compiler()
    subprocess.run([cc, "-std=c++17", "-O3", "-DNDEBUG", str(patched), "-o", str(binary)],
                   check=True, capture_output=True)

    if verify:
        # The reference build keeps the sufficient-statistic recomputation and
        # the unseeded engine; only the boost shim is applied, because without
        # it nothing compiles here at all.
        import re
        ref_src = src
        for label, old, new, n in PATCHES:
            if label.startswith("boost/"):
                ref_src = ref_src.replace(old, new)
        ref = build_dir / "dchain_reference.cpp"
        ref.write_text(ref_src)
        ref_bin = build_dir / "dchain_reference"
        subprocess.run([cc, "-std=c++17", "-O2", "-DNDEBUG", str(ref), "-o", str(ref_bin)],
                       check=True, capture_output=True)
        verify_equivalence(binary, ref_bin, files["data/viability_data.csv"],
                           build_dir / "_equivalence")
    return binary


OUTPUT_FILES = ("drugs.csv", "theta.csv", "theta_AB.csv", "lambda.csv",
                "lambda_AB.csv", "beta_residual.csv")


def run(binary: Path, data_csv: Path, out_dir: Path, cell: str = "SIM",
        iterations: int = PUBLISHED_MCMC["iterations"],
        burn: int = PUBLISHED_MCMC["burn"],
        subsample: int = PUBLISHED_MCMC["subsample"],
        init_phase: int = PUBLISHED_MCMC["init_phase"],
        seed: int = 0) -> dict:
    """Run the sampler. Returns a diagnostics dict; raises on a nonzero exit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), str(data_csv), "-o", str(out_dir), "-c", cell,
           "-i", str(iterations), "-b", str(burn), "-s", str(subsample),
           "-n", str(init_phase), "-r", str(seed)]
    import time
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"dchain exited {proc.returncode}\n{proc.stderr[-2000:]}")
    missing = [f for f in OUTPUT_FILES if not (out_dir / f).exists()]
    if missing:
        raise RuntimeError(f"dchain produced no {missing}; stdout tail:\n"
                           f"{proc.stdout[-1000:]}")
    n_stored = sum(1 for _ in open(out_dir / "lambda.csv"))
    expected = len([k for k in range(iterations) if k > burn and k % subsample == 0])
    return {
        "command": cmd, "seconds": elapsed, "n_samples": n_stored,
        "n_samples_expected": expected,
        "iterations": iterations, "burn": burn, "subsample": subsample,
        "init_phase": init_phase, "seed": seed,
        "stdout_tail": proc.stdout[-400:],
    }


def verify_equivalence(patched: Path, reference: Path, data_csv: Path,
                       work: Path, iterations: int = 60_000) -> dict:
    """Refuse the optimised build unless it is byte-identical to the original.

    The optimisation is an argument ("that value is constant"); this is the
    check. Run at ``--seed 0``, which is the default-constructed engine the
    published program uses, so the two chains are the same chain.
    """
    work.mkdir(parents=True, exist_ok=True)
    a, b = work / "patched", work / "reference"
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    run(patched, data_csv, a, cell="PANC1", iterations=iterations,
        burn=iterations // 3, subsample=100, init_phase=iterations // 3, seed=0)
    run(reference, data_csv, b, cell="PANC1", iterations=iterations,
        burn=iterations // 3, subsample=100, init_phase=iterations // 3, seed=0)
    diffs = [f for f in OUTPUT_FILES
             if (a / f).read_bytes() != (b / f).read_bytes()]
    if diffs:
        raise RuntimeError(
            f"the patched sampler does not reproduce the original: {diffs} "
            f"differ. The performance patch changed the posterior; do not use it.")
    return {"identical": True, "files": list(OUTPUT_FILES),
            "iterations": iterations}


@dataclass(frozen=True)
class Posterior:
    """Parsed MCMC output. Sample-major, matching the R parser's arrays."""

    drugs: tuple[str, ...]
    theta: np.ndarray        # (S, n, 3)
    lam: np.ndarray          # (S, n)
    theta_AB: np.ndarray     # (S, n, n, 3)  [first, second]
    lam_AB: np.ndarray       # (S, n, n)     [first, second]
    beta_residual: np.ndarray  # (S, n)

    @property
    def n_samples(self) -> int:
        return self.theta.shape[0]

    @property
    def n_drugs(self) -> int:
        return self.theta.shape[1]


def _parse_theta(path: Path, n: int) -> np.ndarray:
    """Rows of ``K;h;alpha`` triples, comma separated. Matches ``decodeParamVector``."""
    txt = path.read_text().strip("\n").split("\n")
    rows = [line.replace(";", ",") for line in txt if line]
    arr = np.array([np.fromstring(r, sep=",") for r in rows])
    if arr.shape[1] != 3 * n:
        raise ValueError(f"{path.name}: {arr.shape[1]} values per row, expected {3*n}")
    return arr.reshape(len(rows), n, 3)


def load_posterior(out_dir: Path) -> Posterior:
    """Parse the sampler's five CSVs.

    Index order is fixed by ``dchain.cpp``: the flat combination arrays are
    written with ``a * drugs.size() + b``, i.e. **row-major with the first drug
    as the row**, which is what ``interpretMCMC.R`` reconstructs with
    ``byrow=TRUE`` and dimension names ``(Sample, First, Second)``.
    """
    drugs = tuple(out_dir.joinpath("drugs.csv").read_text().strip().split(","))
    n = len(drugs)
    theta = _parse_theta(out_dir / "theta.csv", n)
    theta_AB = _parse_theta(out_dir / "theta_AB.csv", n * n)
    theta_AB = theta_AB.reshape(theta_AB.shape[0], n, n, 3)
    lam = np.loadtxt(out_dir / "lambda.csv", delimiter=",", ndmin=2)
    lam_AB = np.loadtxt(out_dir / "lambda_AB.csv", delimiter=",", ndmin=2)
    lam_AB = lam_AB.reshape(lam_AB.shape[0], n, n)
    beta = np.loadtxt(out_dir / "beta_residual.csv", delimiter=",", ndmin=2)
    sizes = {theta.shape[0], theta_AB.shape[0], lam.shape[0], lam_AB.shape[0],
             beta.shape[0]}
    if len(sizes) != 1:
        raise ValueError(f"output files disagree on sample count: {sizes}")
    return Posterior(drugs=drugs, theta=theta, lam=lam, theta_AB=theta_AB,
                     lam_AB=lam_AB, beta_residual=beta)


def deposit_identities(synergy: np.ndarray, lam_ab_mean: np.ndarray,
                       n_samples: int) -> dict:
    """The four identities the deposited tables satisfy, checked on any output.

    These are the strongest available fidelity checks on the measurement layer,
    because the *inputs* to the synergy formula -- posterior samples of theta and
    lambda -- were never deposited, so a direct numerical comparison against the
    published scores is impossible for anyone. What can be checked is that a
    reconstruction reproduces the structural signature the real deposit carries:

    1. every ``|lambda|`` is an exact multiple of ``1 / n_samples``;
    2. ``lambda == 0`` implies ``synergy_measure == 0`` exactly;
    3. ``|synergy_measure| <= |lambda|`` everywhere, which follows from the
       per-sample difference lying in ``[-1, 1]``;
    4. the retained-sample count implied by (1) matches the MCMC settings.

    All four hold on the deposited A375 and PANC1 tables at ``n_samples=1999``.
    """
    q = np.abs(lam_ab_mean) * n_samples
    zero = lam_ab_mean == 0
    return {
        "n_samples": int(n_samples),
        "lambda_is_multiple_of_1_over_n": bool(
            np.abs(q - np.round(q)).max() < 1e-6),
        "n_lambda_zero": int(zero.sum()),
        "zero_lambda_implies_zero_synergy": bool(
            np.all(synergy[zero] == 0.0)) if zero.any() else True,
        "abs_synergy_le_abs_lambda": bool(
            np.all(np.abs(synergy) <= np.abs(lam_ab_mean) + 1e-12)),
    }


def deposited_reference(raw_dir: Path) -> dict:
    """The same four identities, measured on the real deposit. The benchmark."""
    out = {}
    for table, label in (("Data Table 1.csv", "A375"), ("Data Table 2.csv", "PANC1")):
        d = pd.read_csv(raw_dir / table)
        out[label] = deposit_identities(
            d["synergy_measure"].to_numpy(), d["lambda"].to_numpy(),
            PUBLISHED_N_SAMPLES)
    return out


def validate_on_deposited_example(binary: Path, dest: Path = DEFAULT_DIR,
                                  work: Path | None = None,
                                  iterations: int = 200_000) -> dict:
    """End-to-end check of sampler + measurement layer on the authors' own data.

    ``d-chain/data/viability_data.csv`` is a 66-row two-drug example shipped with
    the source. It is not the screen -- the screen's raw data was never deposited
    -- but it is *real upstream input in the real schema*, and running it through
    the compiled sampler and then through the Python port of
    ``interpretMCMC.R`` exercises every layer between a CSV and a
    ``synergy_measure`` on something nobody in this project constructed.

    What it can establish: that the pipeline runs on the authors' input, and that
    the output satisfies the four structural identities the deposited screen
    satisfies (:func:`deposit_identities`). What it cannot: agreement with a
    published number, because no posterior samples were deposited for any
    published value. That limit is a property of the deposit, not of this check.
    """
    from .synergy import synergy_posterior
    files = fetch_source(dest)
    work = work or (dest / "build" / "_validation")
    out = work / "mcmc"
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    diag = run(binary, files["data/viability_data.csv"], out, cell="PANC1",
               iterations=iterations, burn=iterations // 5,
               subsample=max(1, (iterations - iterations // 5) // 1999),
               init_phase=iterations // 10, seed=1)
    post = load_posterior(out)
    sp = synergy_posterior(post.theta, post.lam, post.theta_AB, post.lam_AB)
    lam_ab_mean = post.lam_AB.mean(axis=0)
    ident = deposit_identities(sp["mean"], lam_ab_mean, post.n_samples)
    n = post.n_drugs
    off = ~np.eye(n, dtype=bool)
    return {
        "source": "d-chain/data/viability_data.csv (66 rows, 2 drugs)",
        "drugs": list(post.drugs),
        "mcmc": {k: diag[k] for k in ("n_samples", "n_samples_expected",
                                      "iterations", "burn", "subsample",
                                      "seed", "seconds")},
        "identities": ident,
        "identities_hold": bool(ident["lambda_is_multiple_of_1_over_n"]
                                and ident["zero_lambda_implies_zero_synergy"]
                                and ident["abs_synergy_le_abs_lambda"]),
        "synergy_mean": [[float(v) for v in row] for row in sp["mean"]],
        "synergy_sd": [[float(v) for v in row] for row in sp["sd"]],
        "lambda_ab_mean": [[float(v) for v in row] for row in lam_ab_mean],
        "single_selector_mean": [float(v) for v in post.lam.mean(axis=0)],
        "beta_residual_mean": [float(v) for v in post.beta_residual.mean(axis=0)],
        "theta_mean": [[float(v) for v in row] for row in post.theta.mean(axis=0)],
        "directional_offdiag": float((sp["mean"] - sp["mean"].T)[off].std()),
    }
