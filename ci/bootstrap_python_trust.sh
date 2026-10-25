#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
refresh_system=0
if [[ "${1:-}" == "--refresh-system" ]]; then
  shift
  refresh_system=1
  elevate=()
  if [[ "$(id -u)" != 0 ]]; then
    elevate=(sudo)
  fi
  "${elevate[@]}" apt-get update
  "${elevate[@]}" apt-get install -y --no-install-recommends ca-certificates
  # TODO(AmbiqAI/helia-rt#509): distrust G3 until distribution bundles remove it.
  "${elevate[@]}" sed -i     '\|^mozilla/emSign_ECC_Root_CA_-_G3\.crt$|s|^|!|'     /etc/ca-certificates.conf
  "${elevate[@]}" update-ca-certificates --fresh
fi

python_bin="${1:-python3}"
export PATH="$(dirname "$("${python_bin}" -I -c 'import sys; print(sys.executable)')"):${PATH}"
"${python_bin}" -I -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"'
pip_flags=()
if [[ "${refresh_system}" == 1 ]] && "${python_bin}" -I -c '
import pathlib, sys, sysconfig
raise SystemExit(not (sys.prefix == sys.base_prefix and
    (pathlib.Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").exists()))
'; then
  pip_flags+=(--break-system-packages)
fi
# see AmbiqAI/helia-rt#509
pip_wheel='https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70'
if ! "${python_bin}" -I -c '
import pip
from pip._vendor.packaging.version import Version
raise SystemExit(Version(pip.__version__) < Version("26.2.2"))
'; then
  "${python_bin}" -I -m pip --isolated --disable-pip-version-check install     "${pip_flags[@]}" --no-index --no-deps --no-cache-dir --upgrade "pip @ ${pip_wheel}"
fi
"${python_bin}" -I -m pip --isolated --disable-pip-version-check install   "${pip_flags[@]}" --no-cache-dir --upgrade -c "${script_dir}/python_trust_constraints.txt" certifi
if command -v uv >/dev/null 2>&1; then
  "${python_bin}" -I -m pip --isolated --disable-pip-version-check install     "${pip_flags[@]}" --no-cache-dir --upgrade -c "${script_dir}/python_trust_constraints.txt" uv
fi
"${python_bin}" -I "${script_dir}/check_python_trust.py"
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "PIP_CONSTRAINT=${script_dir}/python_trust_constraints.txt" >> "${GITHUB_ENV}"
fi
