#!/usr/bin/env bash

ogds_normalize_data_mode() {
  local raw_mode="${1:-pure-sd}"
  local mode
  mode="$(printf '%s' "${raw_mode}" | tr '[:upper:]' '[:lower:]')"
  case "${mode}" in
    pure-sd|pure-ds)
      printf '%s\n' "pure-sd"
      ;;
    pure-real|pure-omomo)
      printf '%s\n' "pure-real"
      ;;
    mix-curriculum|mix-curriculum-omomo)
      printf '%s\n' "mix-curriculum"
      ;;
    mix-naive)
      printf '%s\n' "mix-naive"
      ;;
    fix-omomo-quater|fix_omomo_quater|fix-real|fixed-real|fix_real|fixed_real)
      printf '%s\n' "fix-omomo-quater"
      ;;
    *)
      printf '%s\n' "${mode}"
      ;;
  esac
}

ogds_resolve_data_root() {
  local ds_data_root="$1"
  local root="${ds_data_root%/}"
  local nested="${root}/scale_mix_all"
  local marker=""

  if [[ -d "${nested}" ]]; then
    for marker in train_g1_w_obj_prepared train_g1_w_obj_prepared_plus_omomo_orig train_g1_w_obj train_g1_w_obj_geometry; do
      if [[ -e "${nested}/${marker}" ]]; then
        printf '%s\n' "${nested}"
        return 0
      fi
    done
  fi

  for marker in train_g1_w_obj_prepared train_g1_w_obj_prepared_plus_omomo_orig train_g1_w_obj train_g1_w_obj_geometry; do
    if [[ -e "${root}/${marker}" ]]; then
      printf '%s\n' "${root}"
      return 0
    fi
  done

  printf '%s\n' "${root}"
}

ogds_default_motion_dir() {
  local ds_data_root="$1"
  local resolved_data_root
  local data_mode
  resolved_data_root="$(ogds_resolve_data_root "${ds_data_root}")"
  data_mode="$(ogds_normalize_data_mode "${2:-pure-sd}")"
  case "${data_mode}" in
    pure-sd)
      printf '%s\n' "${resolved_data_root%/}/train_g1_w_obj_prepared"
      ;;
    pure-real|mix-naive|mix-curriculum|fix-omomo-quater)
      printf '%s\n' "${resolved_data_root%/}/train_g1_w_obj_prepared_plus_omomo_orig"
      ;;
    *)
      return 1
      ;;
  esac
}

ogds_default_raw_motion_dir() {
  local ds_data_root="$1"
  local resolved_data_root
  resolved_data_root="$(ogds_resolve_data_root "${ds_data_root}")"
  printf '%s\n' "${resolved_data_root%/}/train_g1_w_obj"
}

ogds_default_geometry_dir() {
  local ds_data_root="$1"
  local resolved_data_root
  resolved_data_root="$(ogds_resolve_data_root "${ds_data_root}")"
  printf '%s\n' "${resolved_data_root%/}/train_g1_w_obj_geometry"
}

ogds_default_replay_subset_dir() {
  local motion_root="$1"
  local scope="$2"
  printf '%s_replay_%s\n' "${motion_root%/}" "${scope}"
}
