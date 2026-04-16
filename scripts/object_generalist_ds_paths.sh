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
    *)
      printf '%s\n' "${mode}"
      ;;
  esac
}

ogds_default_motion_dir() {
  local ds_data_root="$1"
  local data_mode
  data_mode="$(ogds_normalize_data_mode "${2:-pure-sd}")"
  case "${data_mode}" in
    pure-sd)
      printf '%s\n' "${ds_data_root%/}/train_g1_w_obj_prepared"
      ;;
    pure-real|mix-naive|mix-curriculum)
      printf '%s\n' "${ds_data_root%/}/train_g1_w_obj_prepared_plus_omomo_orig"
      ;;
    *)
      return 1
      ;;
  esac
}

ogds_default_replay_subset_dir() {
  local motion_root="$1"
  local scope="$2"
  printf '%s_replay_%s\n' "${motion_root%/}" "${scope}"
}
