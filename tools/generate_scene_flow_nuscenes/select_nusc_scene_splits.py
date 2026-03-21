import argparse
import json
import os
from collections import Counter, defaultdict

from nuscenes.nuscenes import NuScenes


KEYWORDS = ("night", "rain", "construction", "intersection", "turn")
KEYWORD_WEIGHTS = {
    "night": 4,
    "rain": 3,
    "construction": 2,
    "intersection": 1,
    "turn": 1,
}
LOCATION_QUOTAS = {
    "train": {
        "boston-seaport": 57,
        "singapore-onenorth": 22,
        "singapore-queenstown": 15,
        "singapore-hollandvillage": 11,
    },
    "val": {
        "boston-seaport": 12,
        "singapore-onenorth": 5,
        "singapore-queenstown": 3,
        "singapore-hollandvillage": 2,
    },
    "test": {
        "boston-seaport": 13,
        "singapore-onenorth": 5,
        "singapore-queenstown": 3,
        "singapore-hollandvillage": 2,
    },
}
KEYWORD_MINIMA = {
    # With fixed location quotas and no cross-split log sharing, train-night=12 is infeasible.
    # The minimal feasible relaxation is train-night=11.
    "train": {"night": 11, "rain": 20, "construction": 14, "intersection": 31, "turn": 27},
    "val": {"night": 3, "rain": 5, "construction": 3, "intersection": 7, "turn": 5},
    "test": {"night": 3, "rain": 5, "construction": 3, "intersection": 7, "turn": 6},
}
WORKING_LOG_CAPS = {
    # The original 3/2/2 cap plan is infeasible together with the fixed location quotas
    # and night minima, because night scenes are concentrated in a handful of logs.
    "train": 4,
    "val": 3,
    "test": 3,
}
SPLIT_ORDER = ["test", "val", "train"]
DEFAULT_SPLIT_SIZES = {"train": 105, "val": 22, "test": 23}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="./Datasets/nuscenes/", type=str)
    parser.add_argument("--version", default="v1.0-trainval", type=str)
    parser.add_argument("--num_train", default=105, type=int)
    parser.add_argument("--num_val", default=22, type=int)
    parser.add_argument("--num_test", default=23, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--output_dir",
        default="./tools/generate_scene_flow_nuscenes/scene_lists",
        type=str,
    )
    return parser.parse_args()


def validate_args(args):
    requested_sizes = {
        "train": args.num_train,
        "val": args.num_val,
        "test": args.num_test,
    }
    if requested_sizes != DEFAULT_SPLIT_SIZES:
        raise ValueError(
            "This selector currently implements the fixed 150-scene plan only: "
            f"expected {DEFAULT_SPLIT_SIZES}, got {requested_sizes}."
        )


def build_scene_records(nusc):
    records = []
    for scene_index, scene in enumerate(nusc.scene):
        log = nusc.get("log", scene["log_token"])
        description = scene["description"].lower()
        record = {
            "scene_index": scene_index,
            "scene_name": scene["name"],
            "scene_token": scene["token"],
            "log_token": scene["log_token"],
            "location": log["location"],
            "nbr_samples": scene["nbr_samples"],
            "description": scene["description"],
        }
        for keyword in KEYWORDS:
            record[keyword] = keyword in description
        records.append(record)
    return records


def candidate_gain(record, keyword_counts, keyword_targets):
    gain = 0
    for keyword in KEYWORDS:
        if record[keyword] and keyword_counts[keyword] < keyword_targets[keyword]:
            gain += KEYWORD_WEIGHTS[keyword]
    return gain


def candidate_penalty(record, keyword_counts, keyword_targets):
    penalty = 0
    for keyword in KEYWORDS:
        if record[keyword] and keyword_counts[keyword] >= keyword_targets[keyword]:
            penalty += KEYWORD_WEIGHTS[keyword]
    return penalty


def candidate_log_support(records, candidate_record, selected_scene_names, required_keyword=None):
    support = 0
    for record in records:
        if record["scene_name"] in selected_scene_names:
            continue
        if record["log_token"] != candidate_record["log_token"]:
            continue
        if required_keyword is not None and not record[required_keyword]:
            continue
        support += 1
    return support


def is_candidate_allowed(
    record,
    split_name,
    selected_scene_names,
    log_assignment,
    split_log_counts,
    location_counts,
    log_caps,
):
    if record["scene_name"] in selected_scene_names:
        return False

    if location_counts[record["location"]] >= LOCATION_QUOTAS[split_name][record["location"]]:
        return False

    assigned_split = log_assignment.get(record["log_token"])
    if assigned_split is not None and assigned_split != split_name:
        return False

    if split_log_counts[record["log_token"]] >= log_caps[split_name]:
        return False

    return True


def choose_candidate(
    records,
    split_name,
    selected_scene_names,
    log_assignment,
    split_log_counts,
    location_counts,
    keyword_counts,
    keyword_targets,
    log_caps,
    required_keyword=None,
):
    candidates = []
    for record in records:
        if not is_candidate_allowed(
            record,
            split_name,
            selected_scene_names,
            log_assignment,
            split_log_counts,
            location_counts,
            log_caps,
        ):
            continue
        if required_keyword is not None and not record[required_keyword]:
            continue

        gain = candidate_gain(record, keyword_counts, keyword_targets)
        penalty = candidate_penalty(record, keyword_counts, keyword_targets)
        log_support = candidate_log_support(
            records,
            record,
            selected_scene_names,
            required_keyword=required_keyword,
        )
        remaining_location_capacity = (
            LOCATION_QUOTAS[split_name][record["location"]] - location_counts[record["location"]]
        )
        candidates.append(
            (
                -gain,
                penalty,
                -remaining_location_capacity,
                -log_support,
                -split_log_counts[record["log_token"]],
                record["scene_name"],
                record,
            )
        )

    if not candidates:
        return None

    candidates.sort()
    return candidates[0][-1]


def select_split(records, split_name, selected_scene_names, log_assignment, log_caps, initial_records=None):
    location_counts = Counter()
    keyword_counts = Counter()
    split_log_counts = Counter()
    selected_records = []
    keyword_targets = KEYWORD_MINIMA[split_name]
    split_size = sum(LOCATION_QUOTAS[split_name].values())

    for record in sorted(initial_records or [], key=lambda x: x["scene_name"]):
        if not is_candidate_allowed(
            record,
            split_name,
            selected_scene_names,
            log_assignment,
            split_log_counts,
            location_counts,
            log_caps,
        ):
            raise RuntimeError(
                f"Invalid preselected record {record['scene_name']} for split={split_name!r}"
            )
        commit_candidate(
            record,
            split_name,
            selected_records,
            selected_scene_names,
            log_assignment,
            split_log_counts,
            location_counts,
            keyword_counts,
        )

    for required_keyword in KEYWORDS:
        while keyword_counts[required_keyword] < keyword_targets[required_keyword]:
            candidate = choose_candidate(
                records,
                split_name,
                selected_scene_names,
                log_assignment,
                split_log_counts,
                location_counts,
                keyword_counts,
                keyword_targets,
                log_caps,
                required_keyword=required_keyword,
            )
            if candidate is None:
                raise RuntimeError(
                    f"Failed to satisfy keyword={required_keyword!r} for split={split_name!r}"
                )
            commit_candidate(
                candidate,
                split_name,
                selected_records,
                selected_scene_names,
                log_assignment,
                split_log_counts,
                location_counts,
                keyword_counts,
            )

    while len(selected_records) < split_size:
        candidate = choose_candidate(
            records,
            split_name,
            selected_scene_names,
            log_assignment,
            split_log_counts,
            location_counts,
            keyword_counts,
            keyword_targets,
            log_caps,
            required_keyword=None,
        )
        if candidate is None:
            raise RuntimeError(f"Failed to fill split={split_name!r} to size {split_size}")
        commit_candidate(
            candidate,
            split_name,
            selected_records,
            selected_scene_names,
            log_assignment,
            split_log_counts,
            location_counts,
            keyword_counts,
        )

    expected_location_counts = LOCATION_QUOTAS[split_name]
    if dict(location_counts) != expected_location_counts:
        raise RuntimeError(
            f"Location quota mismatch for split={split_name!r}: "
            f"expected {expected_location_counts}, got {dict(location_counts)}"
        )

    for keyword, target in keyword_targets.items():
        if keyword_counts[keyword] < target:
            raise RuntimeError(
                f"Keyword quota mismatch for split={split_name!r}, keyword={keyword!r}: "
                f"expected at least {target}, got {keyword_counts[keyword]}"
            )

    return selected_records


def commit_candidate(
    record,
    split_name,
    selected_records,
    selected_scene_names,
    log_assignment,
    split_log_counts,
    location_counts,
    keyword_counts,
):
    selected_records.append(record)
    selected_scene_names.add(record["scene_name"])
    log_assignment[record["log_token"]] = split_name
    split_log_counts[record["log_token"]] += 1
    location_counts[record["location"]] += 1
    for keyword in KEYWORDS:
        keyword_counts[keyword] += int(record[keyword])


def summarize_split(records):
    location_counts = Counter(record["location"] for record in records)
    keyword_counts = Counter()
    for record in records:
        for keyword in KEYWORDS:
            keyword_counts[keyword] += int(record[keyword])

    return {
        "num_scenes": len(records),
        "num_logs": len({record["log_token"] for record in records}),
        "location_counts": dict(sorted(location_counts.items())),
        "keyword_counts": dict(sorted(keyword_counts.items())),
        "estimated_keyframe_pairs": int(sum(max(record["nbr_samples"] - 1, 0) for record in records)),
        "scene_names": sorted(record["scene_name"] for record in records),
    }


def validate_no_cross_split(split_records):
    log_to_split = {}
    for split_name, records in split_records.items():
        for record in records:
            previous_split = log_to_split.setdefault(record["log_token"], split_name)
            if previous_split != split_name:
                raise RuntimeError(
                    f"log_token {record['log_token']} appears in both {previous_split} and {split_name}"
                )


def validate_outputs(split_records):
    total_records = sum(len(records) for records in split_records.values())
    if total_records != 150:
        raise RuntimeError(f"Expected 150 selected scenes, got {total_records}")

    for split_name, expected_size in DEFAULT_SPLIT_SIZES.items():
        if len(split_records[split_name]) != expected_size:
            raise RuntimeError(
                f"Split {split_name!r} size mismatch: expected {expected_size}, got {len(split_records[split_name])}"
            )

    validate_no_cross_split(split_records)


def pick_records_from_log(records, log_token, *, keyword_value, count, selected_scene_names):
    picked_records = []
    for record in sorted(records, key=lambda x: x["scene_name"]):
        if record["scene_name"] in selected_scene_names:
            continue
        if record["log_token"] != log_token:
            continue
        if record["night"] != keyword_value:
            continue
        picked_records.append(record)
        if len(picked_records) == count:
            return picked_records
    raise RuntimeError(
        f"Unable to pick {count} records from log={log_token} with night={keyword_value}"
    )


def build_preselected_records(records):
    records_by_log = defaultdict(list)
    for record in records:
        records_by_log[record["log_token"]].append(record)

    night_logs_by_location = defaultdict(list)
    non_night_logs_by_location = defaultdict(list)
    for log_token, log_records in records_by_log.items():
        location = log_records[0]["location"]
        if any(record["night"] for record in log_records):
            night_logs_by_location[location].append((log_token, sum(int(record["night"]) for record in log_records)))
        else:
            non_night_logs_by_location[location].append((log_token, len(log_records)))

    hv_night_logs = sorted(
        night_logs_by_location["singapore-hollandvillage"],
        key=lambda x: (-x[1], x[0]),
    )
    qs_night_logs = sorted(
        night_logs_by_location["singapore-queenstown"],
        key=lambda x: (-x[1], x[0]),
    )
    hv_non_night_logs = sorted(
        non_night_logs_by_location["singapore-hollandvillage"],
        key=lambda x: (-x[1], x[0]),
    )

    if len(hv_night_logs) < 4 or len(qs_night_logs) < 2 or len(hv_non_night_logs) < 1:
        raise RuntimeError("Night-scene preselection assumptions are not satisfied for nuScenes metadata.")

    preselected = {"train": [], "val": [], "test": []}
    selected_scene_names = set()

    test_qs_log = qs_night_logs[0][0]
    val_qs_log = qs_night_logs[1][0]
    test_hv_non_night_log = hv_non_night_logs[0][0]
    val_hv_night_log = hv_night_logs[-1][0]
    train_hv_night_logs = [log_token for log_token, _ in hv_night_logs[:-1]]

    preselected["test"].extend(
        pick_records_from_log(records, test_qs_log, keyword_value=True, count=3, selected_scene_names=selected_scene_names)
    )
    selected_scene_names.update(record["scene_name"] for record in preselected["test"])
    preselected["test"].extend(
        pick_records_from_log(
            records,
            test_hv_non_night_log,
            keyword_value=False,
            count=2,
            selected_scene_names=selected_scene_names,
        )
    )
    selected_scene_names.update(record["scene_name"] for record in preselected["test"])

    preselected["val"].extend(
        pick_records_from_log(records, val_qs_log, keyword_value=True, count=1, selected_scene_names=selected_scene_names)
    )
    selected_scene_names.update(record["scene_name"] for record in preselected["val"])
    preselected["val"].extend(
        pick_records_from_log(
            records,
            val_hv_night_log,
            keyword_value=True,
            count=2,
            selected_scene_names=selected_scene_names,
        )
    )
    selected_scene_names.update(record["scene_name"] for record in preselected["val"])

    train_night_counts = [4, 4, 3]
    for log_token, count in zip(train_hv_night_logs, train_night_counts):
        picked = pick_records_from_log(
            records,
            log_token,
            keyword_value=True,
            count=count,
            selected_scene_names=selected_scene_names,
        )
        preselected["train"].extend(picked)
        selected_scene_names.update(record["scene_name"] for record in picked)

    return preselected


def write_scene_lists(output_dir, split_records, log_caps_used, seed):
    os.makedirs(output_dir, exist_ok=True)
    total_num_scenes = sum(len(records) for records in split_records.values())

    all_records = []
    for split_name in ["train", "val", "test"]:
        all_records.extend(split_records[split_name])
    all_scene_names = sorted(record["scene_name"] for record in all_records)

    all_list_path = os.path.join(output_dir, f"scene_{total_num_scenes}_all.txt")
    with open(all_list_path, "w") as file:
        file.write("\n".join(all_scene_names) + "\n")

    split_list_paths = {}
    for split_name, records in split_records.items():
        split_scene_names = sorted(record["scene_name"] for record in records)
        split_list_path = os.path.join(output_dir, f"scene_{total_num_scenes}_{split_name}.txt")
        with open(split_list_path, "w") as file:
            file.write("\n".join(split_scene_names) + "\n")
        split_list_paths[split_name] = split_list_path

    stats = {
        "total_num_scenes": total_num_scenes,
        "seed": seed,
        "log_caps_used": log_caps_used,
        "list_paths": {
            "all": all_list_path,
            **split_list_paths,
        },
        "splits": {
            split_name: summarize_split(records)
            for split_name, records in split_records.items()
        },
    }
    stats_path = os.path.join(output_dir, f"scene_{total_num_scenes}_stats.json")
    with open(stats_path, "w") as file:
        json.dump(stats, file, indent=2, sort_keys=True)

    return stats_path


def main():
    args = parse_args()
    validate_args(args)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    records = build_scene_records(nusc)
    preselected_records = build_preselected_records(records)
    selected_scene_names = set()
    log_assignment = {}
    split_records = {}
    used_log_caps = WORKING_LOG_CAPS.copy()

    for split_name in SPLIT_ORDER:
        split_records[split_name] = select_split(
            records,
            split_name,
            selected_scene_names,
            log_assignment,
            used_log_caps.copy(),
            initial_records=preselected_records[split_name],
        )

    stats_path = write_scene_lists(args.output_dir, split_records, used_log_caps, args.seed)
    print(f"Saved scene lists and stats to {args.output_dir}")
    print(f"Stats file: {stats_path}")


if __name__ == "__main__":
    main()
