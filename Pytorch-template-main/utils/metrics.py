from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def calculate_metrics(logger, metrics_list, eval_result):
    eval_result = tuple(eval_result)
    metrics_list = metrics_list.split("_")
    metrics_dict = {}
    for name in metrics_list:
        if name == "F1score":
            metrics_dict["F1score"] = calculate_f1score(eval_result)
        elif name == "Precision":
            metrics_dict["Precision"] = calculate_precision(eval_result)
        elif name == "Recall":
            metrics_dict["Recall"] = calculate_recall(eval_result)
        elif name == "ClassificationReport":
            metrics_dict = classification_report_own(eval_result)
        elif name == "StanceVAST":
            metrics_dict = evaluation_vast(eval_result)
        elif name == "StanceWT":
            metrics_dict = evaluation_wt(eval_result)
        else:
            logger.error(f"Please provide specific {name} functions")
    return metrics_dict


def calculate_f1score(eval_result):
    label, prediction = eval_result
    return round(f1_score(label, prediction, average="macro", zero_division=0), 6)


def calculate_precision(eval_result):
    label, prediction = eval_result
    return round(
        precision_score(label, prediction, average="macro", zero_division=0), 6
    )


def calculate_recall(eval_result):
    label, prediction = eval_result
    return round(recall_score(label, prediction, average="macro", zero_division=0), 6)


def classification_report_own(eval_result):
    label, prediction = eval_result
    report_dict = classification_report(
        label, prediction, zero_division=0, output_dict=True
    )
    metrics_dict = {}
    for key in report_dict:
        if key == "accuracy":
            metrics_dict["accuracy"] = report_dict[key]
        else:
            key_name = "_".join(key.split(" ")) + "_"
            for subkey in report_dict[key]:
                metrics_dict[key_name + subkey] = report_dict[key][subkey]
    return metrics_dict


def evaluation_vast(eval_result):
    all_labels, all_predicts, all_seens = eval_result
    metrics_dict = {}

    res_all = f1_score(
        all_labels, all_predicts, average=None, labels=[0, 1, 2], zero_division=0.0
    )
    metrics_dict["All_con"] = res_all[0]
    metrics_dict["All_pro"] = res_all[1]
    metrics_dict["All_neu"] = res_all[2]
    metrics_dict["All_avg"] = sum(res_all) / 3

    labels = [[], []]
    predicts = [[], []]
    for label, predict, seen in zip(all_labels, all_predicts, all_seens):
        labels[seen].append(label)
        predicts[seen].append(predict)

    res_few_shot = f1_score(
        labels[1], predicts[1], average=None, labels=[0, 1, 2], zero_division=0.0
    )

    metrics_dict["few_con"] = res_few_shot[0]
    metrics_dict["few_pro"] = res_few_shot[1]
    metrics_dict["few_neu"] = res_few_shot[2]
    metrics_dict["few_avg"] = sum(res_few_shot) / 3

    res_zero_shot = f1_score(
        labels[0], predicts[0], average=None, labels=[0, 1, 2], zero_division=0.0
    )

    metrics_dict["zero_con"] = res_zero_shot[0]
    metrics_dict["zero_pro"] = res_zero_shot[1]
    metrics_dict["zero_neu"] = res_zero_shot[2]
    metrics_dict["zero_avg"] = sum(res_zero_shot) / 3

    return metrics_dict


def evaluation_wt(eval_result):
    all_labels, all_predicts, all_seens = eval_result
    metrics_dict = {}

    res_all = f1_score(
        all_labels, all_predicts, average=None, labels=[0, 1, 2, 3], zero_division=0.0
    )
    metrics_dict["All_support"] = res_all[0]
    metrics_dict["All_refute"] = res_all[1]
    metrics_dict["All_comment"] = res_all[2]
    metrics_dict["All_unrelated"] = res_all[3]
    metrics_dict["All_avg"] = sum(res_all) / 4

    labels = [[], []]
    predicts = [[], []]
    for label, predict, seen in zip(all_labels, all_predicts, all_seens):
        labels[seen].append(label)
        predicts[seen].append(predict)

    res_few_shot = f1_score(
        labels[1], predicts[1], average=None, labels=[0, 1, 2, 3], zero_division=0.0
    )

    metrics_dict["few_support"] = res_few_shot[0]
    metrics_dict["few_refute"] = res_few_shot[1]
    metrics_dict["few_comment"] = res_few_shot[2]
    metrics_dict["few_unrelated"] = res_few_shot[3]
    metrics_dict["few_avg"] = sum(res_few_shot) / 4

    res_zero_shot = f1_score(
        labels[0], predicts[0], average=None, labels=[0, 1, 2, 3], zero_division=0.0
    )

    metrics_dict["zero_support"] = res_zero_shot[0]
    metrics_dict["zero_refute"] = res_zero_shot[1]
    metrics_dict["zero_comment"] = res_zero_shot[2]
    metrics_dict["zero_unrelated"] = res_zero_shot[3]
    metrics_dict["zero_avg"] = sum(res_zero_shot) / 4

    return metrics_dict
