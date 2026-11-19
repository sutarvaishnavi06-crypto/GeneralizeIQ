
def analyze_model(row):
    train_acc = row["Train Accuracy"]
    val_acc = row["Validation Accuracy"]
    gap = row["Generalization Gap"]
    test_acc = row["Test Accuracy"]

    insights = []

    if train_acc < 0.80 and val_acc < 0.80:
        insights.append(
            ("Underfitting Risk",
             "Both training and validation performance are relatively low.")
        )

    elif gap > 0.03:
        insights.append(
            ("Overfitting Risk",
             "Training performance exceeds validation performance significantly.")
        )

    elif abs(gap) <= 0.01:
        insights.append(
            ("Strong Generalization Alignment",
             "Training and validation accuracy are closely aligned.")
        )

    else:
        insights.append(
            ("Moderate Generalization",
             "The model shows some train-validation divergence.")
        )

    if test_acc >= 0.88:
        insights.append(
            ("Strong Test Performance",
             "The model performs well on unseen test data.")
        )
    elif test_acc >= 0.85:
        insights.append(
            ("Good Test Performance",
             "The model has reasonable unseen-data performance.")
        )
    else:
        insights.append(
            ("Test Performance Needs Improvement",
             "The model may require further tuning.")
        )

    return insights
