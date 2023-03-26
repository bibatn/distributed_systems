"""
This is a boilerplate pipeline
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import make_predictions, report_accuracy, split_data, read_data, compute, chose_and_write


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=read_data,
                inputs=["meteo_data", "parameters"],
                outputs="data",
                name="read",
            ),
            node(
                func=compute,
                inputs=["data", "params:number_nodes", "params:node1"],
                outputs=["p1", "p2", "avg_sq_dev1", "avg1", "point1"],
                name="compute_1",
            ),
            node(
                func=compute,
                inputs=["data", "params:number_nodes", "params:node2"],
                outputs=["p3", "p4", "avg_sq_dev2", "avg2", "point2"],
                name="compute_2",
            ),
            node(
                func=chose_and_write,
                inputs=["p1", "p2", "p3", "p4", "avg_sq_dev1", "avg_sq_dev2", "avg1", "avg2", "point1", "point2"],
                outputs=None,
                name="final_report"
            ),
            # node(
            #     func=split_data,
            #     inputs=["example_iris_data", "parameters"],
            #     outputs=["X_train", "X_test", "y_train", "y_test"],
            #     name="split",
            # ),
            # node(
            #     func=make_predictions,
            #     inputs=["X_train", "X_test", "y_train"],
            #     outputs="y_pred",
            #     name="make_predictions",
            # ),
            # node(
            #     func=report_accuracy,
            #     inputs=["y_pred", "y_test"],
            #     outputs=None,
            #     name="report_accuracy",
            # ),
        ]
    )
