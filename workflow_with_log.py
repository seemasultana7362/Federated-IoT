from logging import INFO

from flwr.common import Context, log
from flwr.server import Grid, LegacyContext

from flwr.server.workflow.secure_aggregation.secaggplus_workflow import (
    SecAggPlusWorkflow,
    WorkflowState,
)


class SecAggPlusWorkflowWithLogs(SecAggPlusWorkflow):
    """
    SecAgg+ workflow with explicit stage logging.
    """

    def __call__(self, grid: Grid, context: Context) -> None:
        log(
            INFO,
            "",
        )

        log(
            INFO,
            "============================================================",
        )

        log(
            INFO,
            "SECURE AGGREGATION START",
        )

        log(
            INFO,
            "============================================================",
        )

        super().__call__(grid, context)

        log(
            INFO,
            "============================================================",
        )

        log(
            INFO,
            "SECURE AGGREGATION END",
        )

        log(
            INFO,
            "============================================================",
        )

    def setup_stage(
        self,
        grid: Grid,
        context: LegacyContext,
        state: WorkflowState,
    ) -> bool:

        log(
            INFO,
            "[SECAGG+] Stage 0: SETUP",
        )

        result = super().setup_stage(
            grid,
            context,
            state,
        )

        log(
            INFO,
            "[SECAGG+] Stage 0 complete",
        )

        return result

    def share_keys_stage(
        self,
        grid: Grid,
        context: LegacyContext,
        state: WorkflowState,
    ) -> bool:

        log(
            INFO,
            "[SECAGG+] Stage 1: SHARE KEYS",
        )

        result = super().share_keys_stage(
            grid,
            context,
            state,
        )

        log(
            INFO,
            "[SECAGG+] Stage 1 complete",
        )

        return result

    def collect_masked_vectors_stage(
        self,
        grid: Grid,
        context: LegacyContext,
        state: WorkflowState,
    ) -> bool:

        log(
            INFO,
            "[SECAGG+] Stage 2: COLLECT MASKED VECTORS",
        )

        result = super().collect_masked_vectors_stage(
            grid,
            context,
            state,
        )

        log(
            INFO,
            "[SECAGG+] Stage 2 complete",
        )

        return result

    def unmask_stage(
        self,
        grid: Grid,
        context: LegacyContext,
        state: WorkflowState,
    ) -> bool:

        log(
            INFO,
            "[SECAGG+] Stage 3: UNMASK",
        )

        result = super().unmask_stage(
            grid,
            context,
            state,
        )

        log(
            INFO,
            "[SECAGG+] Stage 3 complete",
        )

        return result