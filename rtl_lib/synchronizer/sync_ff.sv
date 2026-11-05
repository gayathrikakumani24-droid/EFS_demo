// Verified Standard Library Module: Multi-Stage Synchronizer
// Latency: STAGES cycles. Metastability mitigation for CDC signals.
module sync_ff #(
    parameter int STAGES = 2,
    parameter int WIDTH  = 1,
    parameter bit RESET_VAL = 1'b0
)(
    input  logic             clk,
    input  logic             rst_n,
    input  logic [WIDTH-1:0] din,
    output logic [WIDTH-1:0] dout
);

    logic [(STAGES*WIDTH)-1:0] stage_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage_reg <= {(STAGES*WIDTH){RESET_VAL}};
        end else begin
            stage_reg <= {stage_reg[(STAGES-1)*WIDTH-1:0], din};
        end
    end

    assign dout = stage_reg[(STAGES*WIDTH)-1:(STAGES-1)*WIDTH];

endmodule
