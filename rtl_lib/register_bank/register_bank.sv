// Verified Standard Library Module: Parameterized CSR Register Bank
// Supports memory-mapped read/write with byte-enable strobes and address decoding.
module register_bank #(
    parameter int ADDR_WIDTH = 8,
    parameter int DATA_WIDTH = 32,
    parameter int NUM_REGS   = 8
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  wr_en,
    input  logic [ADDR_WIDTH-1:0] addr,
    input  logic [DATA_WIDTH-1:0] wdata,
    input  logic [(DATA_WIDTH/8)-1:0] wstrb,
    input  logic                  rd_en,
    output logic [DATA_WIDTH-1:0] rdata,
    output logic [DATA_WIDTH-1:0] reg_out [0:NUM_REGS-1],
    input  logic [DATA_WIDTH-1:0] reg_in  [0:NUM_REGS-1]
);

    logic [DATA_WIDTH-1:0] registers [0:NUM_REGS-1];
    localparam int STRIDE = (DATA_WIDTH == 64) ? 8 : 4;
    localparam int IDX_WIDTH = (NUM_REGS > 1) ? $clog2(NUM_REGS) : 1;
    wire [IDX_WIDTH-1:0] reg_idx = (addr / STRIDE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_REGS; i++) begin
                registers[i] <= '0;
            end
            rdata <= '0;
        end else begin
            // Write with byte strobes
            if (wr_en && (reg_idx < NUM_REGS)) begin
                for (int b = 0; b < DATA_WIDTH/8; b++) begin
                    if (wstrb[b]) begin
                        registers[reg_idx][b*8 +: 8] <= wdata[b*8 +: 8];
                    end
                end
            end

            // Read
            if (rd_en) begin
                if (reg_idx < NUM_REGS) begin
                    rdata <= registers[reg_idx] | reg_in[reg_idx];
                end else begin
                    rdata <= '0;
                end
            end
        end
    end

    // Direct output connections for internal status/control
    generate
        for (genvar g = 0; g < NUM_REGS; g++) begin : gen_outputs
            assign reg_out[g] = registers[g];
        end
    endgenerate

endmodule
