// Verified Standard Library Module: Synchronous FIFO
// Parameterized width & depth with status flags (full, empty, almost_full, almost_empty)
module fifo_sync #(
    parameter int DATA_WIDTH = 32,
    parameter int DEPTH      = 16,
    parameter int ALMOST_FULL_THRESH  = 14,
    parameter int ALMOST_EMPTY_THRESH = 2
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  wr_en,
    input  logic [DATA_WIDTH-1:0] wr_data,
    input  logic                  rd_en,
    output logic [DATA_WIDTH-1:0] rd_data,
    output logic                  full,
    output logic                  empty,
    output logic                  almost_full,
    output logic                  almost_empty,
    output logic [$clog2(DEPTH+1)-1:0] count
);

    localparam int ADDR_WIDTH = (DEPTH > 1) ? $clog2(DEPTH) : 1;

    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    logic [ADDR_WIDTH-1:0] wr_ptr;
    logic [ADDR_WIDTH-1:0] rd_ptr;
    logic [$clog2(DEPTH+1)-1:0] fifo_count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr     <= '0;
            rd_ptr     <= '0;
            fifo_count <= '0;
            rd_data    <= '0;
        end else begin
            // Write operation
            if (wr_en && (!full || rd_en)) begin
                mem[wr_ptr] <= wr_data;
                wr_ptr      <= (wr_ptr == DEPTH - 1) ? '0 : wr_ptr + 1'b1;
            end

            // Read operation
            if (rd_en && !empty) begin
                rd_data <= mem[rd_ptr];
                rd_ptr  <= (rd_ptr == DEPTH - 1) ? '0 : rd_ptr + 1'b1;
            end

            // Counter update
            case ({wr_en && (!full || rd_en), rd_en && !empty})
                2'b10:   fifo_count <= fifo_count + 1'b1;
                2'b01:   fifo_count <= fifo_count - 1'b1;
                default: fifo_count <= fifo_count;
            endcase
        end
    end

    assign count        = fifo_count;
    assign empty        = (fifo_count == '0);
    assign full         = (fifo_count == DEPTH);
    assign almost_empty = (fifo_count <= ALMOST_EMPTY_THRESH);
    assign almost_full  = (fifo_count >= ALMOST_FULL_THRESH);

endmodule
