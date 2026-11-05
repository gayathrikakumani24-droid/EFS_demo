// Self-checking Testbench for fifo_sync
module fifo_sync_tb;
    parameter int DATA_WIDTH = 8;
    parameter int DEPTH      = 4;

    logic                  clk;
    logic                  rst_n;
    logic                  wr_en;
    logic [DATA_WIDTH-1:0] wr_data;
    logic                  rd_en;
    logic [DATA_WIDTH-1:0] rd_data;
    logic                  full;
    logic                  empty;
    logic                  almost_full;
    logic                  almost_empty;
    logic [$clog2(DEPTH+1)-1:0] count;

    fifo_sync #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .wr_data(wr_data),
        .rd_en(rd_en),
        .rd_data(rd_data),
        .full(full),
        .empty(empty),
        .almost_full(almost_full),
        .almost_empty(almost_empty),
        .count(count)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 0;
        wr_en = 0;
        rd_en = 0;
        wr_data = 8'h00;
        #20;
        rst_n = 1;
        #10;

        // Push 1 item
        wr_en = 1;
        wr_data = 8'hA5;
        @(posedge clk);
        wr_en = 0;
        @(posedge clk);

        // Pop item
        rd_en = 1;
        @(posedge clk);
        rd_en = 0;
        #1;
        if (rd_data !== 8'hA5) begin
            $display("ERROR: fifo_sync mismatch. Expected 8'hA5, got %h", rd_data);
            $finish(1);
        end else begin
            $display("PASS: fifo_sync verified successfully.");
            $finish(0);
        end
    end
endmodule
