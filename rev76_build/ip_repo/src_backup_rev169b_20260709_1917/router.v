// router.v  --  Flexible NoC router: unicast/multicast/broadcast to N_SPEC=4 SPECs
// Paper: "Routers" Fig. 8 (CLAUDE.md §8.16)
//
// Routes one DATA bus (8-bit weight or activation byte) to a selected subset of
// N_SPEC=4 SPEC clusters. Configured by ARM via router_mode + sel register.
//
// Three modes:
//   UNICAST   (2'b00): one source → one destination (sel[1:0] picks which SPEC)
//   MULTICAST (2'b01): one source → subset of SPECs (sel[3:0] bitmask)
//   BROADCAST (2'b10): one source → all 4 SPECs simultaneously
//
// ARM guarantees no conflicts; no arbitration needed.
// This module routes a single byte; caller instantiates one per data bus width.

`timescale 1ns/1ps

module router (
    input  wire [1:0]  router_mode, // 00=UNICAST 01=MULTICAST 10=BROADCAST
    input  wire [3:0]  sel,         // UNICAST: sel[1:0]=dest index; MULTICAST: bitmask
    input  wire [7:0]  data_in,     // source data
    input  wire        valid_in,    // source valid

    output wire [7:0]  data_out_0,  // SPEC 0 data
    output wire [7:0]  data_out_1,  // SPEC 1 data
    output wire [7:0]  data_out_2,  // SPEC 2 data
    output wire [7:0]  data_out_3,  // SPEC 3 data
    output wire        valid_out_0, // SPEC 0 valid
    output wire        valid_out_1,
    output wire        valid_out_2,
    output wire        valid_out_3
);

    localparam UNICAST   = 2'b00;
    localparam MULTICAST = 2'b01;
    localparam BROADCAST = 2'b10;

    // Data routing: all destinations always receive data_in (mux is on valid)
    assign data_out_0 = data_in;
    assign data_out_1 = data_in;
    assign data_out_2 = data_in;
    assign data_out_3 = data_in;

    // Valid routing per mode
    wire uc_en_0 = (sel[1:0] == 2'd0);
    wire uc_en_1 = (sel[1:0] == 2'd1);
    wire uc_en_2 = (sel[1:0] == 2'd2);
    wire uc_en_3 = (sel[1:0] == 2'd3);

    wire en_0 = (router_mode == BROADCAST) ? 1'b1 :
                (router_mode == MULTICAST) ? sel[0] :
                                              uc_en_0;

    wire en_1 = (router_mode == BROADCAST) ? 1'b1 :
                (router_mode == MULTICAST) ? sel[1] :
                                              uc_en_1;

    wire en_2 = (router_mode == BROADCAST) ? 1'b1 :
                (router_mode == MULTICAST) ? sel[2] :
                                              uc_en_2;

    wire en_3 = (router_mode == BROADCAST) ? 1'b1 :
                (router_mode == MULTICAST) ? sel[3] :
                                              uc_en_3;

    assign valid_out_0 = valid_in & en_0;
    assign valid_out_1 = valid_in & en_1;
    assign valid_out_2 = valid_in & en_2;
    assign valid_out_3 = valid_in & en_3;

endmodule
