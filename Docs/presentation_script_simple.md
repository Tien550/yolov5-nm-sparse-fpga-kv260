# Presentation Script — SIMPLE version (easy to memorize)
**Part:** Hardware → Results → Conclusions  (Speaker 2 = Tien)
**Style:** short sentences · easy words · point at the slide first, then say it.
`[chỉ ...]` = point at that part.  **💡 Nhớ** = the one line to remember for each slide.

---

## SLIDE 1 — Weight Decoder (COO / Bitmap)
**💡 Nhớ:** *"Two ways to pack, decoder picks the smaller one, output is always the same."*

Thank you, Dat. Good morning, everyone. I will now show the hardware, block by block.

*[chỉ bullet đầu]* The first block is the **weight decoder**. Its job is simple: it opens up the packed weights so the engine can use them.

*[chỉ hình COO/Bitmap]* We pack the weights in two ways. **COO**, on the left, just lists each non-zero weight with its position. **Bitmap**, on the right, uses a 16-bit map to mark the positions, then lists the values.

Why two ways? For very few weights, COO is smaller. For more weights, Bitmap is smaller. So the decoder always picks the smaller one — using one simple rule: use bitmap when N is 5 or more.

*[chỉ flowchart]* And no matter which one it picks, the output is always the same. So the rest of the engine never has to care.

---

## SLIDE 2 — I-SPEC (Compute Core)
**💡 Nhớ:** *"768 multipliers. Decoder sends a weight, address picks the input, we multiply and add. Zeros are skipped."*

*[chỉ ô 768]* This is the heart of the engine — the **I-SPEC**. It does the real math: multiply and add. It has **768 multipliers** — that is 12 clusters, times 4 groups, times 16.

*[rê theo sơ đồ từ trên xuống]* Here is how it works. At the top, we load three things: the settings, the input data, and the packed weights. The decoder on the right opens the weights and, every clock, sends out one weight and its address. *[chỉ mũi tên address]* That address picks the right input on the left. *[chỉ COMPUTE]* Then we multiply, and add everything up at the bottom.

*[chỉ ★ zeros skipped]* And the key idea: we only ever send real weights, never zeros. So no multiplier wastes time on a zero. **This is where the pruning becomes real speed.**

---

## SLIDE 3 — RTL System Integration (Vivado)
**💡 Nhớ:** *"One engine for all layer types, wrapped with AXI, driven by the CPU, tested before we trust it."*

*[chỉ bullet 1]* Now we put the engine into a full system. **One** engine handles all three layer types — 1×1, 3×3, and depthwise — just by setting a mode. That is about 23 modules and 7,000 lines of Verilog.

*[chỉ block design]* We wrap it with two AXI ports — one for control, one for data — and connect it to the ARM processor, shown in this Vivado design.

*[chỉ bullet cuối]* On the board, a Linux driver and a Python layer run the model one layer at a time. And before we trust it, we test every layer type against known-correct answers.

---

## SLIDE 4 — System Operational Dataflow
**💡 Nhớ:** *"CPU on top, FPGA on the bottom, they talk through DDR. Numbered arrows = the data's journey, repeated 59 times."*

*[chỉ sơ đồ]* This diagram shows the whole system working. Each arrow is numbered, in the order the data moves.

The top row is the processor. The bottom is the FPGA. They talk through the shared DDR memory in the middle.

*[rê theo các số]* Let me trace it. The image goes into memory. The processor sets up the FPGA over the control line. The DMA pulls the weights and data into on-chip buffers. Then the weights are decoded, sent to the 768 multipliers, summed, and rescaled. The result goes back to memory. *[chỉ layer_sequencer]* And the sequencer repeats this for all 59 layers.

So the processor conducts, and the FPGA does the heavy work.

---

## SLIDE 5 — Resource, Timing & Power
**💡 Nhớ:** *"Three questions: does it fit, is it fast, is it efficient? Yes, yes, yes."*

*[chỉ tiêu đề]* Now the results. Every hardware design must answer three questions: does it fit, is it fast, and is it efficient? This slide answers all three.

*[chỉ biểu đồ utilization]* Does it fit? Yes. We use about 63% of the logic and 64% of the DSPs. The memory is high, near 90% — but that is on purpose: we keep the data on-chip to stay fast.

*[chỉ timing summary]* Is it fast? Yes. We reached 200 megahertz, and the timing passes with room to spare.

*[chỉ power]* Is it efficient? Yes. The whole board uses only 4.4 watts.

---

## SLIDE 6 — Bit-Exact Verification & Detection
**💡 Nhớ:** *"Two things: the chip is exactly correct (100%), and here is what compression costs. The hardware adds no error."*

*[chỉ ô 100%]* This is our most important slide. It says two different things — please keep them apart.

*[chỉ bullet 1]* First: is the hardware correct? Yes. We tested all 59 layers against the software. Every single number matched — zero error. The chip is **bit-exact**.

*[chỉ bảng IoU]* Second, a different question: what does the compression cost? These numbers show how much the boxes move after pruning and after quantizing. *[chỉ 3 ảnh]* You can see it in these three pictures — the same parking lot, three versions.

*[nhấn]* Here is the key link: because the chip is bit-exact, it gives the **same** result as the computer. So the compression moves the boxes a little, but the hardware adds no error at all. *[chỉ ô 1.30 FPS]* On the board, we run at 1.3 frames per second.

---

## SLIDE 7 — Benchmark & Limitations
**💡 Nhớ:** *"We use less power and are more efficient — but it's a fair reference, not a winner. Here are our honest limits."*

*[chỉ bảng]* To give some context, we compare with a well-known design. Read across the table: we use fewer resources, about one-third the power, and much better efficiency.

But I will be honest — it is not a perfect comparison. Different models, different boards. So I show it as a reference, not as a winner.

*[chỉ danh sách limitations]* We also list our limits openly: moving data takes about half the time, the final head still runs on the CPU, and we tested on one dataset. These are exactly our next steps.

---

## SLIDE 8 — End-to-End Flow
**💡 Nhớ:** *"One image, camera to result. CPU prepares, FPGA runs 59 layers, CPU finishes. Teamwork."*

*[rê theo sơ đồ từ trên xuống]* This diagram ties everything together — one image, from the camera to the result.

An image comes in. The CPU prepares it. Then the FPGA runs all 59 layers — for each one, the CPU starts it, the engine computes, and signals done. *[chỉ khối CPU dưới]* After that, the CPU runs the final head and cleans up the boxes.

*[chỉ Output]* And out comes the result: boxes with labels. So it is teamwork — the FPGA does the heavy lifting, and the CPU runs the show.

---

## SLIDE 9 — Conclusion & Future Work
**💡 Nhớ:** *"Five goals, all done. Three next steps: faster data, more classes, a custom chip."*

*[chỉ cột Objectives]* To finish — our goals, and we met them all. We built the small model. We compressed it about 40 times. We built the accelerator. We deployed it, 100% bit-exact. And we closed timing at 200 megahertz.

*[chỉ cột Future Works]* For the future, three steps: overlap the data movement to go faster; support more classes and newer models like YOLOv8; and one day, move to a custom chip.

Thank you very much for your attention. We are happy to take any questions.

---

# How to memorize fast
1. Learn the **💡 Nhớ** line for each slide first — that is the skeleton.
2. Each slide = **point → say what it is → say why it matters.** Same pattern every time.
3. Say it out loud 3 times while looking at the slide. Your hand pointing helps your memory.
