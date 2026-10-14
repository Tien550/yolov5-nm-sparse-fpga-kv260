# Graduation Defense — Presentation Script v3 (Slide-Grounded, Detailed)
**Project:** Hardware Accelerator Design for YOLOv5 Object Detection on FPGA Using N:M Structured Sparsity Pruning
**Team:** Vu Tien Dat (22119006) · Nguyen Tran Tien (22119030) — Advisor: Assoc. Prof. Dr. Vo Minh Huan
**Speaker split:** Speaker 1 = **Dat** (slides 1–12) · Speaker 2 = **Tien** (slides 13–21)

> Every slide follows: **① orient to what is on screen → ② explain each element the audience sees, in detail → ③ end with one "big picture" sentence.**
> `[chỉ vào ...]` = stage direction (point at that part of the slide). Say the plain text in easy, natural English.
> Golden habit: never say a number the audience can't see — always tie your words to a box, a row, or a diagram on the slide.

---

## SLIDE 1 — Title  *(DAT)*
*[Đứng giữa, tự tin; chỉ vào tiêu đề đề tài khi đọc, rồi chỉ vào ảnh board]*

Good morning, respected members of the committee. Thank you all for taking the time to attend our defense today.

My name is **Vu Tien Dat**, and this is my teammate, **Nguyen Tran Tien** — our names and student IDs are shown here on the title slide. We are students of Computer Engineering Technology, in the Faculty of Advanced Education, and we completed this graduation project under the guidance of Associate Professor Doctor **Vo Minh Huan**.

The title of our project, in the center of the screen, is *"Hardware Accelerator Design for YOLOv5 Object Detection Network on FPGA Using N:M Structured Sparsity Pruning."* In simple words: we take a modern but heavy object-detection network, compress it step by step until it is small enough for a low-power chip, and then we design our own hardware to run it.

*[chỉ vào ảnh board bên phải]* On the right is a photo of our target hardware — the AMD Kria KV260 FPGA board. This is not a simulation: our final system actually runs on this physical board. So our work covers the full path, from the software model at the top, all the way down to real silicon. Let me begin with the roadmap of today's presentation.

---

## SLIDE 2 — Overview  *(DAT)*
*[chỉ lần lượt từng vòng tròn 01 → 05]*

This slide is the roadmap of our talk. The presentation is organized into five parts, and I will follow them in the order shown around this circle.

*[chỉ 01]* Part one, **Introduction**, explains our motivation — why this problem is worth solving — and states our objectives.
*[chỉ 02]* Part two, **Background**, gives the two key ideas we build on: the YOLOv5 network itself, and the theory of N:M structured sparsity.
*[chỉ 03]* Part three, **Implementation**, is the core and the longest part. It has two halves — the pipeline that compresses the model, and the hardware accelerator that runs it.
*[chỉ 04]* Part four, **Results**, presents the measured FPGA performance, the verification, and a comparison with other work.
*[chỉ 05]* And part five, **Conclusions**, summarizes what we achieved and where this work can go next.

The big picture is simple: parts one to three are *"how we built the system,"* and parts four and five are *"does it actually work."* I will present the first three parts myself, and then my teammate Tien will take over from the hardware section. To keep the flow smooth, we kindly ask you to hold your questions until the end.

---

## SLIDE 3 — 1. Introduction  *(DAT)*
*[chỉ tiêu đề "Why FPGA" bên trái]*

Let's begin with the motivation, on the left side of the slide, under the question *"Why FPGA for edge object detection?"*

*[chỉ bullet đầu]* The first point is the heart of the problem. When we run detection on a small device in real time, we need three things at the same time: high accuracy, low and predictable latency, and low power. The difficulty is that these three goals pull against each other — improving one often hurts another.

*[chỉ 3 chip ở đáy slide]* The three chips at the bottom compare the usual hardware choices against these goals. An **embedded GPU**, like a Jetson, gives high throughput, but it draws 10 to 15 watts, and because it runs on an operating system, its timing has jitter — it is not perfectly predictable. An **ARM CPU** is very power-efficient, but it simply does not have enough compute power to run a detection network in real time. The **FPGA** — highlighted on the right — is the balanced middle ground: it processes many operations in parallel, its latency is deterministic, and it is energy-efficient.

*[chỉ dòng "the gap"]* But there is one obstacle, written here as *"the gap."* The original YOLOv5s network needs 7.2 million parameters and 16.5 GFLOPs of computation — far more than our KV260 board can provide, with its limited logic, DSPs, and memory. So the model, as it stands, simply cannot fit.

*[chỉ 4 ô mục tiêu bên phải]* Closing that gap is exactly what this project does, and it defines our four objectives on the right. First, design a lightweight version of YOLOv5s sized for an edge FPGA. Second, build a compression pipeline that shrinks the model by at least ten times while preserving its accuracy. Third, design a single, unified accelerator that decodes our sparse weights natively, in hardware. And fourth, deploy the whole system on the real KV260 board and verify that it computes bit-exact results. These four objectives are the storyline for the rest of the talk.

---

## SLIDE 4 — 2. Background: YOLOv5s Architecture  *(DAT)*
*[chỉ bullet single-stage, rồi rê theo sơ đồ trái→phải]*

Before we compress the network, let me briefly explain what is inside it, because its structure directly shapes how we compress and accelerate it. This slide introduces YOLOv5.

*[chỉ bullet đầu]* Its most important property is that YOLOv5 is a **single-stage** detector. This means it finds the location of objects and classifies them in one single forward pass. Older two-stage detectors are more complex and slower; the single-stage design is what gives YOLO its speed.

*[rê tay theo sơ đồ]* Now let me walk through the diagram at the bottom, from left to right, because it has three parts. On the left is the **Backbone** — this part looks at the raw image and gradually extracts features, from simple edges up to high-level shapes; it is built from these Conv and C3 blocks and ends with the SPP block here. In the middle is the **Neck**, which mixes features from different depths together, so that both very small and very large objects end up well described. And on the right is the **Head**, which makes the final predictions at three different scales — the three output boxes here, 80×80, 40×40, and 20×20, dedicated to small, medium, and large objects.

*[chỉ dòng baseline cost]* Finally, the last line gives the cost of this network: at a 416-by-416 input, about 7.2 million parameters and around 7 GFLOPs. The big picture: this is an accurate, well-designed network — but it is simply too heavy for our board, so the entire next section is about making it lighter without making it worse.

---

## SLIDE 5 — 2. Background: N:M Structured Sparsity  *(DAT)*
*[chỉ cột trái "The Pruning Spectrum"]*

This slide answers a key design question: of all the ways to prune a network, why did we choose N:M sparsity? Pruning means deliberately removing the weights that contribute the least. But the *pattern* in which you remove them matters enormously for hardware.

*[chỉ lần lượt 3 hộp trái]* On the left is what we call the pruning spectrum — three options, from most flexible to most rigid. The top box, **unstructured pruning**, removes individual weights from anywhere. It can reach very high sparsity with the best accuracy — but, as the red note underneath says, the surviving weights are scattered randomly, so the hardware needs index tables to find them, and that overhead cancels the speedup on an FPGA. The middle box, **channel pruning**, removes entire channels at once. The result stays dense and is easy for hardware, but it is far too coarse — removing whole channels throws away useful information, and accuracy drops sharply.

*[chỉ hộp N:M + dòng xanh]* Our choice is the bottom box, **N:M structured pruning**, and it takes the best of both. The idea: we look at the weights in small groups, and in every group we keep only the strongest few and set the rest to zero. So it is as fine-grained as unstructured pruning, yet it follows a fixed, regular pattern like channel pruning. The green line captures it: it balances both — the sweet spot for FPGA acceleration.

*[chỉ cột phải "Why N:M Suits FPGAs"]* On the right, the slide lists three concrete reasons this suits an FPGA: the decode logic is fixed, so there is no branching; each surviving weight maps directly onto one processing element; and only the surviving weights travel from memory, which cuts the memory traffic.

*[chỉ hộp "Prior Art" + công thức bên trái]* And this box notes where we stand relative to prior art: NVIDIA's GPUs already use a fixed two-out-of-four pattern, but that is one single hard-coded ratio. Our work generalizes this to a variable N, chosen individually for each layer. The formula on the left shows the relationship — sparsity equals M minus N, over M — so keeping just 2 of every 16 weights means removing 87.5 percent.

---

## SLIDE 6 — 2. Background: DominoSearch  *(DAT)*
*[chỉ câu mở đầu, rồi 5 hộp đánh số]*

This slide explains **DominoSearch**, the algorithm that decides how aggressively to prune each individual layer.

*[chỉ câu đầu]* The sentence at the top states the motivation. If we pruned every layer by the same fixed amount, we would waste the layers that tolerate pruning well, and at the same time damage the fragile ones. So instead, DominoSearch searches for a separate N for each layer — and it does this purely from the statistics of the weights and the hardware cost, with no need to retrain the whole network from scratch.

*[chỉ lần lượt 5 hộp]* The five numbered boxes show how it works. Step one, scoring: we measure how important each column of weights is, using its average magnitude. Step two, an adaptive threshold: instead of one global cutoff, each block of sixteen weights gets its own threshold, so it adapts to the local distribution. Step three, the vote: we count how many columns in the block pass that threshold, and that count becomes the N for the block. Step four, snap to hardware: we round N to one of the values our hardware actually supports — the set shown here. And step five, the most important, greedy descent: we start from the fully dense network, then lower N one step at a time, always choosing the layer that can best afford it, and we repeat this until the whole model fits within our fixed hardware budget.

The big picture, and the reason for the name "Domino": we do not decide everything at once — we topple the pruning one careful step at a time, starting dense and coming down only where the network can spare it.

---

## SLIDE 7 — 3. Implementation: Compression Pipeline  *(DAT)*
*[chỉ timeline 3 giai đoạn]*

We now enter the Implementation section, and this slide is the overview of our compression pipeline. We shrink the model in three stages, shown on this timeline, and each stage attacks the size from a completely different angle.

*[chỉ 01 + factor]* Stage one, **lightweight architecture**, redesigns the structure of the network itself with lighter building blocks — and, as the factor below it shows, this gives roughly 1.4 to 1.5 times reduction. *[chỉ 02]* Stage two, **INT8 quantization**, changes the number format from decimals to compact 8-bit integers, contributing another 2.3 times. *[chỉ 03]* Stage three, **N:M sparsity**, prunes the weights in that regular pattern, adding a further 3.56 times.

*[chỉ dải tối ở đáy]* The key message is that these three stages *multiply* together. Each one, on its own, is safe for accuracy — but stacked, as the dark band at the bottom shows, they achieve about 40 times weight compression while still keeping 98.7 percent of the original accuracy. The next slides open up each of these three stages, in this exact order.

---

## SLIDE 8 — 3. Implementation: Stage 1 — YOLOv5s-Lite  *(DAT)*
*[chỉ 3 bullet Mod, rồi vào khối C3_DWS cam]*

Here is stage one in detail. On the left, the slide lists the three specific modifications we made to turn the original YOLOv5s into our lighter version.

*[chỉ Mod 1 + khối cam trên sơ đồ]* The first modification is inside the network's main blocks: we replace the standard, expensive convolution with a depthwise-separable convolution. Instead of doing all the work in one costly step, this splits it into two much cheaper steps that together do almost the same job. You can see the result as the new **C3_DWS** blocks, colored orange, in the diagram on the right. *[chỉ Mod 2]* The second modification reduces the depth of the network — we keep just one bottleneck unit inside each block, instead of the original three. *[chỉ Mod 3]* The third is specifically for hardware: we replace the original smooth activation with LeakyReLU, which, as the note says, the chip computes with a single constant multiply and a small shift — just one DSP, with only about 0.1 percent error.

*[chỉ lần lượt 3 ô số]* The three boxes below give the outcome: the parameters drop from 7.2 to 5.15 million, the computation from about 7 down to 4.68 GFLOPs, and the final Lite model has 63 convolutional layers. I want to be honest on one point: this stage gives only a *moderate* shrink. The really dramatic compression comes in the next two stages — the true purpose of this first stage is to make the network friendly for the hardware.

---

## SLIDE 9 — 3. Implementation: Dataset & Stage-1 Results  *(DAT)*
*[chỉ bullets + 2 ảnh, rồi bảng phải]*

This slide shows the dataset we used, and how well the model performed after stage one.

*[chỉ bullets + 2 ảnh bãi xe]* The bullets on the left describe the data. We used **PKLot**, a real-world dataset of over twelve thousand parking-lot images, collected from three different sites, under varied weather and lighting — you can see two typical examples here. The task is binary: for each parking space, decide whether it is empty or occupied. We split the data properly into training, validation, and test sets, and trained for 100 epochs with the standard settings listed.

*[chỉ bảng bên phải, rê theo từng dòng]* The table on the right reports the result of this first stage. The precision and recall are both close to 0.99, the mAP at 0.5 is 0.9945, and the whole model is only 2.9 megabytes. *[chỉ dòng chú thích dưới bảng]* As the note under the table concludes, these near-perfect numbers confirm something important: even after we made the network lightweight in stage one, it kept its full ability to detect on this task. In other words, stage one cost us essentially no accuracy.

---

## SLIDE 10 — 3. Implementation: Stage 2 — INT8 QAT  *(DAT)*
*[chỉ 3 bullet, rồi bảng, rồi ô xanh]*

Stage two is quantization — converting the model from decimal, floating-point numbers into 8-bit integers. This matters for hardware because integer arithmetic is far smaller, faster, and cheaper on a chip than floating-point.

*[chỉ bullets]* The bullets explain how we do it carefully. We use 8-bit integers, with a separate scale per channel for the weights. Using the Brevitas framework, we first calibrate the model, and then we fine-tune it while it is *aware* of the rounding, so it learns to compensate for the error. We also fold the batch-normalization layers into the convolutions, so they add no cost at run time.

*[chỉ hàng PTQ rồi hàng QAT]* The table proves why this care is necessary. Look at the middle row, plain post-training quantization — it drops the stricter accuracy noticeably. But the bottom row, quantization-aware training, recovers it almost completely: the mAP returns to 0.9943, essentially the same as the floating-point row just above it. *[chỉ biểu đồ trái + ô xanh]* The chart on the left shows the payoff in size — the model shrinks from 2.9 down to 1.28 megabytes. And the blue box states the headline: our training recovers 99.98 percent of the original accuracy, with only a two-hundredths drop.

---

## SLIDE 11 — 3. Implementation: Stage 3 — N:M Pruning  *(DAT)*
*[chỉ Rule, rồi 2 heatmap, rồi 3 ô số]*

Stage three is the N:M pruning itself, now applied to the real model and guided by DominoSearch.

*[chỉ Rule]* The rule, at the top-left, is the one from the theory: within every block of sixteen weights, we keep only the N largest and zero out the rest. DominoSearch assigns that N individually for each layer, based on how sensitive the layer is.

*[chỉ 2 heatmap bên phải]* These two heatmaps are the evidence for that sensitivity. Each cell shows how well a layer holds its accuracy when pruned — red means the layer breaks, green means it tolerates pruning easily. *[chỉ hàng model.1/2/4 đỏ]* Notice that the earliest layers — model one, two, and four — are red, so DominoSearch wisely keeps them fully dense. Most of the other layers are green, so they can be pruned very hard. *[chỉ biểu đồ cột dưới]* The bar chart below plots the final N chosen for every single layer, and you can see most of them are pruned down to just a few weights.

*[chỉ 3 ô số + dòng encoding]* On the left, the three boxes give the summary: 54 configured layers, an overall sparsity of 71.9 percent, and an additional 3.56 times compression. And the note lists how we store the surviving weights — 35 layers use one compact format called COO, and 16 use another called bitmap. That storage detail will matter directly in the hardware section.

---

## SLIDE 12 — 3. Implementation: Cumulative Compression  *(DAT)* → HANDOVER
*[chỉ bảng phải, đọc từng hàng; rồi 4 biểu đồ; rồi 4 ô số]*

This final software slide brings all three stages together into one table, so you can see the model shrink step by step.

*[rê theo từng hàng]* Reading from the top: the original network has 7.2 million parameters and a 14-megabyte weight file. After the lightweight redesign, 5.15 million parameters. After INT8 quantization, the file drops to 1.28 megabytes. And after N:M pruning, the file is only 0.36 megabytes, with about 1.16 million weights actually non-zero, at an mAP of 0.987.

*[chỉ 4 biểu đồ bên trái]* The four small charts on the left show these same journeys visually — the parameters, the computation, the file size, and the accuracy, all across the three stages. *[chỉ 4 ô số dưới]* And the four boxes give the headline figures: about 40 times smaller weights, roughly 1.5 times fewer operations from the architecture, 3.56 times from sparsity, and 98.7 percent of the accuracy preserved.

This is the central claim of the entire software side: the model became dramatically smaller without becoming meaningfully worse. Now that the model is small enough to fit our board, I will hand over to my teammate Tien, who will explain the hardware we built to run it. Tien, please.

---

## SLIDE 13 — 3. Accelerator: Weight Decoder  *(TIEN)*
*[nhận mic; chỉ hình COO/Bitmap, rồi flowchart]*

Thank you, Dat, and good morning, everyone. I will now present the hardware accelerator, and I will walk you through it block by block, following the data from input to output.

*[chỉ bullet đầu]* The first block is the **weight decoder** — its job is to unpack the compressed weights, because the compute engine cannot use them while they are still packed. We store the weights in two formats, and the decoder picks one per layer , depending by N.

*[chỉ hình minh hoạ COO vs Bitmap]* This picture shows the two formats on one block of 16 weights, where only three are non-zero. **COO** simply lists each non-zero weight as an index-and-value pair. **Bitmap** keeps a 16-bit mask that marks the non-zero positions, followed by the packed values — so here COO costs 36 bits, bitmap costs 40, and the decoder always keeps whichever is smaller.

*[rê theo flowchart]* The flowchart on the right shows the flow: a block arrives, we check its N, take the COO or the bitmap path, and both merge into one unified output — so the compute array downstream never has to know which format was used.

---

## SLIDE 14 — 3. Accelerator: I-SPEC Compute Core  *(TIEN)*
*[chỉ ô "768", rồi rê theo sơ đồ từ trên xuống]*

This slide shows the heart of the accelerator — the block we call the **I-SPEC** — where the actual multiplication and addition happen.

*[chỉ bullet 1 + ô "768"]* The name means an *input-channel sparse processing cluster* — in plain terms, one weight decoder plus an array of compute units. As the box shows, 12 of these clusters, times 4 groups, times 16 units, give us **768 MAC units** in total, all working in parallel.

*[rê theo sơ đồ từ trên xuống dưới]* Now let me trace the diagram, because it shows exactly how one tile is computed. At the top we load three things: the configuration, the input activations, and the compressed weights. On the right, the weight decoders unpack those weights and, every clock cycle, produce one address-and-weight pair. *[chỉ mũi tên addr sang trái]* That address tells the block on the left which activation to pick — so we only ever fetch the activations we actually need. *[chỉ khối Computation]* The pair then flows down into the multiplier array, and each result is added into the 64 partial sums at the bottom, which accumulate across the input channels.


---

## SLIDE 15 — 3. Accelerator: System Integration  *(TIEN)*
*[chỉ bullet 1 + ô số, rồi block design]*

To turn that engine into a complete, working system, we integrated everything, as shown on this slide.

*[chỉ bullet 1 + ô "23" và "~7,000"]* The first bullet describes the core: a single, *unified* kernel that can perform all three types of convolution — one-by-one, three-by-three, and depthwise — just by setting a mode register. We built one flexible engine rather than a separate circuit for each type, which is what keeps it small enough to fit; it is about 23 modules and 7,000 lines of hand-written Verilog, the numbers in these boxes. *[chỉ bullet 2 + block design]* The second bullet: we packaged this engine with standard AXI interfaces — one channel for control commands and one for bulk data — and connected it to the ARM processor in the Vivado block design you see here.

*[chỉ bullet 3, 4]* On the board, a Linux driver together with a Python layer drives the inference one layer at a time. And before we ever trust the results, we test every layer type against known-correct reference outputs. So by the end of this slide, we have a complete, and independently verified, accelerator running on real silicon — which brings us to the results.

---

## SLIDE 16 — 4. Results: Resource, Timing & Power  *(TIEN)*
*[chỉ bảng, rồi 3 ô số]*

Now the results. Any hardware design has to answer three basic questions — does it fit, does it run fast enough, and does it stay efficient — and this slide answers all three at once.

*[rê theo bảng]* The table shows how much of the chip we use. We use about 63 percent of the logic and 64 percent of the DSP units, so we still have comfortable headroom there. *[chỉ 2 dòng BRAM/URAM]* The on-chip memory, however, is used heavily — between 88 and 94 percent. That is a deliberate design choice: we keep the weights and data on-chip so we avoid slow, repeated trips to external memory.

*[chỉ 3 ô số]* The three boxes below answer the other two questions. We reached our full target clock of 200 megahertz — and, crucially, the timing analysis passes with a positive margin, meaning the design is solid, not just barely working. The entire board consumes only 4.375 watts, and the peak efficiency is about 94 GOPS per watt. So, in one sentence: it fits, it is fast, and it is very efficient.

---

## SLIDE 17 — 4. Results: Verification & On-Board Detection  *(TIEN)*
*[chỉ bullet 1 + ô 100%; rồi bảng IoU + 3 ảnh]*

This is, for us, the most important slide of all — and it makes two separate claims that are easy to confuse, so I will keep them clearly apart.

*[chỉ bullet 1 + ô "100%"]* The first claim is about the correctness of the hardware. As the bullets say, we wrote 14 dedicated testbenches, from the smallest unit up to the full system, and the hardware matched our trusted software reference exactly — on all 59 layers, with zero error. That is what the "100%" box means: the FPGA is **bit-exact**. Whatever the model is supposed to compute, the chip computes the identical value.

*[chỉ bảng IoU + 3 ảnh detection]* The second claim is a completely different question — how much the *compression itself* changes the detections. This table on the right measures that at each step: first the effect of pruning, then the effect of quantization. And the three images below make it visible — the same parking lot, detected by the original dense model, then the pruned model, then the final INT8 model; you can see the boxes shift a little. *[nhấn mạnh]* The crucial link between the two claims is this: because the hardware is bit-exact, the result on the FPGA is identical to the compressed model on a computer. So the compression moves the boxes slightly, but the hardware itself adds absolutely no error. *[chỉ ô "1.30 FPS"]* And running the whole thing end to end on the board, we measure 1.30 frames per second.

---

## SLIDE 18 — 4. Results: Benchmark & Limitations  *(TIEN)*
*[chỉ bảng so sánh, rồi disclaimer, rồi limitations]*

To put our work in context, this slide compares our accelerator against a well-known N:M design from the literature.

*[rê theo các hàng bảng]* Reading across the table: the other design uses more MAC units than ours, and far more logic — 84 percent of its chip, against our 62 percent. It also uses about three times the power, and its efficiency is several times lower than ours. So on the metrics that matter most at the edge — resource cost and power — our design compares very favorably.

*[chỉ dòng disclaimer]* That said, I want to be a fair engineer about it. As the note states, this is not a perfectly equal comparison: the two designs run different networks, on different-sized boards. So I present it as a useful reference point, not as a knockout. *[chỉ 4 bullet limitations]* And we state our own limitations openly, because they point directly to our future work: data movement currently takes about half of the total time; our configuration is fixed to one setting; the final detection head still runs on the CPU; and we have validated on a single dataset so far.

---

## SLIDE 19 — 5. Conclusions  *(TIEN)*
*[chỉ cột Objectives, rồi cột Future Works]*

To conclude, this slide places our original objectives next to what we actually achieved.

*[rê theo 5 mục có dấu ✓]* On the left, every objective is met, marked with a check. We designed the lightweight model. We aimed for at least ten times compression and achieved about forty. We built the unified accelerator with native N:M decoding. We deployed it on the KV260 and verified it is 100 percent bit-exact. And we closed timing at 200 megahertz.

*[chỉ cột Future Works bên phải]* On the right is our future work, and it follows directly from those limitations I just mentioned. First, overlap the data movement with the computation, so the engine spends less time waiting. Second, support more object classes and move to a newer model such as YOLOv8. And third, in the longer term, migrate the design toward a custom ASIC chip for mass production. Let me finish with one diagram that ties the whole system together.

---

## SLIDE 20 — 5. Conclusions: End-to-End Flow  *(TIEN)*
*[rê tay theo sơ đồ từ trên xuống]*

This last diagram shows the complete system in motion — the full journey of a single image, from the camera to the final result.

*[chỉ từng khối]* Let me trace it from top to bottom. First, an image comes in. Then the CPU prepares it — resizing it and converting it into the integer form the chip expects. Next comes the main work: the FPGA runs the 59 layers, shown in the middle — that is 43 one-by-one convolutions, 8 three-by-three, and 8 depthwise. For each of those layers, the CPU writes the configuration and starts the engine; the engine then loads its data, computes, stores the result, and signals that it is done. *[chỉ khối CPU dưới]* After all the layers, the CPU runs the four final output layers — the detection head — and cleans up any overlapping boxes. *[chỉ khối OUTPUT]* And at the very bottom is the output we wanted from the start: the objects in the image, drawn as boxes with class labels.

The big picture is a true hardware-software co-design: the FPGA does the heavy computational lifting, while the CPU conducts the process and finishes it off. That completes our presentation.

---

## SLIDE 21 — Thank You  *(TIEN)*

Thank you very much for your kind attention. To summarize everything in a single sentence: we took a large object-detection network, compressed it in three stages, designed our own hardware to run it, and proved on a real FPGA board that every layer is computed exactly right — efficiently, and at low power. We would now be very glad to answer any questions the committee may have.

---

# ⏱ TIMING NOTE
This detailed, slide-grounded version runs about **25–30 minutes** at a calm, measured pace. Two habits keep you on time and clear:
1. **Point, then speak.** Touch the element on the slide first, then explain it — never read a number that isn't visible.
2. **Pause after each "big picture" sentence.** That is where the committee absorbs the idea; a two-second silence there is worth more than extra words.
If you run short of time on the day, the safe places to trim are the middle explanations of slides 5, 6, and 18 — but keep every "big picture" closing sentence, because those carry the argument.
