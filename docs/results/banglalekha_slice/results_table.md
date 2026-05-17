# BanglaLekha slice run — bridged GiT + LoRA

Init strategy: `donor`
Vocab: 30522 → 59577 (+29055)
Trainable: 109,513,401 (33.11%)
Train / val: 200 / 20
Steps: 300 (batch 4, lr 0.0001)
Wall: 124.6s

## Evaluation points

| Step | Train loss | Val loss |
|---:|---:|---:|
| 100 | 5.576 | 4.055 |
| 200 | 3.393 | 3.778 |
| 300 | 2.930 | 3.624 |

## Sample generations (final eval point)

**4.png**
- Reference: `ছয় জন মানুষ দাড়িয়ে আছে।`
- Generated: `মানুষ আছে`

**31.png**
- Reference: `একটি বাচ্চা ছেলে বসে আছে।`
- Generated: `মানুষ আছে`

**41.png**
- Reference: `কয়েক জন মানুষ দাঁড়িয়ে ও কয়েক জন মানুষ বসে আছে।`
- Generated: `মানুষ আছে`

**45.png**
- Reference: `জলাশয় এর পাশে গাছ আছে। উপরে নীল আকাশে অল্প সাদা মেঘ আছে।`
- Generated: `পুরুষ মানুষ আছে`

**48.png**
- Reference: `একজন মহিলা ও একটি কিশোর ছেলে বাচ্চা কোলে নিয়ে বসে আছে।`
- Generated: `পুরুষ মানুষ আছে`

