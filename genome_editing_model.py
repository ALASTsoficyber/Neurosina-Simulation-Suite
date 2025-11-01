import random
import string
import matplotlib.pyplot as plt

# شبیه‌سازی رشته DNA مصنوعی و ویرایش کریسپر متناسب با پروژه نوروسایبرنتیک تو

# تولید رشته DNA فرضی (10000 نوکلئوتید)
def generate_dna_sequence(length=10000):
    return ''.join(random.choices('ACGT', k=length))

# موقعیت‌های هدف ویرایش (gene editing sites)
edit_sites = [1500, 3200, 5800, 7700]

# عملکرد ویرایشگر کریسپر — برش و جایگزینی نوکلئوتید
def crispr_edit(dna_seq, sites, new_base='G'):
    dna_list = list(dna_seq)
    for site in sites:
        original_base = dna_list[site]
        dna_list[site] = new_base
        print(f'Edited site {site}: {original_base} → {new_base}')
    return ''.join(dna_list)

# اجرای شبیه‌سازی
dna_seq = generate_dna_sequence()
edited_dna_seq = crispr_edit(dna_seq, edit_sites)

# شمارش فراوانی نوکلئوتیدها قبل و بعد از ویرایش
def count_bases(dna_seq):
    return {base: dna_seq.count(base) for base in 'ACGT'}

counts_before = count_bases(dna_seq)
counts_after = count_bases(edited_dna_seq)

# نمایش تغییرات فراوانی نوکلئوتید
labels = ['A', 'C', 'G', 'T']
x = range(len(labels))
plt.bar(x, [counts_before[b] for b in labels], width=0.4, label='Before', align='center')
plt.bar([i+0.4 for i in x], [counts_after[b] for b in labels], width=0.4, label='After', align='center')
plt.xticks(x, labels)
plt.ylabel('Count')
plt.title('Base Frequency Before and After CRISPR Editing')
plt.legend()
plt.show()
