from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Restriction import *
from Bio.Alphabet import IUPAC

# توالی ژنوم هدف برای ویرایش
sequence = Seq("ATGCGTACGTAGCTAGCTACGATCGATCGTAGCTAGCTA", IUPAC.unambiguous_dna)

# ایجاد sgRNA هدف
crispr_target = "CGTAGCTAGCTA"

# بررسی محل‌های برش Cas9
cas9 = RestrictionBatch([BsaI, EcoRI, BamHI])
sites = cas9.search(sequence)

print("محل‌های برش Cas9:")
for enzyme, positions in sites.items():
    print(f"{enzyme}: {positions}")

# ساخت رکورد برای خروجی
record = SeqRecord(sequence, id="TestGene", description="CRISPR target test sequence")

# ذخیره به فایل
with open("gene_sequence.fasta", "w") as output_handle:
    SeqIO.write(record, output_handle, "fasta")

print("توالی ژنوم ویرایش‌پذیر آماده و ذخیره شد.")
