/-- Statement of Graham's conjecture (which is now a theorem in the literature).

Graham's conjecture states that if $0 < a_1 < \dots a_n$ are integers, then
$\max_{i, j} \frac{a_i}{\gcd(a_i, a_j)} \ge n$. -/
def GrahamConjecture (n : ℕ) (f : ℕ → ℕ) : Prop :=
  n ≠ 0 → StrictMonoOn f (Set.Iio n) → ∃ i < n, ∃ j < n, (f i).gcd (f j) * n ≤ f i


/-- The special case of Graham's conjecture where all numbers are squarefree. -/
lemma grahamConjecture_of_squarefree {n : ℕ} (f : ℕ → ℕ) (hf' : ∀ k < n, Squarefree (f k)) :
    GrahamConjecture n f := by
  /-
    n : Nat
    f : Nat → Nat
    hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
    ⊢ n.GrahamConjecture f
  -/
  rintro hn hf
  /-
    n : Nat
    f : Nat → Nat
    hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
    hn : Ne n 0
    hf : StrictMonoOn f (Set.Iio n)
    ⊢ Exists fun i => And (LT.lt i n) (Exists fun j => And (LT.lt j n) (LE.le (HMu …
  -/
  by_contra!
  /-
    n : Nat
    f : Nat → Nat
    hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
    hn : Ne n 0
    hf : StrictMonoOn f (Set.Iio n)
    this : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → LT.lt (f i) (HMul.hMu …
    ⊢ False
  -/
  set 𝒜 := (Iio n).image fun n ↦ primeFactors (f n)
  have hf'' : ∀ i < n, ∀ j, Squarefree (f i / (f i).gcd (f j)) :=
    fun i hi j ↦ (hf' _ hi).squarefree_of_dvd <| div_dvd_of_dvd <| gcd_dvd_left _ _
  /-
    n : Nat
    f : Nat → Nat
    hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
    hn : Ne n 0
    hf : StrictMonoOn f (Set.Iio n)
    this : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → LT.lt (f i) (HMul.hMu …
    𝒜 : Finset (Finset Nat) := Finset.image (fun n => (f n).primeFactors) (Finset. …
    hf'' : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), Squarefree (HDiv.hDiv (f i) ((f i …
    ⊢ False
  -/
  refine lt_irrefl n ?_
  calc
    n = #𝒜 := ?_
    _ ≤ #(𝒜 \\ 𝒜) := 𝒜.card_le_card_diffs
    _ ≤ #(Ioo 0 n) := card_le_card_of_injOn (fun s ↦ ∏ p ∈ s, p) ?_ ?_
    _ = n - 1 := by rw [card_Ioo, tsub_zero]
    _ < n := tsub_lt_self hn.bot_lt zero_lt_one
    /-
      case calc_1
      n : Nat
      f : Nat → Nat
      hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
      hn : Ne n 0
      hf : StrictMonoOn f (Set.Iio n)
      this : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → LT.lt (f i) (HMul.hMu …
      𝒜 : Finset (Finset Nat) := Finset.image (fun n => (f n).primeFactors) (Finset. …
      hf'' : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), Squarefree (HDiv.hDiv (f i) ((f i …
      ⊢ Eq n 𝒜.card
    -/
  · rw [Finset.card_image_of_injOn, card_Iio]
    /-
      case calc_1
      n : Nat
      f : Nat → Nat
      hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
      hn : Ne n 0
      hf : StrictMonoOn f (Set.Iio n)
      this : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → LT.lt (f i) (HMul.hMu …
      𝒜 : Finset (Finset Nat) := Finset.image (fun n => (f n).primeFactors) (Finset. …
      hf'' : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), Squarefree (HDiv.hDiv (f i) ((f i …
      ⊢ Set.InjOn (fun n => (f n).primeFactors) ↑(Finset.Iio n)
    -/
    simpa using prod_primeFactors_invOn_squarefree.2.injOn.comp hf.injOn hf'
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      n : Nat
      f : Nat → Nat
      hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
      hn : Ne n 0
      hf : StrictMonoOn f (Set.Iio n)
      this : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → LT.lt (f i) (HMul.hMu …
      𝒜 : Finset (Finset Nat) := Finset.image (fun n => (f n).primeFactors) (Finset. …
      hf'' : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), Squarefree (HDiv.hDiv (f i) ((f i …
      ⊢ ∀ (a : Finset Nat), Membership.mem (𝒜.diffs 𝒜) a → Membership.mem (Finset.Io …
    -/
  · simp only [𝒜, forall_mem_diffs, forall_mem_image, mem_Ioo, mem_Iio]
    /-
      case calc_2
      n : Nat
      f : Nat → Nat
      hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
      hn : Ne n 0
      hf : StrictMonoOn f (Set.Iio n)
      this : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → LT.lt (f i) (HMul.hMu …
      𝒜 : Finset (Finset Nat) := Finset.image (fun n => (f n).primeFactors) (Finset. …
      hf'' : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), Squarefree (HDiv.hDiv (f i) ((f i …
      ⊢ ∀ ⦃x : Nat⦄, LT.lt x n → ∀ ⦃x_1 : Nat⦄, LT.lt x_1 n → And (LT.lt 0 ((SDiff.s …
    -/
    rintro i hi j hj
    rw [← primeFactors_div_gcd (hf' _ hi) (hf' _ hj).ne_zero,
      prod_primeFactors_of_squarefree <| hf'' _ hi _]
    exact ⟨Nat.div_pos (gcd_le_left _ (hf' _ hi).ne_zero.bot_lt) <|
      Nat.gcd_pos_of_pos_left _ (hf' _ hi).ne_zero.bot_lt, Nat.div_lt_of_lt_mul <| this _ hi _ hj⟩
    /-
      case calc_3
      n : Nat
      f : Nat → Nat
      hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
      hn : Ne n 0
      hf : StrictMonoOn f (Set.Iio n)
      this : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → LT.lt (f i) (HMul.hMu …
      𝒜 : Finset (Finset Nat) := Finset.image (fun n => (f n).primeFactors) (Finset. …
      hf'' : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), Squarefree (HDiv.hDiv (f i) ((f i …
      ⊢ Set.InjOn (fun s => s.prod fun p => p) ↑(𝒜.diffs 𝒜)
    -/
  · simp only [𝒜, Set.InjOn, mem_coe, forall_mem_diffs, forall_mem_image, mem_Ioo, mem_Iio]
    /-
      case calc_3
      n : Nat
      f : Nat → Nat
      hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
      hn : Ne n 0
      hf : StrictMonoOn f (Set.Iio n)
      this : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → LT.lt (f i) (HMul.hMu …
      𝒜 : Finset (Finset Nat) := Finset.image (fun n => (f n).primeFactors) (Finset. …
      hf'' : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), Squarefree (HDiv.hDiv (f i) ((f i …
      ⊢ ∀ ⦃x : Nat⦄, LT.lt x n → ∀ ⦃x_1 : Nat⦄, LT.lt x_1 n → ∀ ⦃x_2 : Nat⦄, LT.lt x …
    -/
    rintro a ha b hb c hc d hd
    rw [← primeFactors_div_gcd (hf' _ ha) (hf' _ hb).ne_zero, ← primeFactors_div_gcd
      (hf' _ hc) (hf' _ hd).ne_zero, prod_primeFactors_of_squarefree (hf'' _ ha _),
      prod_primeFactors_of_squarefree (hf'' _ hc _)]
    /-
      case calc_3
      n : Nat
      f : Nat → Nat
      hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
      hn : Ne n 0
      hf : StrictMonoOn f (Set.Iio n)
      this : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → LT.lt (f i) (HMul.hMu …
      𝒜 : Finset (Finset Nat) := Finset.image (fun n => (f n).primeFactors) (Finset. …
      hf'' : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), Squarefree (HDiv.hDiv (f i) ((f i …
      a : Nat
      ha : LT.lt a n
      b : Nat
      hb : LT.lt b n
      c : Nat
      hc : LT.lt c n
      d : Nat
      hd : LT.lt d n
      ⊢ Eq (HDiv.hDiv (f a) ((f a).gcd (f b))) (HDiv.hDiv (f c) ((f c).gcd (f d))) → …
    -/
    rintro h
    /-
      case calc_3
      n : Nat
      f : Nat → Nat
      hf' : ∀ (k : Nat), LT.lt k n → Squarefree (f k)
      hn : Ne n 0
      hf : StrictMonoOn f (Set.Iio n)
      this : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → LT.lt (f i) (HMul.hMu …
      𝒜 : Finset (Finset Nat) := Finset.image (fun n => (f n).primeFactors) (Finset. …
      hf'' : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), Squarefree (HDiv.hDiv (f i) ((f i …
      a : Nat
      ha : LT.lt a n
      b : Nat
      hb : LT.lt b n
      c : Nat
      hc : LT.lt c n
      d : Nat
      hd : LT.lt d n
      h : Eq (HDiv.hDiv (f a) ((f a).gcd (f b))) (HDiv.hDiv (f c) ((f c).gcd (f d)))
      ⊢ Eq (HDiv.hDiv (f a) ((f a).gcd (f b))).primeFactors (HDiv.hDiv (f c) ((f c). …
    -/
    rw [h]
    /-
      🎉 no goals
    -/


