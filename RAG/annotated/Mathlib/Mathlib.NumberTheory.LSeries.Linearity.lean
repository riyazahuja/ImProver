lemma LSeries.term_add (f g : ℕ → ℂ) (s : ℂ) : term (f + g) s = term f s + term g s := by
  /-
    f g : Nat → Complex
    s : Complex
    ⊢ Eq (LSeries.term (HAdd.hAdd f g) s) (HAdd.hAdd (LSeries.term f s) (LSeries.t …
  -/
  ext ⟨- | n⟩
    /-
      case h.zero
      f g : Nat → Complex
      s : Complex
      ⊢ Eq (LSeries.term (HAdd.hAdd f g) s 0) (HAdd.hAdd (LSeries.term f s) (LSeries …
    -/
  · simp only [term_zero, Pi.add_apply, add_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      f g : Nat → Complex
      s : Complex
      n✝ : Nat
      ⊢ Eq (LSeries.term (HAdd.hAdd f g) s (HAdd.hAdd n✝ 1)) (HAdd.hAdd (LSeries.ter …
    -/
  · simp only [term_of_ne_zero (Nat.succ_ne_zero _), Pi.add_apply, add_div]
    /-
      🎉 no goals
    -/


lemma LSeries.term_add_apply (f g : ℕ → ℂ) (s : ℂ) (n : ℕ) :
    term (f + g) s n = term f s n + term g s n := by
  /-
    f g : Nat → Complex
    s : Complex
    n : Nat
    ⊢ Eq (LSeries.term (HAdd.hAdd f g) s n) (HAdd.hAdd (LSeries.term f s n) (LSeri …
  -/
  rw [term_add, Pi.add_apply]
  /-
    🎉 no goals
  -/


lemma LSeriesHasSum.add {f g : ℕ → ℂ} {s a b : ℂ} (hf : LSeriesHasSum f s a)
    (hg : LSeriesHasSum g s b) :
    LSeriesHasSum (f + g) s (a + b) := by
  /-
    f g : Nat → Complex
    s a b : Complex
    hf : LSeriesHasSum f s a
    hg : LSeriesHasSum g s b
    ⊢ LSeriesHasSum (HAdd.hAdd f g) s (HAdd.hAdd a b)
  -/
  simpa only [LSeriesHasSum, term_add] using HasSum.add hf hg
  /-
    🎉 no goals
  -/


lemma LSeriesSummable.add {f g : ℕ → ℂ} {s : ℂ} (hf : LSeriesSummable f s)
    (hg : LSeriesSummable g s) :
    LSeriesSummable (f + g) s := by
  /-
    f g : Nat → Complex
    s : Complex
    hf : LSeriesSummable f s
    hg : LSeriesSummable g s
    ⊢ LSeriesSummable (HAdd.hAdd f g) s
  -/
  simpa only [LSeriesSummable, ← term_add_apply] using Summable.add hf hg
  /-
    🎉 no goals
  -/


@[simp]
lemma LSeries_add {f g : ℕ → ℂ} {s : ℂ} (hf : LSeriesSummable f s) (hg : LSeriesSummable g s) :
    LSeries (f + g) s = LSeries f s + LSeries g s := by
  /-
    f g : Nat → Complex
    s : Complex
    hf : LSeriesSummable f s
    hg : LSeriesSummable g s
    ⊢ Eq (LSeries (HAdd.hAdd f g) s) (HAdd.hAdd (LSeries f s) (LSeries g s))
  -/
  simpa only [LSeries, term_add, Pi.add_apply] using tsum_add hf hg
  /-
    🎉 no goals
  -/


lemma LSeries.term_neg (f : ℕ → ℂ) (s : ℂ) : term (-f) s = -term f s := by
  /-
    f : Nat → Complex
    s : Complex
    ⊢ Eq (LSeries.term (Neg.neg f) s) (Neg.neg (LSeries.term f s))
  -/
  ext ⟨- | n⟩
    /-
      case h.zero
      f : Nat → Complex
      s : Complex
      ⊢ Eq (LSeries.term (Neg.neg f) s 0) (Neg.neg (LSeries.term f s) 0)
    -/
  · simp only [term_zero, Pi.neg_apply, neg_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      f : Nat → Complex
      s : Complex
      n✝ : Nat
      ⊢ Eq (LSeries.term (Neg.neg f) s (HAdd.hAdd n✝ 1)) (Neg.neg (LSeries.term f s) …
    -/
  · simp only [term_of_ne_zero (Nat.succ_ne_zero _), Pi.neg_apply, Nat.cast_succ, neg_div]
    /-
      🎉 no goals
    -/


lemma LSeries.term_neg_apply (f : ℕ → ℂ) (s : ℂ) (n : ℕ) : term (-f) s n = -term f s n := by
  /-
    f : Nat → Complex
    s : Complex
    n : Nat
    ⊢ Eq (LSeries.term (Neg.neg f) s n) (Neg.neg (LSeries.term f s n))
  -/
  rw [term_neg, Pi.neg_apply]
  /-
    🎉 no goals
  -/


lemma LSeriesHasSum.neg {f : ℕ → ℂ} {s a : ℂ} (hf : LSeriesHasSum f s a) :
    LSeriesHasSum (-f) s (-a) := by
  /-
    f : Nat → Complex
    s a : Complex
    hf : LSeriesHasSum f s a
    ⊢ LSeriesHasSum (Neg.neg f) s (Neg.neg a)
  -/
  simpa only [LSeriesHasSum, term_neg] using HasSum.neg hf
  /-
    🎉 no goals
  -/


lemma LSeriesSummable.neg {f : ℕ → ℂ} {s : ℂ} (hf : LSeriesSummable f s) :
    LSeriesSummable (-f) s := by
  /-
    f : Nat → Complex
    s : Complex
    hf : LSeriesSummable f s
    ⊢ LSeriesSummable (Neg.neg f) s
  -/
  simpa only [LSeriesSummable, term_neg] using Summable.neg hf
  /-
    🎉 no goals
  -/


@[simp]
lemma LSeriesSummable.neg_iff {f : ℕ → ℂ} {s : ℂ} :
    LSeriesSummable (-f) s ↔ LSeriesSummable f s :=
  ⟨fun H ↦ neg_neg f ▸ H.neg, .neg⟩


@[simp]
lemma LSeries_neg (f : ℕ → ℂ) (s : ℂ) : LSeries (-f) s = -LSeries f s := by
  /-
    f : Nat → Complex
    s : Complex
    ⊢ Eq (LSeries (Neg.neg f) s) (Neg.neg (LSeries f s))
  -/
  simp only [LSeries, term_neg_apply, tsum_neg]
  /-
    🎉 no goals
  -/


lemma LSeries.term_sub (f g : ℕ → ℂ) (s : ℂ) : term (f - g) s = term f s - term g s := by
  /-
    f g : Nat → Complex
    s : Complex
    ⊢ Eq (LSeries.term (HSub.hSub f g) s) (HSub.hSub (LSeries.term f s) (LSeries.t …
  -/
  simp_rw [sub_eq_add_neg, term_add, term_neg]
  /-
    🎉 no goals
  -/


lemma LSeries.term_sub_apply (f g : ℕ → ℂ) (s : ℂ) (n : ℕ) :
    term (f - g) s n = term f s n - term g s n := by
  /-
    f g : Nat → Complex
    s : Complex
    n : Nat
    ⊢ Eq (LSeries.term (HSub.hSub f g) s n) (HSub.hSub (LSeries.term f s n) (LSeri …
  -/
  rw [term_sub, Pi.sub_apply]
  /-
    🎉 no goals
  -/


lemma LSeriesHasSum.sub {f g : ℕ → ℂ} {s a b : ℂ} (hf : LSeriesHasSum f s a)
    (hg : LSeriesHasSum g s b) :
    LSeriesHasSum (f - g) s (a - b) := by
  /-
    f g : Nat → Complex
    s a b : Complex
    hf : LSeriesHasSum f s a
    hg : LSeriesHasSum g s b
    ⊢ LSeriesHasSum (HSub.hSub f g) s (HSub.hSub a b)
  -/
  simpa only [LSeriesHasSum, term_sub] using HasSum.sub hf hg
  /-
    🎉 no goals
  -/


lemma LSeriesSummable.sub {f g : ℕ → ℂ} {s : ℂ} (hf : LSeriesSummable f s)
    (hg : LSeriesSummable g s) :
    LSeriesSummable (f - g) s := by
  /-
    f g : Nat → Complex
    s : Complex
    hf : LSeriesSummable f s
    hg : LSeriesSummable g s
    ⊢ LSeriesSummable (HSub.hSub f g) s
  -/
  simpa only [LSeriesSummable, ← term_sub_apply] using Summable.sub hf hg
  /-
    🎉 no goals
  -/


@[simp]
lemma LSeries_sub {f g : ℕ → ℂ} {s : ℂ} (hf : LSeriesSummable f s) (hg : LSeriesSummable g s) :
    LSeries (f - g) s = LSeries f s - LSeries g s := by
  /-
    f g : Nat → Complex
    s : Complex
    hf : LSeriesSummable f s
    hg : LSeriesSummable g s
    ⊢ Eq (LSeries (HSub.hSub f g) s) (HSub.hSub (LSeries f s) (LSeries g s))
  -/
  simpa only [LSeries, term_sub, Pi.sub_apply] using tsum_sub hf hg
  /-
    🎉 no goals
  -/


lemma LSeries.term_smul (f : ℕ → ℂ) (c s : ℂ) : term (c • f) s = c • term f s := by
  /-
    f : Nat → Complex
    c s : Complex
    ⊢ Eq (LSeries.term (HSMul.hSMul c f) s) (HSMul.hSMul c (LSeries.term f s))
  -/
  ext ⟨- | n⟩
    /-
      case h.zero
      f : Nat → Complex
      c s : Complex
      ⊢ Eq (LSeries.term (HSMul.hSMul c f) s 0) (HSMul.hSMul c (LSeries.term f s) 0)
    -/
  · simp only [term_zero, Pi.smul_apply, smul_eq_mul, mul_zero]
    /-
      🎉 no goals
    -/
  · simp only [term_of_ne_zero (Nat.succ_ne_zero _), Pi.smul_apply, smul_eq_mul, Nat.cast_succ,
      mul_div_assoc]


lemma LSeries.term_smul_apply (f : ℕ → ℂ) (c s : ℂ) (n : ℕ) :
    term (c • f) s n = c * term f s n := by
  /-
    f : Nat → Complex
    c s : Complex
    n : Nat
    ⊢ Eq (LSeries.term (HSMul.hSMul c f) s n) (HMul.hMul c (LSeries.term f s n))
  -/
  rw [term_smul, Pi.smul_apply, smul_eq_mul]
  /-
    🎉 no goals
  -/


lemma LSeriesHasSum.smul {f : ℕ → ℂ} (c : ℂ) {s a : ℂ} (hf : LSeriesHasSum f s a) :
    LSeriesHasSum (c • f) s (c * a) := by
  /-
    f : Nat → Complex
    c s a : Complex
    hf : LSeriesHasSum f s a
    ⊢ LSeriesHasSum (HSMul.hSMul c f) s (HMul.hMul c a)
  -/
  simpa only [LSeriesHasSum, term_smul, smul_eq_mul] using hf.const_smul c
  /-
    🎉 no goals
  -/


lemma LSeriesSummable.smul {f : ℕ → ℂ} (c : ℂ) {s : ℂ} (hf : LSeriesSummable f s) :
    LSeriesSummable (c • f) s := by
  /-
    f : Nat → Complex
    c s : Complex
    hf : LSeriesSummable f s
    ⊢ LSeriesSummable (HSMul.hSMul c f) s
  -/
  simpa only [LSeriesSummable, term_smul, smul_eq_mul] using hf.const_smul c
  /-
    🎉 no goals
  -/


lemma LSeriesSummable.of_smul {f : ℕ → ℂ} {c s : ℂ} (hc : c ≠ 0) (hf : LSeriesSummable (c • f) s) :
    LSeriesSummable f s := by
  /-
    f : Nat → Complex
    c s : Complex
    hc : Ne c 0
    hf : LSeriesSummable (HSMul.hSMul c f) s
    ⊢ LSeriesSummable f s
  -/
  simpa only [ne_eq, hc, not_false_eq_true, inv_smul_smul₀] using hf.smul (c⁻¹)
  /-
    🎉 no goals
  -/


lemma LSeriesSummable.smul_iff {f : ℕ → ℂ} {c s : ℂ} (hc : c ≠ 0) :
    LSeriesSummable (c • f) s ↔ LSeriesSummable f s :=
  ⟨of_smul hc, smul c⟩


@[simp]
lemma LSeries_smul (f : ℕ → ℂ) (c s : ℂ) : LSeries (c • f) s = c * LSeries f s := by
  /-
    f : Nat → Complex
    c s : Complex
    ⊢ Eq (LSeries (HSMul.hSMul c f) s) (HMul.hMul c (LSeries f s))
  -/
  simp only [LSeries, term_smul_apply, tsum_mul_left]
  /-
    🎉 no goals
  -/


@[simp]
lemma LSeries.term_sum_apply (n : ℕ) :
    term (∑ i ∈ S, f i) s n  = ∑ i ∈ S, term (f i) s n := by
  /-
    ι : Type u_1
    f : ι → Nat → Complex
    S : Finset ι
    s : Complex
    n : Nat
    ⊢ Eq (LSeries.term (S.sum fun i => f i) s n) (S.sum fun i => LSeries.term (f i …
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      ι : Type u_1
      f : ι → Nat → Complex
      S : Finset ι
      s : Complex
      ⊢ Eq (LSeries.term (S.sum fun i => f i) s 0) (S.sum fun i => LSeries.term (f i …
    -/
  · simp only [term_zero, Finset.sum_const_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      f : ι → Nat → Complex
      S : Finset ι
      s : Complex
      n : Nat
      hn : Ne n 0
      ⊢ Eq (LSeries.term (S.sum fun i => f i) s n) (S.sum fun i => LSeries.term (f i …
    -/
  · simp only [ne_eq, hn, not_false_eq_true, term_of_ne_zero, Finset.sum_apply, Finset.sum_div]
    /-
      🎉 no goals
    -/


lemma LSeries.term_sum : term (∑ i ∈ S, f i) s  = ∑ i ∈ S, term (f i) s :=
                    /-
                      ι : Type u_1
                      f : ι → Nat → Complex
                      S : Finset ι
                      s : Complex
                      x✝ : Nat
                      ⊢ Eq (LSeries.term (S.sum fun i => f i) s x✝) (S.sum (fun i => LSeries.term (f …
                    -/
  funext fun _ ↦ by rw [Finset.sum_apply]; exact term_sum_apply f S s _
                                           /-
                                             🎉 no goals
                                           -/


lemma LSeriesHasSum.sum {a : ι → ℂ} (hf : ∀ i ∈ S, LSeriesHasSum (f i) s (a i)) :
    LSeriesHasSum (∑ i ∈ S, f i) s (∑ i ∈ S, a i) := by
  /-
    ι : Type u_1
    f : ι → Nat → Complex
    S : Finset ι
    s : Complex
    a : ι → Complex
    hf : ∀ (i : ι), Membership.mem S i → LSeriesHasSum (f i) s (a i)
    ⊢ LSeriesHasSum (S.sum fun i => f i) s (S.sum fun i => a i)
  -/
  simpa only [LSeriesHasSum, term_sum, Finset.sum_fn S fun i ↦ term (f i) s] using hasSum_sum hf
  /-
    🎉 no goals
  -/


lemma LSeriesSummable.sum (hf : ∀ i ∈ S, LSeriesSummable (f i) s) :
    LSeriesSummable (∑ i ∈ S, f i) s := by
  /-
    ι : Type u_1
    f : ι → Nat → Complex
    S : Finset ι
    s : Complex
    hf : ∀ (i : ι), Membership.mem S i → LSeriesSummable (f i) s
    ⊢ LSeriesSummable (S.sum fun i => f i) s
  -/
  simpa only [LSeriesSummable, ← term_sum_apply] using summable_sum hf
  /-
    🎉 no goals
  -/


@[simp]
lemma LSeries_sum (hf : ∀ i ∈ S, LSeriesSummable (f i) s) :
    LSeries (∑ i ∈ S, f i) s = ∑ i ∈ S, LSeries (f i) s := by
  /-
    ι : Type u_1
    f : ι → Nat → Complex
    S : Finset ι
    s : Complex
    hf : ∀ (i : ι), Membership.mem S i → LSeriesSummable (f i) s
    ⊢ Eq (LSeries (S.sum fun i => f i) s) (S.sum fun i => LSeries (f i) s)
  -/
  simpa only [LSeries, term_sum, Finset.sum_apply] using tsum_sum hf
  /-
    🎉 no goals
  -/


