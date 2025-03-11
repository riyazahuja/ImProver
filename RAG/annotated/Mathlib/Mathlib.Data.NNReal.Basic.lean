noncomputable instance : FloorSemiring ℝ≥0 := Nonneg.floorSemiring


@[simp, norm_cast]
theorem coe_indicator {α} (s : Set α) (f : α → ℝ≥0) (a : α) :
    ((s.indicator f a : ℝ≥0) : ℝ) = s.indicator (fun x => ↑(f x)) a :=
  (toRealHom : ℝ≥0 →+ ℝ).map_indicator _ _ _


@[norm_cast]
theorem coe_list_sum (l : List ℝ≥0) : ((l.sum : ℝ≥0) : ℝ) = (l.map (↑)).sum :=
  map_list_sum toRealHom l


@[norm_cast]
theorem coe_list_prod (l : List ℝ≥0) : ((l.prod : ℝ≥0) : ℝ) = (l.map (↑)).prod :=
  map_list_prod toRealHom l


@[norm_cast]
theorem coe_multiset_sum (s : Multiset ℝ≥0) : ((s.sum : ℝ≥0) : ℝ) = (s.map (↑)).sum :=
  map_multiset_sum toRealHom s


@[norm_cast]
theorem coe_multiset_prod (s : Multiset ℝ≥0) : ((s.prod : ℝ≥0) : ℝ) = (s.map (↑)).prod :=
  map_multiset_prod toRealHom s


@[simp, norm_cast]
theorem coe_sum (s : Finset ι) (f : ι → ℝ≥0) : ∑ i ∈ s, f i = ∑ i ∈ s, (f i : ℝ) :=
  map_sum toRealHom _ _


@[simp, norm_cast]
lemma coe_expect (s : Finset ι) (f : ι → ℝ≥0) : 𝔼 i ∈ s, f i = 𝔼 i ∈ s, (f i : ℝ) :=
  map_expect toRealHom ..


theorem _root_.Real.toNNReal_sum_of_nonneg (hf : ∀ i ∈ s, 0 ≤ f i) :
    Real.toNNReal (∑ a ∈ s, f a) = ∑ a ∈ s, Real.toNNReal (f a) := by
  /-
    ι : Type u_1
    s : Finset ι
    f : ι → Real
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    ⊢ Eq (s.sum fun a => f a).toNNReal (s.sum fun a => (f a).toNNReal)
  -/
  rw [← coe_inj, NNReal.coe_sum, Real.coe_toNNReal _ (Finset.sum_nonneg hf)]
  /-
    ι : Type u_1
    s : Finset ι
    f : ι → Real
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    ⊢ Eq (s.sum fun i => f i) (s.sum fun i => ↑(f i).toNNReal)
  -/
  exact Finset.sum_congr rfl fun x hxs => by rw [Real.coe_toNNReal _ (hf x hxs)]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_prod (s : Finset ι) (f : ι → ℝ≥0) : ↑(∏ a ∈ s, f a) = ∏ a ∈ s, (f a : ℝ) :=
  map_prod toRealHom _ _


theorem _root_.Real.toNNReal_prod_of_nonneg (hf : ∀ a, a ∈ s → 0 ≤ f a) :
    Real.toNNReal (∏ a ∈ s, f a) = ∏ a ∈ s, Real.toNNReal (f a) := by
  /-
    ι : Type u_1
    s : Finset ι
    f : ι → Real
    hf : ∀ (a : ι), Membership.mem s a → LE.le 0 (f a)
    ⊢ Eq (s.prod fun a => f a).toNNReal (s.prod fun a => (f a).toNNReal)
  -/
  rw [← coe_inj, NNReal.coe_prod, Real.coe_toNNReal _ (Finset.prod_nonneg hf)]
  /-
    ι : Type u_1
    s : Finset ι
    f : ι → Real
    hf : ∀ (a : ι), Membership.mem s a → LE.le 0 (f a)
    ⊢ Eq (s.prod fun i => f i) (s.prod fun a => ↑(f a).toNNReal)
  -/
  exact Finset.prod_congr rfl fun x hxs => by rw [Real.coe_toNNReal _ (hf x hxs)]
  /-
    🎉 no goals
  -/


theorem le_iInf_add_iInf {ι ι' : Sort*} [Nonempty ι] [Nonempty ι'] {f : ι → ℝ≥0} {g : ι' → ℝ≥0}
    {a : ℝ≥0} (h : ∀ i j, a ≤ f i + g j) : a ≤ (⨅ i, f i) + ⨅ j, g j := by
  /-
    ι : Sort u_2
    ι' : Sort u_3
    inst✝¹ : Nonempty ι
    inst✝ : Nonempty ι'
    f : ι → NNReal
    g : ι' → NNReal
    a : NNReal
    h : ∀ (i : ι) (j : ι'), LE.le a (HAdd.hAdd (f i) (g j))
    ⊢ LE.le a (HAdd.hAdd (iInf fun i => f i) (iInf fun j => g j))
  -/
  rw [← NNReal.coe_le_coe, NNReal.coe_add, coe_iInf, coe_iInf]
  /-
    ι : Sort u_2
    ι' : Sort u_3
    inst✝¹ : Nonempty ι
    inst✝ : Nonempty ι'
    f : ι → NNReal
    g : ι' → NNReal
    a : NNReal
    h : ∀ (i : ι) (j : ι'), LE.le a (HAdd.hAdd (f i) (g j))
    ⊢ LE.le (↑a) (HAdd.hAdd (iInf fun i => ↑(f i)) (iInf fun i => ↑(g i)))
  -/
  exact le_ciInf_add_ciInf h
  /-
    🎉 no goals
  -/


theorem mul_finset_sup {α} (r : ℝ≥0) (s : Finset α) (f : α → ℝ≥0) :
    r * s.sup f = s.sup fun a => r * f a :=
  Finset.comp_sup_eq_sup_comp _ (NNReal.mul_sup r) (mul_zero r)


theorem finset_sup_mul {α} (s : Finset α) (f : α → ℝ≥0) (r : ℝ≥0) :
    s.sup f * r = s.sup fun a => f a * r :=
  Finset.comp_sup_eq_sup_comp (· * r) (fun x y => NNReal.sup_mul x y r) (zero_mul r)


theorem finset_sup_div {α} {f : α → ℝ≥0} {s : Finset α} (r : ℝ≥0) :
                                               /-
                                                 α : Type u_2
                                                 f : α → NNReal
                                                 s : Finset α
                                                 r : NNReal
                                                 ⊢ Eq (HDiv.hDiv (s.sup f) r) (s.sup fun a => HDiv.hDiv (f a) r)
                                               -/
    s.sup f / r = s.sup fun a => f a / r := by simp only [div_eq_inv_mul, mul_finset_sup]
                                               /-
                                                 🎉 no goals
                                               -/


theorem sub_div (a b c : ℝ≥0) : (a - b) / c = a / c - b / c :=
  tsub_div _ _ _


theorem iInf_mul (f : ι → ℝ≥0) (a : ℝ≥0) : iInf f * a = ⨅ i, f i * a := by
  /-
    ι : Sort u_2
    f : ι → NNReal
    a : NNReal
    ⊢ Eq (HMul.hMul (iInf f) a) (iInf fun i => HMul.hMul (f i) a)
  -/
  rw [← coe_inj, NNReal.coe_mul, coe_iInf, coe_iInf]
  /-
    ι : Sort u_2
    f : ι → NNReal
    a : NNReal
    ⊢ Eq (HMul.hMul (iInf fun i => ↑(f i)) ↑a) (iInf fun i => ↑(HMul.hMul (f i) a))
  -/
  exact Real.iInf_mul_of_nonneg (NNReal.coe_nonneg _) _
  /-
    🎉 no goals
  -/


theorem mul_iInf (f : ι → ℝ≥0) (a : ℝ≥0) : a * iInf f = ⨅ i, a * f i := by
  /-
    ι : Sort u_2
    f : ι → NNReal
    a : NNReal
    ⊢ Eq (HMul.hMul a (iInf f)) (iInf fun i => HMul.hMul a (f i))
  -/
  simpa only [mul_comm] using iInf_mul f a
  /-
    🎉 no goals
  -/


theorem mul_iSup (f : ι → ℝ≥0) (a : ℝ≥0) : (a * ⨆ i, f i) = ⨆ i, a * f i := by
  /-
    ι : Sort u_2
    f : ι → NNReal
    a : NNReal
    ⊢ Eq (HMul.hMul a (iSup fun i => f i)) (iSup fun i => HMul.hMul a (f i))
  -/
  rw [← coe_inj, NNReal.coe_mul, NNReal.coe_iSup, NNReal.coe_iSup]
  /-
    ι : Sort u_2
    f : ι → NNReal
    a : NNReal
    ⊢ Eq (HMul.hMul (↑a) (iSup fun i => ↑(f i))) (iSup fun i => ↑(HMul.hMul a (f i …
  -/
  exact Real.mul_iSup_of_nonneg (NNReal.coe_nonneg _) _
  /-
    🎉 no goals
  -/


theorem iSup_mul (f : ι → ℝ≥0) (a : ℝ≥0) : (⨆ i, f i) * a = ⨆ i, f i * a := by
  /-
    ι : Sort u_2
    f : ι → NNReal
    a : NNReal
    ⊢ Eq (HMul.hMul (iSup fun i => f i) a) (iSup fun i => HMul.hMul (f i) a)
  -/
  rw [mul_comm, mul_iSup]
  /-
    ι : Sort u_2
    f : ι → NNReal
    a : NNReal
    ⊢ Eq (iSup fun i => HMul.hMul a (f i)) (iSup fun i => HMul.hMul (f i) a)
  -/
  simp_rw [mul_comm]
  /-
    🎉 no goals
  -/


theorem iSup_div (f : ι → ℝ≥0) (a : ℝ≥0) : (⨆ i, f i) / a = ⨆ i, f i / a := by
  /-
    ι : Sort u_2
    f : ι → NNReal
    a : NNReal
    ⊢ Eq (HDiv.hDiv (iSup fun i => f i) a) (iSup fun i => HDiv.hDiv (f i) a)
  -/
  simp only [div_eq_mul_inv, iSup_mul]
  /-
    🎉 no goals
  -/

-- Porting note: generalized to allow empty `ι`

theorem mul_iSup_le {a : ℝ≥0} {g : ℝ≥0} {h : ι → ℝ≥0} (H : ∀ j, g * h j ≤ a) : g * iSup h ≤ a := by
  /-
    ι : Sort u_2
    a g : NNReal
    h : ι → NNReal
    H : ∀ (j : ι), LE.le (HMul.hMul g (h j)) a
    ⊢ LE.le (HMul.hMul g (iSup h)) a
  -/
  rw [mul_iSup]
  /-
    ι : Sort u_2
    a g : NNReal
    h : ι → NNReal
    H : ∀ (j : ι), LE.le (HMul.hMul g (h j)) a
    ⊢ LE.le (iSup fun i => HMul.hMul g (h i)) a
  -/
  exact ciSup_le' H
  /-
    🎉 no goals
  -/

-- Porting note: generalized to allow empty `ι`

theorem iSup_mul_le {a : ℝ≥0} {g : ι → ℝ≥0} {h : ℝ≥0} (H : ∀ i, g i * h ≤ a) : iSup g * h ≤ a := by
  /-
    ι : Sort u_2
    a : NNReal
    g : ι → NNReal
    h : NNReal
    H : ∀ (i : ι), LE.le (HMul.hMul (g i) h) a
    ⊢ LE.le (HMul.hMul (iSup g) h) a
  -/
  rw [iSup_mul]
  /-
    ι : Sort u_2
    a : NNReal
    g : ι → NNReal
    h : NNReal
    H : ∀ (i : ι), LE.le (HMul.hMul (g i) h) a
    ⊢ LE.le (iSup fun i => HMul.hMul (g i) h) a
  -/
  exact ciSup_le' H
  /-
    🎉 no goals
  -/

-- Porting note: generalized to allow empty `ι`

theorem iSup_mul_iSup_le {a : ℝ≥0} {g h : ι → ℝ≥0} (H : ∀ i j, g i * h j ≤ a) :
    iSup g * iSup h ≤ a :=
  iSup_mul_le fun _ => mul_iSup_le <| H _


theorem le_mul_iInf {a : ℝ≥0} {g : ℝ≥0} {h : ι → ℝ≥0} (H : ∀ j, a ≤ g * h j) : a ≤ g * iInf h := by
  /-
    ι : Sort u_2
    inst✝ : Nonempty ι
    a g : NNReal
    h : ι → NNReal
    H : ∀ (j : ι), LE.le a (HMul.hMul g (h j))
    ⊢ LE.le a (HMul.hMul g (iInf h))
  -/
  rw [mul_iInf]
  /-
    ι : Sort u_2
    inst✝ : Nonempty ι
    a g : NNReal
    h : ι → NNReal
    H : ∀ (j : ι), LE.le a (HMul.hMul g (h j))
    ⊢ LE.le a (iInf fun i => HMul.hMul g (h i))
  -/
  exact le_ciInf H
  /-
    🎉 no goals
  -/


theorem le_iInf_mul {a : ℝ≥0} {g : ι → ℝ≥0} {h : ℝ≥0} (H : ∀ i, a ≤ g i * h) : a ≤ iInf g * h := by
  /-
    ι : Sort u_2
    inst✝ : Nonempty ι
    a : NNReal
    g : ι → NNReal
    h : NNReal
    H : ∀ (i : ι), LE.le a (HMul.hMul (g i) h)
    ⊢ LE.le a (HMul.hMul (iInf g) h)
  -/
  rw [iInf_mul]
  /-
    ι : Sort u_2
    inst✝ : Nonempty ι
    a : NNReal
    g : ι → NNReal
    h : NNReal
    H : ∀ (i : ι), LE.le a (HMul.hMul (g i) h)
    ⊢ LE.le a (iInf fun i => HMul.hMul (g i) h)
  -/
  exact le_ciInf H
  /-
    🎉 no goals
  -/


theorem le_iInf_mul_iInf {a : ℝ≥0} {g h : ι → ℝ≥0} (H : ∀ i j, a ≤ g i * h j) :
    a ≤ iInf g * iInf h :=
  le_iInf_mul fun i => le_mul_iInf <| H i


