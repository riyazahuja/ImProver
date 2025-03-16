@[simp]
theorem Nat.comap_cast_atTop [StrictOrderedSemiring R] [Archimedean R] :
    comap ((↑) : ℕ → R) atTop = atTop :=
  comap_embedding_atTop (fun _ _ => Nat.cast_le) exists_nat_ge


theorem tendsto_natCast_atTop_iff [StrictOrderedSemiring R] [Archimedean R] {f : α → ℕ}
    {l : Filter α} : Tendsto (fun n => (f n : R)) l atTop ↔ Tendsto f l atTop :=
  tendsto_atTop_embedding (fun _ _ => Nat.cast_le) exists_nat_ge


@[deprecated (since := "2024-04-17")]
alias tendsto_nat_cast_atTop_iff := tendsto_natCast_atTop_iff


theorem tendsto_natCast_atTop_atTop [OrderedSemiring R] [Archimedean R] :
    Tendsto ((↑) : ℕ → R) atTop atTop :=
  Nat.mono_cast.tendsto_atTop_atTop exists_nat_ge


@[deprecated (since := "2024-04-17")]
alias tendsto_nat_cast_atTop_atTop := tendsto_natCast_atTop_atTop


theorem Filter.Eventually.natCast_atTop [OrderedSemiring R] [Archimedean R] {p : R → Prop}
    (h : ∀ᶠ (x : R) in atTop, p x) : ∀ᶠ (n : ℕ) in atTop, p n :=
  tendsto_natCast_atTop_atTop.eventually h


@[deprecated (since := "2024-04-17")]
alias Filter.Eventually.nat_cast_atTop := Filter.Eventually.natCast_atTop


@[simp] theorem Int.comap_cast_atTop [StrictOrderedRing R] [Archimedean R] :
    comap ((↑) : ℤ → R) atTop = atTop :=
  comap_embedding_atTop (fun _ _ => Int.cast_le) fun r =>
    let ⟨n, hn⟩ := exists_nat_ge r; ⟨n, mod_cast hn⟩


@[simp]
theorem Int.comap_cast_atBot [StrictOrderedRing R] [Archimedean R] :
    comap ((↑) : ℤ → R) atBot = atBot :=
  comap_embedding_atBot (fun _ _ => Int.cast_le) fun r =>
    let ⟨n, hn⟩ := exists_nat_ge (-r)
            /-
              R : Type u_2
              inst✝¹ : StrictOrderedRing R
              inst✝ : Archimedean R
              r : R
              n : Nat
              hn : LE.le (Neg.neg r) ↑n
              ⊢ LE.le (↑(Neg.neg ↑n)) r
            -/
    ⟨-n, by simpa [neg_le] using hn⟩
            /-
              🎉 no goals
            -/


theorem tendsto_intCast_atTop_iff [StrictOrderedRing R] [Archimedean R] {f : α → ℤ}
    {l : Filter α} : Tendsto (fun n => (f n : R)) l atTop ↔ Tendsto f l atTop := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : StrictOrderedRing R
    inst✝ : Archimedean R
    f : α → Int
    l : Filter α
    ⊢ Iff (Filter.Tendsto (fun n => ↑(f n)) l Filter.atTop) (Filter.Tendsto f l Fi …
  -/
  rw [← @Int.comap_cast_atTop R, tendsto_comap_iff]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


@[deprecated (since := "2024-04-17")]
alias tendsto_int_cast_atTop_iff := tendsto_intCast_atTop_iff


theorem tendsto_intCast_atBot_iff [StrictOrderedRing R] [Archimedean R] {f : α → ℤ}
    {l : Filter α} : Tendsto (fun n => (f n : R)) l atBot ↔ Tendsto f l atBot := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : StrictOrderedRing R
    inst✝ : Archimedean R
    f : α → Int
    l : Filter α
    ⊢ Iff (Filter.Tendsto (fun n => ↑(f n)) l Filter.atBot) (Filter.Tendsto f l Fi …
  -/
  rw [← @Int.comap_cast_atBot R, tendsto_comap_iff]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


@[deprecated (since := "2024-04-17")]
alias tendsto_int_cast_atBot_iff := tendsto_intCast_atBot_iff


theorem tendsto_intCast_atTop_atTop [StrictOrderedRing R] [Archimedean R] :
    Tendsto ((↑) : ℤ → R) atTop atTop :=
  tendsto_intCast_atTop_iff.2 tendsto_id


@[deprecated (since := "2024-04-17")]
alias tendsto_int_cast_atTop_atTop := tendsto_intCast_atTop_atTop


theorem Filter.Eventually.intCast_atTop [StrictOrderedRing R] [Archimedean R] {p : R → Prop}
    (h : ∀ᶠ (x : R) in atTop, p x) : ∀ᶠ (n : ℤ) in atTop, p n := by
  /-
    R : Type u_2
    inst✝¹ : StrictOrderedRing R
    inst✝ : Archimedean R
    p : R → Prop
    h : Filter.Eventually (fun x => p x) Filter.atTop
    ⊢ Filter.Eventually (fun n => p ↑n) Filter.atTop
  -/
  rw [← Int.comap_cast_atTop (R := R)]; exact h.comap _
                                        /-
                                          🎉 no goals
                                        -/


@[deprecated (since := "2024-04-17")]
alias Filter.Eventually.int_cast_atTop := Filter.Eventually.intCast_atTop


theorem Filter.Eventually.intCast_atBot [StrictOrderedRing R] [Archimedean R] {p : R → Prop}
    (h : ∀ᶠ (x : R) in atBot, p x) : ∀ᶠ (n : ℤ) in atBot, p n := by
  /-
    R : Type u_2
    inst✝¹ : StrictOrderedRing R
    inst✝ : Archimedean R
    p : R → Prop
    h : Filter.Eventually (fun x => p x) Filter.atBot
    ⊢ Filter.Eventually (fun n => p ↑n) Filter.atBot
  -/
  rw [← Int.comap_cast_atBot (R := R)]; exact h.comap _
                                        /-
                                          🎉 no goals
                                        -/


@[deprecated (since := "2024-04-17")]
alias Filter.Eventually.int_cast_atBot := Filter.Eventually.intCast_atBot


@[simp]
theorem Rat.comap_cast_atTop [LinearOrderedField R] [Archimedean R] :
    comap ((↑) : ℚ → R) atTop = atTop :=
  comap_embedding_atTop (fun _ _ => Rat.cast_le) fun r =>
                                           /-
                                             R : Type u_2
                                             inst✝¹ : LinearOrderedField R
                                             inst✝ : Archimedean R
                                             r : R
                                             n : Nat
                                             hn : LE.le r ↑n
                                             ⊢ LE.le r ↑↑n
                                           -/
    let ⟨n, hn⟩ := exists_nat_ge r; ⟨n, by simpa⟩
                                           /-
                                             🎉 no goals
                                           -/


@[simp] theorem Rat.comap_cast_atBot [LinearOrderedField R] [Archimedean R] :
    comap ((↑) : ℚ → R) atBot = atBot :=
  comap_embedding_atBot (fun _ _ => Rat.cast_le) fun r =>
    let ⟨n, hn⟩ := exists_nat_ge (-r)
            /-
              R : Type u_2
              inst✝¹ : LinearOrderedField R
              inst✝ : Archimedean R
              r : R
              n : Nat
              hn : LE.le (Neg.neg r) ↑n
              ⊢ LE.le (↑(Neg.neg ↑n)) r
            -/
    ⟨-n, by simpa [neg_le]⟩
            /-
              🎉 no goals
            -/


theorem tendsto_ratCast_atTop_iff [LinearOrderedField R] [Archimedean R] {f : α → ℚ}
    {l : Filter α} : Tendsto (fun n => (f n : R)) l atTop ↔ Tendsto f l atTop := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : LinearOrderedField R
    inst✝ : Archimedean R
    f : α → Rat
    l : Filter α
    ⊢ Iff (Filter.Tendsto (fun n => ↑(f n)) l Filter.atTop) (Filter.Tendsto f l Fi …
  -/
  rw [← @Rat.comap_cast_atTop R, tendsto_comap_iff]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


@[deprecated (since := "2024-04-17")]
alias tendsto_rat_cast_atTop_iff := tendsto_ratCast_atTop_iff


theorem tendsto_ratCast_atBot_iff [LinearOrderedField R] [Archimedean R] {f : α → ℚ}
    {l : Filter α} : Tendsto (fun n => (f n : R)) l atBot ↔ Tendsto f l atBot := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : LinearOrderedField R
    inst✝ : Archimedean R
    f : α → Rat
    l : Filter α
    ⊢ Iff (Filter.Tendsto (fun n => ↑(f n)) l Filter.atBot) (Filter.Tendsto f l Fi …
  -/
  rw [← @Rat.comap_cast_atBot R, tendsto_comap_iff]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


@[deprecated (since := "2024-04-17")]
alias tendsto_rat_cast_atBot_iff := tendsto_ratCast_atBot_iff


theorem Filter.Eventually.ratCast_atTop [LinearOrderedField R] [Archimedean R] {p : R → Prop}
    (h : ∀ᶠ (x : R) in atTop, p x) : ∀ᶠ (n : ℚ) in atTop, p n := by
  /-
    R : Type u_2
    inst✝¹ : LinearOrderedField R
    inst✝ : Archimedean R
    p : R → Prop
    h : Filter.Eventually (fun x => p x) Filter.atTop
    ⊢ Filter.Eventually (fun n => p ↑n) Filter.atTop
  -/
  rw [← Rat.comap_cast_atTop (R := R)]; exact h.comap _
                                        /-
                                          🎉 no goals
                                        -/


@[deprecated (since := "2024-04-17")]
alias Filter.Eventually.rat_cast_atTop := Filter.Eventually.ratCast_atTop


theorem Filter.Eventually.ratCast_atBot [LinearOrderedField R] [Archimedean R] {p : R → Prop}
    (h : ∀ᶠ (x : R) in atBot, p x) : ∀ᶠ (n : ℚ) in atBot, p n := by
  /-
    R : Type u_2
    inst✝¹ : LinearOrderedField R
    inst✝ : Archimedean R
    p : R → Prop
    h : Filter.Eventually (fun x => p x) Filter.atBot
    ⊢ Filter.Eventually (fun n => p ↑n) Filter.atBot
  -/
  rw [← Rat.comap_cast_atBot (R := R)]; exact h.comap _
                                        /-
                                          🎉 no goals
                                        -/


@[deprecated (since := "2024-04-17")]
alias Filter.Eventually.rat_cast_atBot := Filter.Eventually.ratCast_atBot


theorem atTop_hasAntitoneBasis_of_archimedean [OrderedSemiring R] [Archimedean R] :
    (atTop : Filter R).HasAntitoneBasis fun n : ℕ => Ici n :=
  hasAntitoneBasis_atTop.comp_mono Nat.mono_cast tendsto_natCast_atTop_atTop


theorem atTop_hasCountableBasis_of_archimedean [OrderedSemiring R] [Archimedean R] :
    (atTop : Filter R).HasCountableBasis (fun _ : ℕ => True) fun n => Ici n :=
  ⟨atTop_hasAntitoneBasis_of_archimedean.1, to_countable _⟩


theorem atBot_hasCountableBasis_of_archimedean [OrderedRing R] [Archimedean R] :
    (atBot : Filter R).HasCountableBasis (fun _ : ℤ => True) fun m => Iic m where
  countable := to_countable _
  toHasBasis :=
    atBot_basis.to_hasBasis
      (fun x _ => let ⟨m, hm⟩ := exists_int_le x; ⟨m, trivial, Iic_subset_Iic.2 hm⟩)
      fun m _ => ⟨m, trivial, Subset.rfl⟩


instance (priority := 100) atTop_isCountablyGenerated_of_archimedean [OrderedSemiring R]
    [Archimedean R] : (atTop : Filter R).IsCountablyGenerated :=
  atTop_hasCountableBasis_of_archimedean.isCountablyGenerated


instance (priority := 100) atBot_isCountablyGenerated_of_archimedean [OrderedRing R]
    [Archimedean R] : (atBot : Filter R).IsCountablyGenerated :=
  atBot_hasCountableBasis_of_archimedean.isCountablyGenerated


/-- If a function tends to infinity along a filter, then this function multiplied by a positive
constant (on the left) also tends to infinity. The archimedean assumption is convenient to get a
statement that works on `ℕ`, `ℤ` and `ℝ`, although not necessary (a version in ordered fields is
given in `Filter.Tendsto.const_mul_atTop`). -/
theorem Tendsto.const_mul_atTop' (hr : 0 < r) (hf : Tendsto f l atTop) :
    Tendsto (fun x => r * f x) l atTop := by
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedSemiring R
    inst✝ : Archimedean R
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atTop
  -/
  refine tendsto_atTop.2 fun b => ?_
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedSemiring R
    inst✝ : Archimedean R
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    b : R
    ⊢ Filter.Eventually (fun a => LE.le b (HMul.hMul r (f a))) l
  -/
  obtain ⟨n : ℕ, hn : 1 ≤ n • r⟩ := Archimedean.arch 1 hr
  /-
    case intro
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedSemiring R
    inst✝ : Archimedean R
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    b : R
    n : Nat
    hn : LE.le 1 (HSMul.hSMul n r)
    ⊢ Filter.Eventually (fun a => LE.le b (HMul.hMul r (f a))) l
  -/
  rw [nsmul_eq_mul'] at hn
  /-
    case intro
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedSemiring R
    inst✝ : Archimedean R
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    b : R
    n : Nat
    hn : LE.le 1 (HMul.hMul r ↑n)
    ⊢ Filter.Eventually (fun a => LE.le b (HMul.hMul r (f a))) l
  -/
  filter_upwards [tendsto_atTop.1 hf (n * max b 0)] with x hx
  calc
    b ≤ 1 * max b 0 := by
    { rw [one_mul]
      exact le_max_left _ _ }
    _ ≤ r * n * max b 0 := by gcongr
    _ = r * (n * max b 0) := by rw [mul_assoc]
    _ ≤ r * f x := by gcongr


/-- If a function tends to infinity along a filter, then this function multiplied by a positive
constant (on the right) also tends to infinity. The archimedean assumption is convenient to get a
statement that works on `ℕ`, `ℤ` and `ℝ`, although not necessary (a version in ordered fields is
given in `Filter.Tendsto.atTop_mul_const`). -/
theorem Tendsto.atTop_mul_const' (hr : 0 < r) (hf : Tendsto f l atTop) :
    Tendsto (fun x => f x * r) l atTop := by
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedSemiring R
    inst✝ : Archimedean R
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atTop
  -/
  refine tendsto_atTop.2 fun b => ?_
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedSemiring R
    inst✝ : Archimedean R
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    b : R
    ⊢ Filter.Eventually (fun a => LE.le b (HMul.hMul (f a) r)) l
  -/
  obtain ⟨n : ℕ, hn : 1 ≤ n • r⟩ := Archimedean.arch 1 hr
  /-
    case intro
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedSemiring R
    inst✝ : Archimedean R
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    b : R
    n : Nat
    hn : LE.le 1 (HSMul.hSMul n r)
    ⊢ Filter.Eventually (fun a => LE.le b (HMul.hMul (f a) r)) l
  -/
  have hn' : 1 ≤ (n : R) * r := by rwa [nsmul_eq_mul] at hn
  /-
    case intro
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedSemiring R
    inst✝ : Archimedean R
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    b : R
    n : Nat
    hn : LE.le 1 (HSMul.hSMul n r)
    hn' : LE.le 1 (HMul.hMul (↑n) r)
    ⊢ Filter.Eventually (fun a => LE.le b (HMul.hMul (f a) r)) l
  -/
  filter_upwards [tendsto_atTop.1 hf (max b 0 * n)] with x hx
  calc
    b ≤ max b 0 * 1 := by
    { rw [mul_one]
      exact le_max_left _ _ }
    _ ≤ max b 0 * (n * r) := by gcongr
    _ = max b 0 * n * r := by rw [mul_assoc]
    _ ≤ f x * r := by gcongr


/-- See also `Filter.Tendsto.atTop_mul_const_of_neg` for a version of this lemma for
`LinearOrderedField`s which does not require the `Archimedean` assumption. -/
theorem Tendsto.atTop_mul_const_of_neg' (hr : r < 0) (hf : Tendsto f l atTop) :
    Tendsto (fun x => f x * r) l atBot := by
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedRing R
    inst✝ : Archimedean R
    hr : LT.lt r 0
    hf : Filter.Tendsto f l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atBot
  -/
  simpa only [tendsto_neg_atTop_iff, mul_neg] using hf.atTop_mul_const' (neg_pos.mpr hr)
  /-
    🎉 no goals
  -/


/-- See also `Filter.Tendsto.atBot_mul_const` for a version of this lemma for
`LinearOrderedField`s which does not require the `Archimedean` assumption. -/
theorem Tendsto.atBot_mul_const' (hr : 0 < r) (hf : Tendsto f l atBot) :
    Tendsto (fun x => f x * r) l atBot := by
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedRing R
    inst✝ : Archimedean R
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atBot
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atBot
  -/
  simp only [← tendsto_neg_atTop_iff, ← neg_mul] at hf ⊢
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedRing R
    inst✝ : Archimedean R
    hr : LT.lt 0 r
    hf : Filter.Tendsto (fun x => Neg.neg (f x)) l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Neg.neg (f x)) r) l Filter.atTop
  -/
  exact hf.atTop_mul_const' hr
  /-
    🎉 no goals
  -/


/-- See also `Filter.Tendsto.atBot_mul_const_of_neg` for a version of this lemma for
`LinearOrderedField`s which does not require the `Archimedean` assumption. -/
theorem Tendsto.atBot_mul_const_of_neg' (hr : r < 0) (hf : Tendsto f l atBot) :
    Tendsto (fun x => f x * r) l atTop := by
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    f : α → R
    r : R
    inst✝¹ : LinearOrderedRing R
    inst✝ : Archimedean R
    hr : LT.lt r 0
    hf : Filter.Tendsto f l Filter.atBot
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atTop
  -/
  simpa only [mul_neg, tendsto_neg_atBot_iff] using hf.atBot_mul_const' (neg_pos.2 hr)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-05-06")]
alias Tendsto.atTop_mul_neg_const' := Tendsto.atTop_mul_const_of_neg'


@[deprecated (since := "2024-05-06")]
alias Tendsto.atBot_mul_neg_const' := Tendsto.atBot_mul_const_of_neg'


theorem Tendsto.atTop_nsmul_const {f : α → ℕ} (hr : 0 < r) (hf : Tendsto f l atTop) :
    Tendsto (fun x => f x • r) l atTop := by
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    r : R
    inst✝¹ : LinearOrderedCancelAddCommMonoid R
    inst✝ : Archimedean R
    f : α → Nat
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HSMul.hSMul (f x) r) l Filter.atTop
  -/
  refine tendsto_atTop.mpr fun s => ?_
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    r : R
    inst✝¹ : LinearOrderedCancelAddCommMonoid R
    inst✝ : Archimedean R
    f : α → Nat
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    s : R
    ⊢ Filter.Eventually (fun a => LE.le s (HSMul.hSMul (f a) r)) l
  -/
  obtain ⟨n : ℕ, hn : s ≤ n • r⟩ := Archimedean.arch s hr
  /-
    case intro
    α : Type u_1
    R : Type u_2
    l : Filter α
    r : R
    inst✝¹ : LinearOrderedCancelAddCommMonoid R
    inst✝ : Archimedean R
    f : α → Nat
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    s : R
    n : Nat
    hn : LE.le s (HSMul.hSMul n r)
    ⊢ Filter.Eventually (fun a => LE.le s (HSMul.hSMul (f a) r)) l
  -/
  exact (tendsto_atTop.mp hf n).mono fun a ha => hn.trans (nsmul_le_nsmul_left hr.le ha)
  /-
    🎉 no goals
  -/


theorem Tendsto.atTop_nsmul_neg_const {f : α → ℕ} (hr : r < 0) (hf : Tendsto f l atTop) :
                                             /-
                                               α : Type u_1
                                               R : Type u_2
                                               l : Filter α
                                               r : R
                                               inst✝¹ : LinearOrderedAddCommGroup R
                                               inst✝ : Archimedean R
                                               f : α → Nat
                                               hr : LT.lt r 0
                                               hf : Filter.Tendsto f l Filter.atTop
                                               ⊢ Filter.Tendsto (fun x => HSMul.hSMul (f x) r) l Filter.atBot
                                             -/
    Tendsto (fun x => f x • r) l atBot := by simpa using hf.atTop_nsmul_const (neg_pos.2 hr)
                                             /-
                                               🎉 no goals
                                             -/


theorem Tendsto.atTop_zsmul_const {f : α → ℤ} (hr : 0 < r) (hf : Tendsto f l atTop) :
    Tendsto (fun x => f x • r) l atTop := by
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    r : R
    inst✝¹ : LinearOrderedAddCommGroup R
    inst✝ : Archimedean R
    f : α → Int
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HSMul.hSMul (f x) r) l Filter.atTop
  -/
  refine tendsto_atTop.mpr fun s => ?_
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    r : R
    inst✝¹ : LinearOrderedAddCommGroup R
    inst✝ : Archimedean R
    f : α → Int
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    s : R
    ⊢ Filter.Eventually (fun a => LE.le s (HSMul.hSMul (f a) r)) l
  -/
  obtain ⟨n : ℕ, hn : s ≤ n • r⟩ := Archimedean.arch s hr
  /-
    case intro
    α : Type u_1
    R : Type u_2
    l : Filter α
    r : R
    inst✝¹ : LinearOrderedAddCommGroup R
    inst✝ : Archimedean R
    f : α → Int
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    s : R
    n : Nat
    hn : LE.le s (HSMul.hSMul n r)
    ⊢ Filter.Eventually (fun a => LE.le s (HSMul.hSMul (f a) r)) l
  -/
  replace hn : s ≤ (n : ℤ) • r := by simpa
  /-
    case intro
    α : Type u_1
    R : Type u_2
    l : Filter α
    r : R
    inst✝¹ : LinearOrderedAddCommGroup R
    inst✝ : Archimedean R
    f : α → Int
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    s : R
    n : Nat
    hn : LE.le s (HSMul.hSMul (↑n) r)
    ⊢ Filter.Eventually (fun a => LE.le s (HSMul.hSMul (f a) r)) l
  -/
  exact (tendsto_atTop.mp hf n).mono fun a ha => hn.trans (zsmul_le_zsmul_left hr.le ha)
  /-
    🎉 no goals
  -/


theorem Tendsto.atTop_zsmul_neg_const {f : α → ℤ} (hr : r < 0) (hf : Tendsto f l atTop) :
                                             /-
                                               α : Type u_1
                                               R : Type u_2
                                               l : Filter α
                                               r : R
                                               inst✝¹ : LinearOrderedAddCommGroup R
                                               inst✝ : Archimedean R
                                               f : α → Int
                                               hr : LT.lt r 0
                                               hf : Filter.Tendsto f l Filter.atTop
                                               ⊢ Filter.Tendsto (fun x => HSMul.hSMul (f x) r) l Filter.atBot
                                             -/
    Tendsto (fun x => f x • r) l atBot := by simpa using hf.atTop_zsmul_const (neg_pos.2 hr)
                                             /-
                                               🎉 no goals
                                             -/


theorem Tendsto.atBot_zsmul_const {f : α → ℤ} (hr : 0 < r) (hf : Tendsto f l atBot) :
    Tendsto (fun x => f x • r) l atBot := by
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    r : R
    inst✝¹ : LinearOrderedAddCommGroup R
    inst✝ : Archimedean R
    f : α → Int
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atBot
    ⊢ Filter.Tendsto (fun x => HSMul.hSMul (f x) r) l Filter.atBot
  -/
  simp only [← tendsto_neg_atTop_iff, ← neg_zsmul] at hf ⊢
  /-
    α : Type u_1
    R : Type u_2
    l : Filter α
    r : R
    inst✝¹ : LinearOrderedAddCommGroup R
    inst✝ : Archimedean R
    f : α → Int
    hr : LT.lt 0 r
    hf : Filter.Tendsto (fun x => Neg.neg (f x)) l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HSMul.hSMul (Neg.neg (f x)) r) l Filter.atTop
  -/
  exact hf.atTop_zsmul_const hr
  /-
    🎉 no goals
  -/


theorem Tendsto.atBot_zsmul_neg_const {f : α → ℤ} (hr : r < 0) (hf : Tendsto f l atBot) :
                                             /-
                                               α : Type u_1
                                               R : Type u_2
                                               l : Filter α
                                               r : R
                                               inst✝¹ : LinearOrderedAddCommGroup R
                                               inst✝ : Archimedean R
                                               f : α → Int
                                               hr : LT.lt r 0
                                               hf : Filter.Tendsto f l Filter.atBot
                                               ⊢ Filter.Tendsto (fun x => HSMul.hSMul (f x) r) l Filter.atTop
                                             -/
    Tendsto (fun x => f x • r) l atTop := by simpa using hf.atBot_zsmul_const (neg_pos.2 hr)
                                             /-
                                               🎉 no goals
                                             -/


