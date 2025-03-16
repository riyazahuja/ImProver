/-- `multiplicity.Finite a b` indicates that the multiplicity of `a` in `b` is finite. -/
abbrev FiniteMultiplicity [Monoid α] (a b : α) : Prop :=
  ∃ n : ℕ, ¬a ^ (n + 1) ∣ b


@[deprecated (since := "2024-11-30")] alias multiplicity.Finite := FiniteMultiplicity


open scoped Classical in
/-- `emultiplicity a b` returns the largest natural number `n` such that
  `a ^ n ∣ b`, as an `ℕ∞`. If `∀ n, a ^ n ∣ b` then it returns `⊤`. -/
noncomputable def emultiplicity [Monoid α] (a b : α) : ℕ∞ :=
  if h : FiniteMultiplicity a b then Nat.find h else ⊤


/-- A `ℕ`-valued version of `emultiplicity`, returning `1` instead of `⊤`. -/
noncomputable def multiplicity [Monoid α] (a b : α) : ℕ :=
  (emultiplicity a b).untop' 1


@[simp]
theorem emultiplicity_eq_top :
    emultiplicity a b = ⊤ ↔ ¬FiniteMultiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    ⊢ Iff (Eq (emultiplicity a b) Top.top) (Not (FiniteMultiplicity a b))
  -/
  simp [emultiplicity]
  /-
    🎉 no goals
  -/


theorem emultiplicity_lt_top {a b : α} : emultiplicity a b < ⊤ ↔ FiniteMultiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    ⊢ Iff (LT.lt (emultiplicity a b) Top.top) (FiniteMultiplicity a b)
  -/
  simp [lt_top_iff_ne_top, emultiplicity_eq_top]
  /-
    🎉 no goals
  -/


theorem finiteMultiplicity_iff_emultiplicity_ne_top :
                                                         /-
                                                           α : Type u_1
                                                           inst✝ : Monoid α
                                                           a b : α
                                                           ⊢ Iff (FiniteMultiplicity a b) (Ne (emultiplicity a b) Top.top)
                                                         -/
    FiniteMultiplicity a b ↔ emultiplicity a b ≠ ⊤ := by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


@[deprecated (since := "2024-11-30")]
alias finite_iff_emultiplicity_ne_top := finiteMultiplicity_iff_emultiplicity_ne_top


alias ⟨FiniteMultiplicity.emultiplicity_ne_top, _⟩ := finite_iff_emultiplicity_ne_top


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.emultiplicity_ne_top := FiniteMultiplicity.emultiplicity_ne_top


@[deprecated (since := "2024-11-08")]
alias Finite.emultiplicity_ne_top := FiniteMultiplicity.emultiplicity_ne_top


theorem finiteMultiplicity_of_emultiplicity_eq_natCast {n : ℕ} (h : emultiplicity a b = n) :
    FiniteMultiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    h : Eq (emultiplicity a b) ↑n
    ⊢ FiniteMultiplicity a b
  -/
  by_contra! nh
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    h : Eq (emultiplicity a b) ↑n
    nh : Not (FiniteMultiplicity a b)
    ⊢ False
  -/
  rw [← emultiplicity_eq_top, h] at nh
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    h : Eq (emultiplicity a b) ↑n
    nh : Eq (↑n) Top.top
    ⊢ False
  -/
  trivial
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias finite_of_emultiplicity_eq_natCast := finiteMultiplicity_of_emultiplicity_eq_natCast


theorem multiplicity_eq_of_emultiplicity_eq_some {n : ℕ} (h : emultiplicity a b = n) :
    multiplicity a b = n := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    h : Eq (emultiplicity a b) ↑n
    ⊢ Eq (multiplicity a b) n
  -/
  simp [multiplicity, h]
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    h : Eq (emultiplicity a b) ↑n
    ⊢ Eq (WithTop.untop' 1 ↑n) n
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem emultiplicity_ne_of_multiplicity_ne {n : ℕ} :
    multiplicity a b ≠ n → emultiplicity a b ≠ n :=
  mt multiplicity_eq_of_emultiplicity_eq_some


theorem FiniteMultiplicity.emultiplicity_eq_multiplicity (h : FiniteMultiplicity a b) :
    emultiplicity a b = multiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    h : FiniteMultiplicity a b
    ⊢ Eq (emultiplicity a b) ↑(multiplicity a b)
  -/
  cases hm : emultiplicity a b
    /-
      case top
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      h : FiniteMultiplicity a b
      hm : Eq (emultiplicity a b) Top.top
      ⊢ Eq Top.top ↑(multiplicity a b)
    -/
  · simp [h] at hm
    /-
      🎉 no goals
    -/
  /-
    case coe
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    h : FiniteMultiplicity a b
    a✝ : Nat
    hm : Eq (emultiplicity a b) ↑a✝
    ⊢ Eq ↑a✝ ↑(multiplicity a b)
  -/
  rw [multiplicity_eq_of_emultiplicity_eq_some hm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.emultiplicity_eq_multiplicity :=
  FiniteMultiplicity.emultiplicity_eq_multiplicity


theorem FiniteMultiplicity.emultiplicity_eq_iff_multiplicity_eq {n : ℕ}
    (h : FiniteMultiplicity a b) : emultiplicity a b = n ↔ multiplicity a b = n := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    h : FiniteMultiplicity a b
    ⊢ Iff (Eq (emultiplicity a b) ↑n) (Eq (multiplicity a b) n)
  -/
  simp [h.emultiplicity_eq_multiplicity]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.emultiplicity_eq_iff_multiplicity_eq :=
  FiniteMultiplicity.emultiplicity_eq_iff_multiplicity_eq


theorem emultiplicity_eq_iff_multiplicity_eq_of_ne_one {n : ℕ} (h : n ≠ 1) :
    emultiplicity a b = n ↔ multiplicity a b = n := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    h : Ne n 1
    ⊢ Iff (Eq (emultiplicity a b) ↑n) (Eq (multiplicity a b) n)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      n : Nat
      h : Ne n 1
      ⊢ Eq (emultiplicity a b) ↑n → Eq (multiplicity a b) n
    -/
  · exact multiplicity_eq_of_emultiplicity_eq_some
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      n : Nat
      h : Ne n 1
      ⊢ Eq (multiplicity a b) n → Eq (emultiplicity a b) ↑n
    -/
  · intro h₂
    /-
      case mpr
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      n : Nat
      h : Ne n 1
      h₂ : Eq (multiplicity a b) n
      ⊢ Eq (emultiplicity a b) ↑n
    -/
    simpa [multiplicity, WithTop.untop'_eq_iff, h] using h₂
    /-
      🎉 no goals
    -/


theorem emultiplicity_eq_zero_iff_multiplicity_eq_zero :
    emultiplicity a b = 0 ↔ multiplicity a b = 0 :=
  emultiplicity_eq_iff_multiplicity_eq_of_ne_one zero_ne_one


@[simp]
theorem multiplicity_eq_one_of_not_finiteMultiplicity (h : ¬FiniteMultiplicity a b) :
    multiplicity a b = 1 := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    h : Not (FiniteMultiplicity a b)
    ⊢ Eq (multiplicity a b) 1
  -/
  simp [multiplicity, emultiplicity_eq_top.2 h]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity_eq_one_of_not_finite :=
  multiplicity_eq_one_of_not_finiteMultiplicity


@[simp]
theorem multiplicity_le_emultiplicity :
    multiplicity a b ≤ emultiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    ⊢ LE.le (↑(multiplicity a b)) (emultiplicity a b)
  -/
  by_cases hf : FiniteMultiplicity a b
    /-
      case pos
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      hf : FiniteMultiplicity a b
      ⊢ LE.le (↑(multiplicity a b)) (emultiplicity a b)
    -/
  · simp [hf.emultiplicity_eq_multiplicity]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      hf : Not (FiniteMultiplicity a b)
      ⊢ LE.le (↑(multiplicity a b)) (emultiplicity a b)
    -/
  · simp [hf, emultiplicity_eq_top.2]
    /-
      🎉 no goals
    -/


@[simp]
theorem multiplicity_eq_of_emultiplicity_eq {c d : β}
    (h : emultiplicity a b = emultiplicity c d) : multiplicity a b = multiplicity c d := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Monoid α
    inst✝ : Monoid β
    a b : α
    c d : β
    h : Eq (emultiplicity a b) (emultiplicity c d)
    ⊢ Eq (multiplicity a b) (multiplicity c d)
  -/
  unfold multiplicity
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Monoid α
    inst✝ : Monoid β
    a b : α
    c d : β
    h : Eq (emultiplicity a b) (emultiplicity c d)
    ⊢ Eq (WithTop.untop' 1 (emultiplicity a b)) (WithTop.untop' 1 (emultiplicity c …
  -/
  rw [h]
  /-
    🎉 no goals
  -/


theorem multiplicity_le_of_emultiplicity_le {n : ℕ} (h : emultiplicity a b ≤ n) :
    multiplicity a b ≤ n := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    h : LE.le (emultiplicity a b) ↑n
    ⊢ LE.le (multiplicity a b) n
  -/
  exact_mod_cast multiplicity_le_emultiplicity.trans h
  /-
    🎉 no goals
  -/


theorem FiniteMultiplicity.emultiplicity_le_of_multiplicity_le (hfin : FiniteMultiplicity a b)
    {n : ℕ} (h : multiplicity a b ≤ n) : emultiplicity a b ≤ n := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    n : Nat
    h : LE.le (multiplicity a b) n
    ⊢ LE.le (emultiplicity a b) ↑n
  -/
  rw [emultiplicity_eq_multiplicity hfin]
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    n : Nat
    h : LE.le (multiplicity a b) n
    ⊢ LE.le ↑(multiplicity a b) ↑n
  -/
  assumption_mod_cast
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.emultiplicity_le_of_multiplicity_le :=
  FiniteMultiplicity.emultiplicity_le_of_multiplicity_le


theorem le_emultiplicity_of_le_multiplicity {n : ℕ} (h : n ≤ multiplicity a b) :
    n ≤ emultiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    h : LE.le n (multiplicity a b)
    ⊢ LE.le (↑n) (emultiplicity a b)
  -/
  exact_mod_cast (WithTop.coe_mono h).trans multiplicity_le_emultiplicity
  /-
    🎉 no goals
  -/


theorem FiniteMultiplicity.le_multiplicity_of_le_emultiplicity (hfin : FiniteMultiplicity a b)
    {n : ℕ} (h : n ≤ emultiplicity a b) : n ≤ multiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    n : Nat
    h : LE.le (↑n) (emultiplicity a b)
    ⊢ LE.le n (multiplicity a b)
  -/
  rw [emultiplicity_eq_multiplicity hfin] at h
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    n : Nat
    h : LE.le ↑n ↑(multiplicity a b)
    ⊢ LE.le n (multiplicity a b)
  -/
  assumption_mod_cast
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.le_multiplicity_of_le_emultiplicity :=
  FiniteMultiplicity.le_multiplicity_of_le_emultiplicity


theorem multiplicity_lt_of_emultiplicity_lt {n : ℕ} (h : emultiplicity a b < n) :
    multiplicity a b < n := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    h : LT.lt (emultiplicity a b) ↑n
    ⊢ LT.lt (multiplicity a b) n
  -/
  exact_mod_cast multiplicity_le_emultiplicity.trans_lt h
  /-
    🎉 no goals
  -/


theorem FiniteMultiplicity.emultiplicity_lt_of_multiplicity_lt (hfin : FiniteMultiplicity a b)
    {n : ℕ} (h : multiplicity a b < n) : emultiplicity a b < n := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    n : Nat
    h : LT.lt (multiplicity a b) n
    ⊢ LT.lt (emultiplicity a b) ↑n
  -/
  rw [emultiplicity_eq_multiplicity hfin]
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    n : Nat
    h : LT.lt (multiplicity a b) n
    ⊢ LT.lt ↑(multiplicity a b) ↑n
  -/
  assumption_mod_cast
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.emultiplicity_lt_of_multiplicity_lt :=
  FiniteMultiplicity.emultiplicity_lt_of_multiplicity_lt


theorem lt_emultiplicity_of_lt_multiplicity {n : ℕ} (h : n < multiplicity a b) :
    n < emultiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    h : LT.lt n (multiplicity a b)
    ⊢ LT.lt (↑n) (emultiplicity a b)
  -/
  exact_mod_cast (WithTop.coe_strictMono h).trans_le multiplicity_le_emultiplicity
  /-
    🎉 no goals
  -/


theorem FiniteMultiplicity.lt_multiplicity_of_lt_emultiplicity (hfin : FiniteMultiplicity a b)
    {n : ℕ} (h : n < emultiplicity a b) : n < multiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    n : Nat
    h : LT.lt (↑n) (emultiplicity a b)
    ⊢ LT.lt n (multiplicity a b)
  -/
  rw [emultiplicity_eq_multiplicity hfin] at h
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    n : Nat
    h : LT.lt ↑n ↑(multiplicity a b)
    ⊢ LT.lt n (multiplicity a b)
  -/
  assumption_mod_cast
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.lt_multiplicity_of_lt_emultiplicity :=
  FiniteMultiplicity.lt_multiplicity_of_lt_emultiplicity


theorem emultiplicity_pos_iff :
    0 < emultiplicity a b ↔ 0 < multiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    ⊢ Iff (LT.lt 0 (emultiplicity a b)) (LT.lt 0 (multiplicity a b))
  -/
  simp [pos_iff_ne_zero, pos_iff_ne_zero, emultiplicity_eq_zero_iff_multiplicity_eq_zero]
  /-
    🎉 no goals
  -/


theorem FiniteMultiplicity.def : FiniteMultiplicity a b ↔ ∃ n : ℕ, ¬a ^ (n + 1) ∣ b :=
  Iff.rfl


@[deprecated (since := "2024-11-30")] alias multiplicity.Finite.def := FiniteMultiplicity.def


theorem FiniteMultiplicity.not_dvd_of_one_right : FiniteMultiplicity a 1 → ¬a ∣ 1 :=
  fun ⟨n, hn⟩ ⟨d, hd⟩ => hn ⟨d ^ (n + 1), (pow_mul_pow_eq_one (n + 1) hd.symm).symm⟩


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.not_dvd_of_one_right := FiniteMultiplicity.not_dvd_of_one_right


@[norm_cast]
theorem Int.natCast_emultiplicity (a b : ℕ) :
    emultiplicity (a : ℤ) (b : ℤ) = emultiplicity a b := by
  /-
    a b : Nat
    ⊢ Eq (emultiplicity ↑a ↑b) (emultiplicity a b)
  -/
  unfold emultiplicity FiniteMultiplicity
  /-
    a b : Nat
    ⊢ Eq (dite (Exists fun n => Not (Dvd.dvd (HPow.hPow (↑a) (HAdd.hAdd n 1)) ↑b)) …
  -/
             /-
               🎉 no goals
             -/
  congr! <;> norm_cast
             /-
               🎉 no goals
             -/


@[norm_cast]
theorem Int.natCast_multiplicity (a b : ℕ) : multiplicity (a : ℤ) (b : ℤ) = multiplicity a b :=
  multiplicity_eq_of_emultiplicity_eq (natCast_emultiplicity a b)


@[deprecated (since := "2024-04-05")] alias Int.coe_nat_multiplicity := Int.natCast_multiplicity


theorem FiniteMultiplicity.not_iff_forall : ¬FiniteMultiplicity a b ↔ ∀ n : ℕ, a ^ n ∣ b :=
  ⟨fun h n =>
    Nat.casesOn n
      (by
        /-
          α : Type u_1
          inst✝ : Monoid α
          a b : α
          h : Not (FiniteMultiplicity a b)
          n : Nat
          ⊢ Dvd.dvd (HPow.hPow a Nat.zero) b
        -/
        rw [_root_.pow_zero]
        /-
          α : Type u_1
          inst✝ : Monoid α
          a b : α
          h : Not (FiniteMultiplicity a b)
          n : Nat
          ⊢ Dvd.dvd 1 b
        -/
        exact one_dvd _)
        /-
          🎉 no goals
        -/
          /-
            α : Type u_1
            inst✝ : Monoid α
            a b : α
            h : Not (FiniteMultiplicity a b)
            n : Nat
            ⊢ ∀ (n : Nat), Dvd.dvd (HPow.hPow a n.succ) b
          -/
      (by simpa [FiniteMultiplicity] using h),
          /-
            🎉 no goals
          -/
       /-
         α : Type u_1
         inst✝ : Monoid α
         a b : α
         ⊢ (∀ (n : Nat), Dvd.dvd (HPow.hPow a n) b) → Not (FiniteMultiplicity a b)
       -/
    by simp [FiniteMultiplicity, multiplicity]; tauto⟩
                                                /-
                                                  🎉 no goals
                                                -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.not_iff_forall := FiniteMultiplicity.not_iff_forall


theorem FiniteMultiplicity.not_unit (h : FiniteMultiplicity a b) : ¬IsUnit a :=
  let ⟨n, hn⟩ := h
  hn ∘ IsUnit.dvd ∘ IsUnit.pow (n + 1)


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.not_unit := FiniteMultiplicity.not_unit


theorem FiniteMultiplicity.mul_left {c : α} :
    FiniteMultiplicity a (b * c) → FiniteMultiplicity a b := fun ⟨n, hn⟩ =>
  ⟨n, fun h => hn (h.trans (dvd_mul_right _ _))⟩


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.mul_left := FiniteMultiplicity.mul_left


theorem pow_dvd_of_le_emultiplicity {k : ℕ} (hk : k ≤ emultiplicity a b) :
    a ^ k ∣ b := by classical
  cases k
  · simp
  unfold emultiplicity at hk
  split at hk
  · norm_cast at hk
    simpa using (Nat.find_min _ (lt_of_succ_le hk))
  · apply FiniteMultiplicity.not_iff_forall.mp ‹_›


theorem pow_dvd_of_le_multiplicity {k : ℕ} (hk : k ≤ multiplicity a b) :
    a ^ k ∣ b := pow_dvd_of_le_emultiplicity (le_emultiplicity_of_le_multiplicity hk)


@[simp]
theorem pow_multiplicity_dvd (a b : α) : a ^ (multiplicity a b) ∣ b :=
  pow_dvd_of_le_multiplicity le_rfl


theorem not_pow_dvd_of_emultiplicity_lt {m : ℕ} (hm : emultiplicity a b < m) :
    ¬a ^ m ∣ b := fun nh => by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    m : Nat
    hm : LT.lt (emultiplicity a b) ↑m
    nh : Dvd.dvd (HPow.hPow a m) b
    ⊢ False
  -/
  unfold emultiplicity at hm
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    m : Nat
    hm : LT.lt (dite (FiniteMultiplicity a b) (fun h => ↑(Nat.find h)) fun h => To …
    nh : Dvd.dvd (HPow.hPow a m) b
    ⊢ False
  -/
  split at hm
    /-
      case isTrue
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      m : Nat
      nh : Dvd.dvd (HPow.hPow a m) b
      h✝ : FiniteMultiplicity a b
      hm : LT.lt ↑(Nat.find h✝) ↑m
      ⊢ False
    -/
  · simp only [cast_lt, find_lt_iff] at hm
    /-
      case isTrue
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      m : Nat
      nh : Dvd.dvd (HPow.hPow a m) b
      h✝ : FiniteMultiplicity a b
      hm : Exists fun m_1 => And (LT.lt m_1 m) (Not (Dvd.dvd (HPow.hPow a (HAdd.hAdd …
      ⊢ False
    -/
    obtain ⟨n, hn1, hn2⟩ := hm
    /-
      case isTrue.intro.intro
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      m : Nat
      nh : Dvd.dvd (HPow.hPow a m) b
      h✝ : FiniteMultiplicity a b
      n : Nat
      hn1 : LT.lt n m
      hn2 : Not (Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1)) b)
      ⊢ False
    -/
    exact hn2 ((pow_dvd_pow _ hn1).trans nh)
    /-
      🎉 no goals
    -/
    /-
      case isFalse
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      m : Nat
      nh : Dvd.dvd (HPow.hPow a m) b
      h✝ : Not (FiniteMultiplicity a b)
      hm : LT.lt Top.top ↑m
      ⊢ False
    -/
  · simp at hm
    /-
      🎉 no goals
    -/


theorem FiniteMultiplicity.not_pow_dvd_of_multiplicity_lt (hf : FiniteMultiplicity a b) {m : ℕ}
    (hm : multiplicity a b < m) : ¬a ^ m ∣ b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hf : FiniteMultiplicity a b
    m : Nat
    hm : LT.lt (multiplicity a b) m
    ⊢ Not (Dvd.dvd (HPow.hPow a m) b)
  -/
  apply not_pow_dvd_of_emultiplicity_lt
  /-
    case hm
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hf : FiniteMultiplicity a b
    m : Nat
    hm : LT.lt (multiplicity a b) m
    ⊢ LT.lt (emultiplicity a b) ↑m
  -/
  rw [hf.emultiplicity_eq_multiplicity]
  /-
    case hm
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hf : FiniteMultiplicity a b
    m : Nat
    hm : LT.lt (multiplicity a b) m
    ⊢ LT.lt ↑(multiplicity a b) ↑m
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.not_pow_dvd_of_multiplicity_lt :=
  FiniteMultiplicity.not_pow_dvd_of_multiplicity_lt


theorem multiplicity_pos_of_dvd (hdiv : a ∣ b) : 0 < multiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hdiv : Dvd.dvd a b
    ⊢ LT.lt 0 (multiplicity a b)
  -/
  refine zero_lt_iff.2 fun h => ?_
  simpa [hdiv] using FiniteMultiplicity.not_pow_dvd_of_multiplicity_lt
    (by by_contra! nh; simp [nh] at h) (lt_one_iff.mpr h)


theorem emultiplicity_pos_of_dvd (hdiv : a ∣ b) : 0 < emultiplicity a b :=
  lt_emultiplicity_of_lt_multiplicity (multiplicity_pos_of_dvd hdiv)


theorem emultiplicity_eq_of_dvd_of_not_dvd {k : ℕ} (hk : a ^ k ∣ b) (hsucc : ¬a ^ (k + 1) ∣ b) :
    emultiplicity a b = k := by classical
  have : FiniteMultiplicity a b := ⟨k, hsucc⟩
  simp only [emultiplicity, this, ↓reduceDIte, Nat.cast_inj, find_eq_iff, hsucc, not_false_eq_true,
    Decidable.not_not, true_and]
  exact fun n hn ↦ (pow_dvd_pow _ hn).trans hk


theorem multiplicity_eq_of_dvd_of_not_dvd {k : ℕ} (hk : a ^ k ∣ b) (hsucc : ¬a ^ (k + 1) ∣ b) :
    multiplicity a b = k :=
  multiplicity_eq_of_emultiplicity_eq_some (emultiplicity_eq_of_dvd_of_not_dvd hk hsucc)


theorem le_emultiplicity_of_pow_dvd {k : ℕ} (hk : a ^ k ∣ b) :
    k ≤ emultiplicity a b :=
  le_of_not_gt fun hk' => not_pow_dvd_of_emultiplicity_lt hk' hk


theorem FiniteMultiplicity.le_multiplicity_of_pow_dvd (hf : FiniteMultiplicity a b)
    {k : ℕ} (hk : a ^ k ∣ b) : k ≤ multiplicity a b :=
  hf.le_multiplicity_of_le_emultiplicity (le_emultiplicity_of_pow_dvd hk)


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.le_multiplicity_of_pow_dvd :=
  FiniteMultiplicity.le_multiplicity_of_pow_dvd


theorem pow_dvd_iff_le_emultiplicity {k : ℕ} :
    a ^ k ∣ b ↔ k ≤ emultiplicity a b :=
  ⟨le_emultiplicity_of_pow_dvd, pow_dvd_of_le_emultiplicity⟩


theorem FiniteMultiplicity.pow_dvd_iff_le_multiplicity (hf : FiniteMultiplicity a b) {k : ℕ} :
    a ^ k ∣ b ↔ k ≤ multiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hf : FiniteMultiplicity a b
    k : Nat
    ⊢ Iff (Dvd.dvd (HPow.hPow a k) b) (LE.le k (multiplicity a b))
  -/
  exact_mod_cast hf.emultiplicity_eq_multiplicity ▸ pow_dvd_iff_le_emultiplicity
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.pow_dvd_iff_le_multiplicity :=
  FiniteMultiplicity.pow_dvd_iff_le_multiplicity


theorem emultiplicity_lt_iff_not_dvd {k : ℕ} :
                                             /-
                                               α : Type u_1
                                               inst✝ : Monoid α
                                               a b : α
                                               k : Nat
                                               ⊢ Iff (LT.lt (emultiplicity a b) ↑k) (Not (Dvd.dvd (HPow.hPow a k) b))
                                             -/
    emultiplicity a b < k ↔ ¬a ^ k ∣ b := by rw [pow_dvd_iff_le_emultiplicity, not_le]
                                             /-
                                               🎉 no goals
                                             -/


theorem FiniteMultiplicity.multiplicity_lt_iff_not_dvd {k : ℕ} (hf : FiniteMultiplicity a b) :
                                            /-
                                              α : Type u_1
                                              inst✝ : Monoid α
                                              a b : α
                                              k : Nat
                                              hf : FiniteMultiplicity a b
                                              ⊢ Iff (LT.lt (multiplicity a b) k) (Not (Dvd.dvd (HPow.hPow a k) b))
                                            -/
    multiplicity a b < k ↔ ¬a ^ k ∣ b := by rw [hf.pow_dvd_iff_le_multiplicity, not_le]
                                            /-
                                              🎉 no goals
                                            -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.multiplicity_lt_iff_not_dvd :=
  FiniteMultiplicity.multiplicity_lt_iff_not_dvd


theorem emultiplicity_eq_coe {n : ℕ} :
    emultiplicity a b = n ↔ a ^ n ∣ b ∧ ¬a ^ (n + 1) ∣ b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    n : Nat
    ⊢ Iff (Eq (emultiplicity a b) ↑n) (And (Dvd.dvd (HPow.hPow a n) b) (Not (Dvd.d …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      n : Nat
      ⊢ Eq (emultiplicity a b) ↑n → And (Dvd.dvd (HPow.hPow a n) b) (Not (Dvd.dvd (H …
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      n : Nat
      h : Eq (emultiplicity a b) ↑n
      ⊢ And (Dvd.dvd (HPow.hPow a n) b) (Not (Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1))  …
    -/
    constructor
      /-
        case mp.left
        α : Type u_1
        inst✝ : Monoid α
        a b : α
        n : Nat
        h : Eq (emultiplicity a b) ↑n
        ⊢ Dvd.dvd (HPow.hPow a n) b
      -/
    · apply pow_dvd_of_le_emultiplicity
      /-
        case mp.left.hk
        α : Type u_1
        inst✝ : Monoid α
        a b : α
        n : Nat
        h : Eq (emultiplicity a b) ↑n
        ⊢ LE.le (↑n) (emultiplicity a b)
      -/
      simp [h]
      /-
        🎉 no goals
      -/
      /-
        case mp.right
        α : Type u_1
        inst✝ : Monoid α
        a b : α
        n : Nat
        h : Eq (emultiplicity a b) ↑n
        ⊢ Not (Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1)) b)
      -/
    · apply not_pow_dvd_of_emultiplicity_lt
      /-
        case mp.right.hm
        α : Type u_1
        inst✝ : Monoid α
        a b : α
        n : Nat
        h : Eq (emultiplicity a b) ↑n
        ⊢ LT.lt (emultiplicity a b) ↑(HAdd.hAdd n 1)
      -/
      rw [h]
      /-
        case mp.right.hm
        α : Type u_1
        inst✝ : Monoid α
        a b : α
        n : Nat
        h : Eq (emultiplicity a b) ↑n
        ⊢ LT.lt ↑n ↑(HAdd.hAdd n 1)
      -/
      norm_cast
      /-
        case mp.right.hm
        α : Type u_1
        inst✝ : Monoid α
        a b : α
        n : Nat
        h : Eq (emultiplicity a b) ↑n
        ⊢ LT.lt n (HAdd.hAdd n 1)
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      n : Nat
      ⊢ And (Dvd.dvd (HPow.hPow a n) b) (Not (Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1))  …
    -/
  · rw [and_imp]
    /-
      case mpr
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      n : Nat
      ⊢ Dvd.dvd (HPow.hPow a n) b → Not (Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1)) b) →  …
    -/
    apply emultiplicity_eq_of_dvd_of_not_dvd
    /-
      🎉 no goals
    -/


theorem FiniteMultiplicity.multiplicity_eq_iff (hf : FiniteMultiplicity a b) {n : ℕ} :
    multiplicity a b = n ↔ a ^ n ∣ b ∧ ¬a ^ (n + 1) ∣ b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hf : FiniteMultiplicity a b
    n : Nat
    ⊢ Iff (Eq (multiplicity a b) n) (And (Dvd.dvd (HPow.hPow a n) b) (Not (Dvd.dvd …
  -/
  simp [← emultiplicity_eq_coe, hf.emultiplicity_eq_multiplicity]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.multiplicity_eq_iff := FiniteMultiplicity.multiplicity_eq_iff


@[simp]
theorem FiniteMultiplicity.not_of_isUnit_left (b : α) (ha : IsUnit a) : ¬FiniteMultiplicity a b :=
  (·.not_unit ha)


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.not_of_isUnit_left := FiniteMultiplicity.not_of_isUnit_left


                                                                                    /-
                                                                                      α : Type u_1
                                                                                      inst✝ : Monoid α
                                                                                      b : α
                                                                                      ⊢ Not (FiniteMultiplicity 1 b)
                                                                                    -/
theorem FiniteMultiplicity.not_of_one_left (b : α) : ¬ FiniteMultiplicity 1 b := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.not_of_one_left := FiniteMultiplicity.not_of_one_left


@[simp]
theorem emultiplicity_one_left (b : α) : emultiplicity 1 b = ⊤ :=
  emultiplicity_eq_top.2 (FiniteMultiplicity.not_of_one_left _)


@[simp]
theorem FiniteMultiplicity.one_right (ha : FiniteMultiplicity a 1) : multiplicity a 1 = 0 := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a : α
    ha : FiniteMultiplicity a 1
    ⊢ Eq (multiplicity a 1) 0
  -/
  simp [ha.multiplicity_eq_iff, ha.not_dvd_of_one_right]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.one_right := FiniteMultiplicity.one_right


theorem FiniteMultiplicity.not_of_unit_left (a : α) (u : αˣ) : ¬ FiniteMultiplicity (u : α) a :=
  FiniteMultiplicity.not_of_isUnit_left a u.isUnit


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.not_of_unit_left := FiniteMultiplicity.not_of_unit_left


theorem emultiplicity_eq_zero :
    emultiplicity a b = 0 ↔ ¬a ∣ b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    ⊢ Iff (Eq (emultiplicity a b) 0) (Not (Dvd.dvd a b))
  -/
  by_cases hf : FiniteMultiplicity a b
    /-
      case pos
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      hf : FiniteMultiplicity a b
      ⊢ Iff (Eq (emultiplicity a b) 0) (Not (Dvd.dvd a b))
    -/
  · rw [← ENat.coe_zero, emultiplicity_eq_coe]
    /-
      case pos
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      hf : FiniteMultiplicity a b
      ⊢ Iff (And (Dvd.dvd (HPow.hPow a 0) b) (Not (Dvd.dvd (HPow.hPow a (HAdd.hAdd 0 …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : Monoid α
      a b : α
      hf : Not (FiniteMultiplicity a b)
      ⊢ Iff (Eq (emultiplicity a b) 0) (Not (Dvd.dvd a b))
    -/
  · simpa [emultiplicity_eq_top.2 hf] using FiniteMultiplicity.not_iff_forall.1 hf 1
    /-
      🎉 no goals
    -/


theorem multiplicity_eq_zero :
    multiplicity a b = 0 ↔ ¬a ∣ b :=
  (emultiplicity_eq_iff_multiplicity_eq_of_ne_one zero_ne_one).symm.trans emultiplicity_eq_zero


theorem emultiplicity_ne_zero :
    emultiplicity a b ≠ 0 ↔ a ∣ b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    ⊢ Iff (Ne (emultiplicity a b) 0) (Dvd.dvd a b)
  -/
  simp [emultiplicity_eq_zero]
  /-
    🎉 no goals
  -/


theorem multiplicity_ne_zero :
    multiplicity a b ≠ 0 ↔ a ∣ b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    ⊢ Iff (Ne (multiplicity a b) 0) (Dvd.dvd a b)
  -/
  simp [multiplicity_eq_zero]
  /-
    🎉 no goals
  -/


theorem FiniteMultiplicity.exists_eq_pow_mul_and_not_dvd (hfin : FiniteMultiplicity a b) :
    ∃ c : α, b = a ^ multiplicity a b * c ∧ ¬a ∣ c := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    ⊢ Exists fun c => And (Eq b (HMul.hMul (HPow.hPow a (multiplicity a b)) c)) (N …
  -/
  obtain ⟨c, hc⟩ := pow_multiplicity_dvd a b
  /-
    case intro
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    c : α
    hc : Eq b (HMul.hMul (HPow.hPow a (multiplicity a b)) c)
    ⊢ Exists fun c => And (Eq b (HMul.hMul (HPow.hPow a (multiplicity a b)) c)) (N …
  -/
  refine ⟨c, hc, ?_⟩
  /-
    case intro
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    c : α
    hc : Eq b (HMul.hMul (HPow.hPow a (multiplicity a b)) c)
    ⊢ Not (Dvd.dvd a c)
  -/
  rintro ⟨k, hk⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    c : α
    hc : Eq b (HMul.hMul (HPow.hPow a (multiplicity a b)) c)
    k : α
    hk : Eq c (HMul.hMul a k)
    ⊢ False
  -/
  rw [hk, ← mul_assoc, ← _root_.pow_succ] at hc
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    c k : α
    hc : Eq b (HMul.hMul (HPow.hPow a (HAdd.hAdd (multiplicity a b) 1)) k)
    hk : Eq c (HMul.hMul a k)
    ⊢ False
  -/
  have h₁ : a ^ (multiplicity a b + 1) ∣ b := ⟨k, hc⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    hfin : FiniteMultiplicity a b
    c k : α
    hc : Eq b (HMul.hMul (HPow.hPow a (HAdd.hAdd (multiplicity a b) 1)) k)
    hk : Eq c (HMul.hMul a k)
    h₁ : Dvd.dvd (HPow.hPow a (HAdd.hAdd (multiplicity a b) 1)) b
    ⊢ False
  -/
  exact (hfin.multiplicity_eq_iff.1 (by simp)).2 h₁
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.exists_eq_pow_mul_and_not_dvd :=
  FiniteMultiplicity.exists_eq_pow_mul_and_not_dvd


theorem emultiplicity_le_emultiplicity_iff {c d : β} :
    emultiplicity a b ≤ emultiplicity c d ↔ ∀ n : ℕ, a ^ n ∣ b → c ^ n ∣ d := by classical
  constructor
  · exact fun h n hab ↦ pow_dvd_of_le_emultiplicity (le_trans (le_emultiplicity_of_pow_dvd hab) h)
  · intro h
    unfold emultiplicity
    -- aesop? says
    split
    next h_1 =>
      obtain ⟨w, h_1⟩ := h_1
      split
      next h_2 =>
        simp_all only [cast_le, le_find_iff, lt_find_iff, Decidable.not_not, le_refl,
          not_true_eq_false, not_false_eq_true, implies_true]
      next h_2 => simp_all only [not_exists, Decidable.not_not, le_top]
    next h_1 =>
      simp_all only [not_exists, Decidable.not_not, not_true_eq_false, top_le_iff,
        dite_eq_right_iff, ENat.coe_ne_top, imp_false, not_false_eq_true, implies_true]


theorem FiniteMultiplicity.multiplicity_le_multiplicity_iff {c d : β} (hab : FiniteMultiplicity a b)
    (hcd : FiniteMultiplicity c d) :
    multiplicity a b ≤ multiplicity c d ↔ ∀ n : ℕ, a ^ n ∣ b → c ^ n ∣ d := by
  rw [← WithTop.coe_le_coe, ENat.some_eq_coe, ← hab.emultiplicity_eq_multiplicity,
    ← hcd.emultiplicity_eq_multiplicity]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Monoid α
    inst✝ : Monoid β
    a b : α
    c d : β
    hab : FiniteMultiplicity a b
    hcd : FiniteMultiplicity c d
    ⊢ Iff (LE.le (emultiplicity a b) (emultiplicity c d)) (∀ (n : Nat), Dvd.dvd (H …
  -/
  apply emultiplicity_le_emultiplicity_iff
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.multiplicity_le_multiplicity_iff :=
  FiniteMultiplicity.multiplicity_le_multiplicity_iff


theorem emultiplicity_eq_emultiplicity_iff {c d : β} :
    emultiplicity a b = emultiplicity c d ↔ ∀ n : ℕ, a ^ n ∣ b ↔ c ^ n ∣ d :=
  ⟨fun h n =>
    ⟨emultiplicity_le_emultiplicity_iff.1 h.le n, emultiplicity_le_emultiplicity_iff.1 h.ge n⟩,
    fun h => le_antisymm (emultiplicity_le_emultiplicity_iff.2 fun n => (h n).mp)
      (emultiplicity_le_emultiplicity_iff.2 fun n => (h n).mpr)⟩


theorem le_emultiplicity_map {F : Type*} [FunLike F α β] [MonoidHomClass F α β]
    (f : F) {a b : α} :
    emultiplicity a b ≤ emultiplicity (f a) (f b) :=
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    inst✝³ : Monoid α
                                                    inst✝² : Monoid β
                                                    F : Type u_3
                                                    inst✝¹ : FunLike F α β
                                                    inst✝ : MonoidHomClass F α β
                                                    f : F
                                                    a b : α
                                                    n : Nat
                                                    ⊢ Dvd.dvd (HPow.hPow a n) b → Dvd.dvd (HPow.hPow (f a) n) (f b)
                                                  -/
  emultiplicity_le_emultiplicity_iff.2 fun n ↦ by rw [← map_pow]; exact map_dvd f
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem emultiplicity_map_eq {F : Type*} [EquivLike F α β] [MulEquivClass F α β]
    (f : F) {a b : α} : emultiplicity (f a) (f b) = emultiplicity a b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Monoid α
    inst✝² : Monoid β
    F : Type u_3
    inst✝¹ : EquivLike F α β
    inst✝ : MulEquivClass F α β
    f : F
    a b : α
    ⊢ Eq (emultiplicity (f a) (f b)) (emultiplicity a b)
  -/
  simp [emultiplicity_eq_emultiplicity_iff, ← map_pow, map_dvd_iff]
  /-
    🎉 no goals
  -/


theorem multiplicity_map_eq {F : Type*} [EquivLike F α β] [MulEquivClass F α β]
    (f : F) {a b : α} : multiplicity (f a) (f b) = multiplicity a b :=
  multiplicity_eq_of_emultiplicity_eq (emultiplicity_map_eq f)


theorem emultiplicity_le_emultiplicity_of_dvd_right {a b c : α} (h : b ∣ c) :
    emultiplicity a b ≤ emultiplicity a c :=
  emultiplicity_le_emultiplicity_iff.2 fun _ hb => hb.trans h


theorem emultiplicity_eq_of_associated_right {a b c : α} (h : Associated b c) :
    emultiplicity a b = emultiplicity a c :=
  le_antisymm (emultiplicity_le_emultiplicity_of_dvd_right h.dvd)
    (emultiplicity_le_emultiplicity_of_dvd_right h.symm.dvd)


theorem multiplicity_eq_of_associated_right {a b c : α} (h : Associated b c) :
    multiplicity a b = multiplicity a c :=
  multiplicity_eq_of_emultiplicity_eq (emultiplicity_eq_of_associated_right h)


theorem dvd_of_emultiplicity_pos {a b : α} (h : 0 < emultiplicity a b) : a ∣ b :=
  pow_one a ▸ pow_dvd_of_le_emultiplicity (Order.add_one_le_of_lt h)


theorem dvd_of_multiplicity_pos {a b : α} (h : 0 < multiplicity a b) : a ∣ b :=
  dvd_of_emultiplicity_pos (lt_emultiplicity_of_lt_multiplicity h)


theorem dvd_iff_multiplicity_pos {a b : α} : 0 < multiplicity a b ↔ a ∣ b :=
                                                               /-
                                                                 α : Type u_1
                                                                 inst✝ : Monoid α
                                                                 a b : α
                                                                 hdvd : Dvd.dvd a b
                                                                 ⊢ Ne (multiplicity a b) 0
                                                               -/
  ⟨dvd_of_multiplicity_pos, fun hdvd => Nat.pos_of_ne_zero (by simpa [multiplicity_eq_zero])⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem dvd_iff_emultiplicity_pos {a b : α} : 0 < emultiplicity a b ↔ a ∣ b :=
  emultiplicity_pos_iff.trans dvd_iff_multiplicity_pos


theorem Nat.finiteMultiplicity_iff {a b : ℕ} : FiniteMultiplicity a b ↔ a ≠ 1 ∧ 0 < b := by
  rw [← not_iff_not, FiniteMultiplicity.not_iff_forall, not_and_or, not_ne_iff, not_lt,
    Nat.le_zero]
  exact
    ⟨fun h =>
      or_iff_not_imp_right.2 fun hb =>
        have ha : a ≠ 0 := fun ha => hb <| zero_dvd_iff.mp <| by rw [ha] at h; exact h 1
        Classical.by_contradiction fun ha1 : a ≠ 1 =>
          have ha_gt_one : 1 < a :=
            lt_of_not_ge fun _ =>
              match a with
              | 0 => ha rfl
              | 1 => ha1 rfl
              | b+2 => by omega
          not_lt_of_ge (le_of_dvd (Nat.pos_of_ne_zero hb) (h b)) (b.lt_pow_self ha_gt_one),
      fun h => by cases h <;> simp [*]⟩


@[deprecated (since := "2024-11-30")]
alias Nat.multiplicity_finite_iff := Nat.finiteMultiplicity_iff


alias ⟨_, Dvd.multiplicity_pos⟩ := dvd_iff_multiplicity_pos


theorem FiniteMultiplicity.mul_right {a b c : α} (hf : FiniteMultiplicity a (b * c)) :
    FiniteMultiplicity a c := (mul_comm b c ▸ hf).mul_left


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.mul_right := FiniteMultiplicity.mul_right


theorem emultiplicity_of_isUnit_right {a b : α} (ha : ¬IsUnit a)
    (hb : IsUnit b) : emultiplicity a b = 0 :=
  emultiplicity_eq_zero.mpr fun h ↦ ha (isUnit_of_dvd_unit h hb)


theorem multiplicity_of_isUnit_right {a b : α} (ha : ¬IsUnit a)
    (hb : IsUnit b) : multiplicity a b = 0 :=
  multiplicity_eq_zero.mpr fun h ↦ ha (isUnit_of_dvd_unit h hb)


theorem emultiplicity_of_one_right {a : α} (ha : ¬IsUnit a) : emultiplicity a 1 = 0 :=
  emultiplicity_of_isUnit_right ha isUnit_one


theorem multiplicity_of_one_right {a : α} (ha : ¬IsUnit a) : multiplicity a 1 = 0 :=
  multiplicity_of_isUnit_right ha isUnit_one


theorem emultiplicity_of_unit_right {a : α} (ha : ¬IsUnit a) (u : αˣ) : emultiplicity a u = 0 :=
  emultiplicity_of_isUnit_right ha u.isUnit


theorem multiplicity_of_unit_right {a : α} (ha : ¬IsUnit a) (u : αˣ) : multiplicity a u = 0 :=
  multiplicity_of_isUnit_right ha u.isUnit


theorem emultiplicity_le_emultiplicity_of_dvd_left {a b c : α} (hdvd : a ∣ b) :
    emultiplicity b c ≤ emultiplicity a c :=
  emultiplicity_le_emultiplicity_iff.2 fun n h => (pow_dvd_pow_of_dvd hdvd n).trans h


theorem emultiplicity_eq_of_associated_left {a b c : α} (h : Associated a b) :
    emultiplicity b c = emultiplicity a c :=
  le_antisymm (emultiplicity_le_emultiplicity_of_dvd_left h.dvd)
    (emultiplicity_le_emultiplicity_of_dvd_left h.symm.dvd)


theorem multiplicity_eq_of_associated_left {a b c : α} (h : Associated a b) :
    multiplicity b c = multiplicity a c :=
  multiplicity_eq_of_emultiplicity_eq (emultiplicity_eq_of_associated_left h)


theorem emultiplicity_mk_eq_emultiplicity {a b : α} :
    emultiplicity (Associates.mk a) (Associates.mk b) = emultiplicity a b := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    a b : α
    ⊢ Eq (emultiplicity (Associates.mk a) (Associates.mk b)) (emultiplicity a b)
  -/
  simp [emultiplicity_eq_emultiplicity_iff, ← Associates.mk_pow, Associates.mk_dvd_mk]
  /-
    🎉 no goals
  -/


theorem FiniteMultiplicity.ne_zero {a b : α} (h : FiniteMultiplicity a b) : b ≠ 0 :=
  let ⟨n, hn⟩ := h
               /-
                 α : Type u_1
                 inst✝ : MonoidWithZero α
                 a b : α
                 h : FiniteMultiplicity a b
                 n : Nat
                 hn : Not (Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1)) b)
                 hb : Eq b 0
                 ⊢ False
               -/
  fun hb => by simp [hb] at hn
               /-
                 🎉 no goals
               -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.ne_zero := FiniteMultiplicity.ne_zero


@[simp]
theorem emultiplicity_zero (a : α) : emultiplicity a 0 = ⊤ :=
  emultiplicity_eq_top.2 (fun v ↦ v.ne_zero rfl)


@[simp]
theorem emultiplicity_zero_eq_zero_of_ne_zero (a : α) (ha : a ≠ 0) : emultiplicity 0 a = 0 :=
  emultiplicity_eq_zero.2 <| mt zero_dvd_iff.1 ha


@[simp]
theorem multiplicity_zero_eq_zero_of_ne_zero (a : α) (ha : a ≠ 0) : multiplicity 0 a = 0 :=
  multiplicity_eq_zero.2 <| mt zero_dvd_iff.1 ha


theorem FiniteMultiplicity.or_of_add {p a b : α} (hf : FiniteMultiplicity p (a + b)) :
    FiniteMultiplicity p a ∨ FiniteMultiplicity p b := by
  /-
    α : Type u_1
    inst✝ : Semiring α
    p a b : α
    hf : FiniteMultiplicity p (HAdd.hAdd a b)
    ⊢ Or (FiniteMultiplicity p a) (FiniteMultiplicity p b)
  -/
  by_contra! nh
  /-
    α : Type u_1
    inst✝ : Semiring α
    p a b : α
    hf : FiniteMultiplicity p (HAdd.hAdd a b)
    nh : And (Not (FiniteMultiplicity p a)) (Not (FiniteMultiplicity p b))
    ⊢ False
  -/
  obtain ⟨c, hc⟩ := hf
  /-
    case intro
    α : Type u_1
    inst✝ : Semiring α
    p a b : α
    nh : And (Not (FiniteMultiplicity p a)) (Not (FiniteMultiplicity p b))
    c : Nat
    hc : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd c 1)) (HAdd.hAdd a b))
    ⊢ False
  -/
  simp_all [dvd_add]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.or_of_add := FiniteMultiplicity.or_of_add


theorem min_le_emultiplicity_add {p a b : α} :
    min (emultiplicity p a) (emultiplicity p b) ≤ emultiplicity p (a + b) := by
  /-
    α : Type u_1
    inst✝ : Semiring α
    p a b : α
    ⊢ LE.le (Min.min (emultiplicity p a) (emultiplicity p b)) (emultiplicity p (HA …
  -/
  cases hm : min (emultiplicity p a) (emultiplicity p b)
    /-
      case top
      α : Type u_1
      inst✝ : Semiring α
      p a b : α
      hm : Eq (Min.min (emultiplicity p a) (emultiplicity p b)) Top.top
      ⊢ LE.le Top.top (emultiplicity p (HAdd.hAdd a b))
    -/
  · simp only [top_le_iff, min_eq_top, emultiplicity_eq_top] at hm ⊢
    /-
      case top
      α : Type u_1
      inst✝ : Semiring α
      p a b : α
      hm : And (Not (FiniteMultiplicity p a)) (Not (FiniteMultiplicity p b))
      ⊢ Not (FiniteMultiplicity p (HAdd.hAdd a b))
    -/
    contrapose hm
    /-
      case top
      α : Type u_1
      inst✝ : Semiring α
      p a b : α
      hm : Not (Not (FiniteMultiplicity p (HAdd.hAdd a b)))
      ⊢ Not (And (Not (FiniteMultiplicity p a)) (Not (FiniteMultiplicity p b)))
    -/
    simp only [not_and_or, not_not] at hm ⊢
    /-
      case top
      α : Type u_1
      inst✝ : Semiring α
      p a b : α
      hm : FiniteMultiplicity p (HAdd.hAdd a b)
      ⊢ Or (FiniteMultiplicity p a) (FiniteMultiplicity p b)
    -/
    exact hm.or_of_add
    /-
      🎉 no goals
    -/
    /-
      case coe
      α : Type u_1
      inst✝ : Semiring α
      p a b : α
      a✝ : Nat
      hm : Eq (Min.min (emultiplicity p a) (emultiplicity p b)) ↑a✝
      ⊢ LE.le (↑a✝) (emultiplicity p (HAdd.hAdd a b))
    -/
  · apply le_emultiplicity_of_pow_dvd
    /-
      case coe.hk
      α : Type u_1
      inst✝ : Semiring α
      p a b : α
      a✝ : Nat
      hm : Eq (Min.min (emultiplicity p a) (emultiplicity p b)) ↑a✝
      ⊢ Dvd.dvd (HPow.hPow p a✝) (HAdd.hAdd a b)
    -/
    simp [dvd_add, pow_dvd_of_le_emultiplicity, ← hm]
    /-
      🎉 no goals
    -/


@[simp]
theorem FiniteMultiplicity.neg_iff {a b : α} :
    FiniteMultiplicity a (-b) ↔ FiniteMultiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Ring α
    a b : α
    ⊢ Iff (FiniteMultiplicity a (Neg.neg b)) (FiniteMultiplicity a b)
  -/
  unfold FiniteMultiplicity
  /-
    α : Type u_1
    inst✝ : Ring α
    a b : α
    ⊢ Iff (Exists fun n => Not (Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1)) (Neg.neg b)) …
  -/
  congr! 3
  /-
    case a.h.e'_2.h.h.e'_1.a
    α : Type u_1
    inst✝ : Ring α
    a b : α
    x✝ : Nat
    ⊢ Iff (Dvd.dvd (HPow.hPow a (HAdd.hAdd x✝ 1)) (Neg.neg b)) (Dvd.dvd (HPow.hPow …
  -/
  simp only [dvd_neg]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.neg_iff := FiniteMultiplicity.neg_iff


alias ⟨_, FiniteMultiplicity.neg⟩ := FiniteMultiplicity.neg_iff


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.neg := FiniteMultiplicity.neg


@[simp]
theorem emultiplicity_neg (a b : α) : emultiplicity a (-b) = emultiplicity a b := by
  /-
    α : Type u_1
    inst✝ : Ring α
    a b : α
    ⊢ Eq (emultiplicity a (Neg.neg b)) (emultiplicity a b)
  -/
  rw [emultiplicity_eq_emultiplicity_iff]
  /-
    α : Type u_1
    inst✝ : Ring α
    a b : α
    ⊢ ∀ (n : Nat), Iff (Dvd.dvd (HPow.hPow a n) (Neg.neg b)) (Dvd.dvd (HPow.hPow a …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem multiplicity_neg (a b : α) : multiplicity a (-b) = multiplicity a b :=
  multiplicity_eq_of_emultiplicity_eq (emultiplicity_neg a b)


theorem Int.emultiplicity_natAbs (a : ℕ) (b : ℤ) :
    emultiplicity a b.natAbs = emultiplicity (a : ℤ) b := by
  /-
    a : Nat
    b : Int
    ⊢ Eq (emultiplicity a b.natAbs) (emultiplicity (↑a) b)
  -/
  cases' Int.natAbs_eq b with h h <;> conv_rhs => rw [h]
    /-
      case inl
      a : Nat
      b : Int
      h : Eq b ↑b.natAbs
      ⊢ Eq (emultiplicity a b.natAbs) (emultiplicity ↑a ↑b.natAbs)
    -/
  · rw [Int.natCast_emultiplicity]
    /-
      🎉 no goals
    -/
    /-
      case inr
      a : Nat
      b : Int
      h : Eq b (Neg.neg ↑b.natAbs)
      ⊢ Eq (emultiplicity a b.natAbs) (emultiplicity (↑a) (Neg.neg ↑b.natAbs))
    -/
  · rw [emultiplicity_neg, Int.natCast_emultiplicity]
    /-
      🎉 no goals
    -/


theorem Int.multiplicity_natAbs (a : ℕ) (b : ℤ) :
    multiplicity a b.natAbs = multiplicity (a : ℤ) b :=
  multiplicity_eq_of_emultiplicity_eq (Int.emultiplicity_natAbs a b)


theorem emultiplicity_add_of_gt {p a b : α} (h : emultiplicity p b < emultiplicity p a) :
    emultiplicity p (a + b) = emultiplicity p b := by
  /-
    α : Type u_1
    inst✝ : Ring α
    p a b : α
    h : LT.lt (emultiplicity p b) (emultiplicity p a)
    ⊢ Eq (emultiplicity p (HAdd.hAdd a b)) (emultiplicity p b)
  -/
  have : FiniteMultiplicity p b := finiteMultiplicity_iff_emultiplicity_ne_top.2 (by simp [·] at h)
  /-
    α : Type u_1
    inst✝ : Ring α
    p a b : α
    h : LT.lt (emultiplicity p b) (emultiplicity p a)
    this : FiniteMultiplicity p b
    ⊢ Eq (emultiplicity p (HAdd.hAdd a b)) (emultiplicity p b)
  -/
  rw [this.emultiplicity_eq_multiplicity] at *
  /-
    α : Type u_1
    inst✝ : Ring α
    p a b : α
    h : LT.lt (↑(multiplicity p b)) (emultiplicity p a)
    this : FiniteMultiplicity p b
    ⊢ Eq (emultiplicity p (HAdd.hAdd a b)) ↑(multiplicity p b)
  -/
  apply emultiplicity_eq_of_dvd_of_not_dvd
    /-
      case hk
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      h : LT.lt (↑(multiplicity p b)) (emultiplicity p a)
      this : FiniteMultiplicity p b
      ⊢ Dvd.dvd (HPow.hPow p (multiplicity p b)) (HAdd.hAdd a b)
    -/
  · apply dvd_add
      /-
        case hk.h₁
        α : Type u_1
        inst✝ : Ring α
        p a b : α
        h : LT.lt (↑(multiplicity p b)) (emultiplicity p a)
        this : FiniteMultiplicity p b
        ⊢ Dvd.dvd (HPow.hPow p (multiplicity p b)) a
      -/
    · apply pow_dvd_of_le_emultiplicity
      /-
        case hk.h₁.hk
        α : Type u_1
        inst✝ : Ring α
        p a b : α
        h : LT.lt (↑(multiplicity p b)) (emultiplicity p a)
        this : FiniteMultiplicity p b
        ⊢ LE.le (↑(multiplicity p b)) (emultiplicity p a)
      -/
      exact h.le
      /-
        🎉 no goals
      -/
      /-
        case hk.h₂
        α : Type u_1
        inst✝ : Ring α
        p a b : α
        h : LT.lt (↑(multiplicity p b)) (emultiplicity p a)
        this : FiniteMultiplicity p b
        ⊢ Dvd.dvd (HPow.hPow p (multiplicity p b)) b
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case hsucc
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      h : LT.lt (↑(multiplicity p b)) (emultiplicity p a)
      this : FiniteMultiplicity p b
      ⊢ Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p b) 1)) (HAdd.hAdd a b))
    -/
  · rw [dvd_add_right]
      /-
        case hsucc
        α : Type u_1
        inst✝ : Ring α
        p a b : α
        h : LT.lt (↑(multiplicity p b)) (emultiplicity p a)
        this : FiniteMultiplicity p b
        ⊢ Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p b) 1)) b)
      -/
    · apply this.not_pow_dvd_of_multiplicity_lt
      /-
        case hsucc
        α : Type u_1
        inst✝ : Ring α
        p a b : α
        h : LT.lt (↑(multiplicity p b)) (emultiplicity p a)
        this : FiniteMultiplicity p b
        ⊢ LT.lt (multiplicity p b) (HAdd.hAdd (multiplicity p b) 1)
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case hsucc
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      h : LT.lt (↑(multiplicity p b)) (emultiplicity p a)
      this : FiniteMultiplicity p b
      ⊢ Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p b) 1)) a
    -/
    apply pow_dvd_of_le_emultiplicity
    /-
      case hsucc.hk
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      h : LT.lt (↑(multiplicity p b)) (emultiplicity p a)
      this : FiniteMultiplicity p b
      ⊢ LE.le (↑(HAdd.hAdd (multiplicity p b) 1)) (emultiplicity p a)
    -/
    exact Order.add_one_le_of_lt h
    /-
      🎉 no goals
    -/


theorem FiniteMultiplicity.multiplicity_add_of_gt {p a b : α} (hf : FiniteMultiplicity p b)
    (h : multiplicity p b < multiplicity p a) :
    multiplicity p (a + b) = multiplicity p b :=
  multiplicity_eq_of_emultiplicity_eq <| emultiplicity_add_of_gt (hf.emultiplicity_eq_multiplicity ▸
      (WithTop.coe_strictMono h).trans_le multiplicity_le_emultiplicity)


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.multiplicity_add_of_gt := FiniteMultiplicity.multiplicity_add_of_gt


theorem emultiplicity_sub_of_gt {p a b : α} (h : emultiplicity p b < emultiplicity p a) :
    emultiplicity p (a - b) = emultiplicity p b := by
  /-
    α : Type u_1
    inst✝ : Ring α
    p a b : α
    h : LT.lt (emultiplicity p b) (emultiplicity p a)
    ⊢ Eq (emultiplicity p (HSub.hSub a b)) (emultiplicity p b)
  -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  rw [sub_eq_add_neg, emultiplicity_add_of_gt] <;> rw [emultiplicity_neg]; assumption
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem multiplicity_sub_of_gt {p a b : α} (h : multiplicity p b < multiplicity p a)
    (hfin : FiniteMultiplicity p b) : multiplicity p (a - b) = multiplicity p b := by
  /-
    α : Type u_1
    inst✝ : Ring α
    p a b : α
    h : LT.lt (multiplicity p b) (multiplicity p a)
    hfin : FiniteMultiplicity p b
    ⊢ Eq (multiplicity p (HSub.hSub a b)) (multiplicity p b)
  -/
                                                           /-
                                                             🎉 no goals
                                                           -/
  rw [sub_eq_add_neg, hfin.neg.multiplicity_add_of_gt] <;> rw [multiplicity_neg]; assumption
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem emultiplicity_add_eq_min {p a b : α}
    (h : emultiplicity p a ≠ emultiplicity p b) :
    emultiplicity p (a + b) = min (emultiplicity p a) (emultiplicity p b) := by
  /-
    α : Type u_1
    inst✝ : Ring α
    p a b : α
    h : Ne (emultiplicity p a) (emultiplicity p b)
    ⊢ Eq (emultiplicity p (HAdd.hAdd a b)) (Min.min (emultiplicity p a) (emultipli …
  -/
  rcases lt_trichotomy (emultiplicity p a) (emultiplicity p b) with (hab | _ | hab)
    /-
      case inl
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      h : Ne (emultiplicity p a) (emultiplicity p b)
      hab : LT.lt (emultiplicity p a) (emultiplicity p b)
      ⊢ Eq (emultiplicity p (HAdd.hAdd a b)) (Min.min (emultiplicity p a) (emultipli …
    -/
  · rw [add_comm, emultiplicity_add_of_gt hab, min_eq_left]
    /-
      case inl
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      h : Ne (emultiplicity p a) (emultiplicity p b)
      hab : LT.lt (emultiplicity p a) (emultiplicity p b)
      ⊢ LE.le (emultiplicity p a) (emultiplicity p b)
    -/
    exact le_of_lt hab
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      h : Ne (emultiplicity p a) (emultiplicity p b)
      h✝ : Eq (emultiplicity p a) (emultiplicity p b)
      ⊢ Eq (emultiplicity p (HAdd.hAdd a b)) (Min.min (emultiplicity p a) (emultipli …
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      h : Ne (emultiplicity p a) (emultiplicity p b)
      hab : LT.lt (emultiplicity p b) (emultiplicity p a)
      ⊢ Eq (emultiplicity p (HAdd.hAdd a b)) (Min.min (emultiplicity p a) (emultipli …
    -/
  · rw [emultiplicity_add_of_gt hab, min_eq_right]
    /-
      case inr.inr
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      h : Ne (emultiplicity p a) (emultiplicity p b)
      hab : LT.lt (emultiplicity p b) (emultiplicity p a)
      ⊢ LE.le (emultiplicity p b) (emultiplicity p a)
    -/
    exact le_of_lt hab
    /-
      🎉 no goals
    -/


theorem multiplicity_add_eq_min {p a b : α} (ha : FiniteMultiplicity p a)
    (hb : FiniteMultiplicity p b) (h : multiplicity p a ≠ multiplicity p b) :
    multiplicity p (a + b) = min (multiplicity p a) (multiplicity p b) := by
  /-
    α : Type u_1
    inst✝ : Ring α
    p a b : α
    ha : FiniteMultiplicity p a
    hb : FiniteMultiplicity p b
    h : Ne (multiplicity p a) (multiplicity p b)
    ⊢ Eq (multiplicity p (HAdd.hAdd a b)) (Min.min (multiplicity p a) (multiplicit …
  -/
  rcases lt_trichotomy (multiplicity p a) (multiplicity p b) with (hab | _ | hab)
    /-
      case inl
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      ha : FiniteMultiplicity p a
      hb : FiniteMultiplicity p b
      h : Ne (multiplicity p a) (multiplicity p b)
      hab : LT.lt (multiplicity p a) (multiplicity p b)
      ⊢ Eq (multiplicity p (HAdd.hAdd a b)) (Min.min (multiplicity p a) (multiplicit …
    -/
  · rw [add_comm, ha.multiplicity_add_of_gt hab, min_eq_left]
    /-
      case inl
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      ha : FiniteMultiplicity p a
      hb : FiniteMultiplicity p b
      h : Ne (multiplicity p a) (multiplicity p b)
      hab : LT.lt (multiplicity p a) (multiplicity p b)
      ⊢ LE.le (multiplicity p a) (multiplicity p b)
    -/
    exact le_of_lt hab
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      ha : FiniteMultiplicity p a
      hb : FiniteMultiplicity p b
      h : Ne (multiplicity p a) (multiplicity p b)
      h✝ : Eq (multiplicity p a) (multiplicity p b)
      ⊢ Eq (multiplicity p (HAdd.hAdd a b)) (Min.min (multiplicity p a) (multiplicit …
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      ha : FiniteMultiplicity p a
      hb : FiniteMultiplicity p b
      h : Ne (multiplicity p a) (multiplicity p b)
      hab : LT.lt (multiplicity p b) (multiplicity p a)
      ⊢ Eq (multiplicity p (HAdd.hAdd a b)) (Min.min (multiplicity p a) (multiplicit …
    -/
  · rw [hb.multiplicity_add_of_gt hab, min_eq_right]
    /-
      case inr.inr
      α : Type u_1
      inst✝ : Ring α
      p a b : α
      ha : FiniteMultiplicity p a
      hb : FiniteMultiplicity p b
      h : Ne (multiplicity p a) (multiplicity p b)
      hab : LT.lt (multiplicity p b) (multiplicity p a)
      ⊢ LE.le (multiplicity p b) (multiplicity p a)
    -/
    exact le_of_lt hab
    /-
      🎉 no goals
    -/


theorem finiteMultiplicity_mul_aux {p : α} (hp : Prime p) {a b : α} :
    ∀ {n m : ℕ}, ¬p ^ (n + 1) ∣ a → ¬p ^ (m + 1) ∣ b → ¬p ^ (n + m + 1) ∣ a * b
  | n, m => fun ha hb ⟨s, hs⟩ =>
                                             /-
                                               α : Type u_1
                                               inst✝ : CancelCommMonoidWithZero α
                                               p : α
                                               hp : Prime p
                                               a b : α
                                               n m : Nat
                                               ha : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) a)
                                               hb : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd m 1)) b)
                                               x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) (HMul.hMul a b)
                                               s : α
                                               hs : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) …
                                               ⊢ Eq (HMul.hMul a b) (HMul.hMul p (HMul.hMul (HPow.hPow p (HAdd.hAdd n m)) s))
                                             -/
    have : p ∣ a * b := ⟨p ^ (n + m) * s, by simp [hs, pow_add, mul_comm, mul_assoc, mul_left_comm]⟩
                                             /-
                                               🎉 no goals
                                             -/
    (hp.2.2 a b this).elim
      (fun ⟨x, hx⟩ =>
        have hn0 : 0 < n :=
                                           /-
                                             α : Type u_1
                                             inst✝ : CancelCommMonoidWithZero α
                                             p : α
                                             hp : Prime p
                                             a b : α
                                             n m : Nat
                                             ha : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) a)
                                             hb : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd m 1)) b)
                                             x✝¹ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) (HMul.hMul a b)
                                             s : α
                                             hs : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) …
                                             this : Dvd.dvd p (HMul.hMul a b)
                                             x✝ : Dvd.dvd p a
                                             x : α
                                             hx : Eq a (HMul.hMul p x)
                                             hn0 : Eq n 0
                                             ⊢ False
                                           -/
          Nat.pos_of_ne_zero fun hn0 => by simp [hx, hn0] at ha
                                           /-
                                             🎉 no goals
                                           -/
        have hpx : ¬p ^ (n - 1 + 1) ∣ x := fun ⟨y, hy⟩ =>
          ha (hx.symm ▸ ⟨y, mul_right_cancel₀ hp.1 <| by
            /-
              α : Type u_1
              inst✝ : CancelCommMonoidWithZero α
              p : α
              hp : Prime p
              a b : α
              n m : Nat
              ha : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) a)
              hb : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd m 1)) b)
              x✝² : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) (HMul.hMul a b)
              s : α
              hs : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) …
              this : Dvd.dvd p (HMul.hMul a b)
              x✝¹ : Dvd.dvd p a
              x : α
              hx : Eq a (HMul.hMul p x)
              hn0 : LT.lt 0 n
              x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HSub.hSub n 1) 1)) x
              y : α
              hy : Eq x (HMul.hMul (HPow.hPow p (HAdd.hAdd (HSub.hSub n 1) 1)) y)
              ⊢ Eq (HMul.hMul (HMul.hMul p x) p) (HMul.hMul (HMul.hMul (HPow.hPow p (HAdd.hA …
            -/
            rw [tsub_add_cancel_of_le (succ_le_of_lt hn0)] at hy
            /-
              α : Type u_1
              inst✝ : CancelCommMonoidWithZero α
              p : α
              hp : Prime p
              a b : α
              n m : Nat
              ha : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) a)
              hb : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd m 1)) b)
              x✝² : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) (HMul.hMul a b)
              s : α
              hs : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) …
              this : Dvd.dvd p (HMul.hMul a b)
              x✝¹ : Dvd.dvd p a
              x : α
              hx : Eq a (HMul.hMul p x)
              hn0 : LT.lt 0 n
              x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HSub.hSub n 1) 1)) x
              y : α
              hy : Eq x (HMul.hMul (HPow.hPow p n) y)
              ⊢ Eq (HMul.hMul (HMul.hMul p x) p) (HMul.hMul (HMul.hMul (HPow.hPow p (HAdd.hA …
            -/
            simp [hy, pow_add, mul_comm, mul_assoc, mul_left_comm]⟩)
            /-
              🎉 no goals
            -/
        have : 1 ≤ n + m := le_trans hn0 (Nat.le_add_right n m)
        finiteMultiplicity_mul_aux hp hpx hb
          ⟨s, mul_right_cancel₀ hp.1 (by
                /-
                  α : Type u_1
                  inst✝ : CancelCommMonoidWithZero α
                  p : α
                  hp : Prime p
                  a b : α
                  n m : Nat
                  ha : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) a)
                  hb : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd m 1)) b)
                  x✝¹ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) (HMul.hMul a b)
                  s : α
                  hs : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) …
                  this✝ : Dvd.dvd p (HMul.hMul a b)
                  x✝ : Dvd.dvd p a
                  x : α
                  hx : Eq a (HMul.hMul p x)
                  hn0 : LT.lt 0 n
                  hpx : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (HSub.hSub n 1) 1)) x)
                  this : LE.le 1 (HAdd.hAdd n m)
                  ⊢ Eq (HMul.hMul (HMul.hMul x b) p) (HMul.hMul (HMul.hMul (HPow.hPow p (HAdd.hA …
                -/
                rw [tsub_add_eq_add_tsub (succ_le_of_lt hn0), tsub_add_cancel_of_le this]
                /-
                  α : Type u_1
                  inst✝ : CancelCommMonoidWithZero α
                  p : α
                  hp : Prime p
                  a b : α
                  n m : Nat
                  ha : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) a)
                  hb : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd m 1)) b)
                  x✝¹ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) (HMul.hMul a b)
                  s : α
                  hs : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) …
                  this✝ : Dvd.dvd p (HMul.hMul a b)
                  x✝ : Dvd.dvd p a
                  x : α
                  hx : Eq a (HMul.hMul p x)
                  hn0 : LT.lt 0 n
                  hpx : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (HSub.hSub n 1) 1)) x)
                  this : LE.le 1 (HAdd.hAdd n m)
                  ⊢ Eq (HMul.hMul (HMul.hMul x b) p) (HMul.hMul (HMul.hMul (HPow.hPow p (HAdd.hA …
                -/
                simp_all [mul_comm, mul_assoc, mul_left_comm, pow_add])⟩)
                /-
                  🎉 no goals
                -/
      fun ⟨x, hx⟩ =>
        have hm0 : 0 < m :=
                                           /-
                                             α : Type u_1
                                             inst✝ : CancelCommMonoidWithZero α
                                             p : α
                                             hp : Prime p
                                             a b : α
                                             n m : Nat
                                             ha : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) a)
                                             hb : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd m 1)) b)
                                             x✝¹ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) (HMul.hMul a b)
                                             s : α
                                             hs : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) …
                                             this : Dvd.dvd p (HMul.hMul a b)
                                             x✝ : Dvd.dvd p b
                                             x : α
                                             hx : Eq b (HMul.hMul p x)
                                             hm0 : Eq m 0
                                             ⊢ False
                                           -/
          Nat.pos_of_ne_zero fun hm0 => by simp [hx, hm0] at hb
                                           /-
                                             🎉 no goals
                                           -/
        have hpx : ¬p ^ (m - 1 + 1) ∣ x := fun ⟨y, hy⟩ =>
          hb
            (hx.symm ▸
              ⟨y,
                mul_right_cancel₀ hp.1 <| by
                  /-
                    α : Type u_1
                    inst✝ : CancelCommMonoidWithZero α
                    p : α
                    hp : Prime p
                    a b : α
                    n m : Nat
                    ha : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) a)
                    hb : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd m 1)) b)
                    x✝² : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) (HMul.hMul a b)
                    s : α
                    hs : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) …
                    this : Dvd.dvd p (HMul.hMul a b)
                    x✝¹ : Dvd.dvd p b
                    x : α
                    hx : Eq b (HMul.hMul p x)
                    hm0 : LT.lt 0 m
                    x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HSub.hSub m 1) 1)) x
                    y : α
                    hy : Eq x (HMul.hMul (HPow.hPow p (HAdd.hAdd (HSub.hSub m 1) 1)) y)
                    ⊢ Eq (HMul.hMul (HMul.hMul p x) p) (HMul.hMul (HMul.hMul (HPow.hPow p (HAdd.hA …
                  -/
                  rw [tsub_add_cancel_of_le (succ_le_of_lt hm0)] at hy
                  /-
                    α : Type u_1
                    inst✝ : CancelCommMonoidWithZero α
                    p : α
                    hp : Prime p
                    a b : α
                    n m : Nat
                    ha : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) a)
                    hb : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd m 1)) b)
                    x✝² : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) (HMul.hMul a b)
                    s : α
                    hs : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) …
                    this : Dvd.dvd p (HMul.hMul a b)
                    x✝¹ : Dvd.dvd p b
                    x : α
                    hx : Eq b (HMul.hMul p x)
                    hm0 : LT.lt 0 m
                    x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HSub.hSub m 1) 1)) x
                    y : α
                    hy : Eq x (HMul.hMul (HPow.hPow p m) y)
                    ⊢ Eq (HMul.hMul (HMul.hMul p x) p) (HMul.hMul (HMul.hMul (HPow.hPow p (HAdd.hA …
                  -/
                  simp [hy, pow_add, mul_comm, mul_assoc, mul_left_comm]⟩)
                  /-
                    🎉 no goals
                  -/
        finiteMultiplicity_mul_aux hp ha hpx
        ⟨s, mul_right_cancel₀ hp.1 (by
              /-
                α : Type u_1
                inst✝ : CancelCommMonoidWithZero α
                p : α
                hp : Prime p
                a b : α
                n m : Nat
                ha : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) a)
                hb : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd m 1)) b)
                x✝¹ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) (HMul.hMul a b)
                s : α
                hs : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) …
                this : Dvd.dvd p (HMul.hMul a b)
                x✝ : Dvd.dvd p b
                x : α
                hx : Eq b (HMul.hMul p x)
                hm0 : LT.lt 0 m
                hpx : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (HSub.hSub m 1) 1)) x)
                ⊢ Eq (HMul.hMul (HMul.hMul a x) p) (HMul.hMul (HMul.hMul (HPow.hPow p (HAdd.hA …
              -/
              rw [add_assoc, tsub_add_cancel_of_le (succ_le_of_lt hm0)]
              /-
                α : Type u_1
                inst✝ : CancelCommMonoidWithZero α
                p : α
                hp : Prime p
                a b : α
                n m : Nat
                ha : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) a)
                hb : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd m 1)) b)
                x✝¹ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) (HMul.hMul a b)
                s : α
                hs : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n m) 1)) …
                this : Dvd.dvd p (HMul.hMul a b)
                x✝ : Dvd.dvd p b
                x : α
                hx : Eq b (HMul.hMul p x)
                hm0 : LT.lt 0 m
                hpx : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (HSub.hSub m 1) 1)) x)
                ⊢ Eq (HMul.hMul (HMul.hMul a x) p) (HMul.hMul (HMul.hMul (HPow.hPow p (HAdd.hA …
              -/
              simp_all [mul_comm, mul_assoc, mul_left_comm, pow_add])⟩
              /-
                🎉 no goals
              -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.finite_mul_aux := finiteMultiplicity_mul_aux


theorem Prime.finiteMultiplicity_mul {p a b : α} (hp : Prime p) :
    FiniteMultiplicity p a → FiniteMultiplicity p b → FiniteMultiplicity p (a * b) :=
  fun ⟨n, hn⟩ ⟨m, hm⟩ => ⟨n + m, finiteMultiplicity_mul_aux hp hn hm⟩


@[deprecated (since := "2024-11-30")]
alias Prime.multiplicity_finite_mul := Prime.finiteMultiplicity_mul


theorem FiniteMultiplicity.mul_iff {p a b : α} (hp : Prime p) :
    FiniteMultiplicity p (a * b) ↔ FiniteMultiplicity p a ∧ FiniteMultiplicity p b :=
  ⟨fun h => ⟨h.mul_left, h.mul_right⟩, fun h =>
    hp.finiteMultiplicity_mul h.1 h.2⟩


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.mul_iff := FiniteMultiplicity.mul_iff


theorem FiniteMultiplicity.pow {p a : α} (hp : Prime p)
    (hfin : FiniteMultiplicity p a) {k : ℕ} : FiniteMultiplicity p (a ^ k) :=
  match k, hfin with
                   /-
                     α : Type u_1
                     inst✝ : CancelCommMonoidWithZero α
                     p a : α
                     hp : Prime p
                     hfin : FiniteMultiplicity p a
                     k : Nat
                     x✝ : FiniteMultiplicity p a
                     ⊢ Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd 0 1)) (HPow.hPow a 0))
                   -/
  | 0, _ => ⟨0, by simp [mt isUnit_iff_dvd_one.2 hp.2.1]⟩
                   /-
                     🎉 no goals
                   -/
                    /-
                      α : Type u_1
                      inst✝ : CancelCommMonoidWithZero α
                      p a : α
                      hp : Prime p
                      hfin : FiniteMultiplicity p a
                      k✝ k : Nat
                      ha : FiniteMultiplicity p a
                      ⊢ FiniteMultiplicity p (HPow.hPow a (HAdd.hAdd k 1))
                    -/
  | k + 1, ha => by rw [_root_.pow_succ']; exact hp.finiteMultiplicity_mul ha (ha.pow hp)
                                           /-
                                             🎉 no goals
                                           -/


@[deprecated (since := "2024-11-30")] alias multiplicity.Finite.pow := FiniteMultiplicity.pow


@[simp]
theorem multiplicity_self {a : α} : multiplicity a a = 1 := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    a : α
    ⊢ Eq (multiplicity a a) 1
  -/
  by_cases ha : FiniteMultiplicity a a
    /-
      case pos
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      ha : FiniteMultiplicity a a
      ⊢ Eq (multiplicity a a) 1
    -/
  · rw [ha.multiplicity_eq_iff]
    /-
      case pos
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      ha : FiniteMultiplicity a a
      ⊢ And (Dvd.dvd (HPow.hPow a 1) a) (Not (Dvd.dvd (HPow.hPow a (HAdd.hAdd 1 1))  …
    -/
    simp only [pow_one, dvd_refl, reduceAdd, true_and]
    /-
      case pos
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      ha : FiniteMultiplicity a a
      ⊢ Not (Dvd.dvd (HPow.hPow a 2) a)
    -/
    rintro ⟨v, hv⟩
    /-
      case pos.intro
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      ha : FiniteMultiplicity a a
      v : α
      hv : Eq a (HMul.hMul (HPow.hPow a 2) v)
      ⊢ False
    -/
    nth_rw 1 [← mul_one a] at hv
    /-
      case pos.intro
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      ha : FiniteMultiplicity a a
      v : α
      hv : Eq (HMul.hMul a 1) (HMul.hMul (HPow.hPow a 2) v)
      ⊢ False
    -/
    simp only [sq, mul_assoc, mul_eq_mul_left_iff] at hv
    /-
      case pos.intro
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      ha : FiniteMultiplicity a a
      v : α
      hv : Or (Eq 1 (HMul.hMul a v)) (Eq a 0)
      ⊢ False
    -/
    obtain hv | rfl := hv
      /-
        case pos.intro.inl
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        a : α
        ha : FiniteMultiplicity a a
        v : α
        hv : Eq 1 (HMul.hMul a v)
        ⊢ False
      -/
    · have : IsUnit a := isUnit_of_mul_eq_one a v hv.symm
      /-
        case pos.intro.inl
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        a : α
        ha : FiniteMultiplicity a a
        v : α
        hv : Eq 1 (HMul.hMul a v)
        this : IsUnit a
        ⊢ False
      -/
      simpa [this] using ha.not_unit
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.inr
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        v : α
        ha : FiniteMultiplicity 0 0
        ⊢ False
      -/
    · simpa using ha.ne_zero
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      ha : Not (FiniteMultiplicity a a)
      ⊢ Eq (multiplicity a a) 1
    -/
  · simp [ha]
    /-
      🎉 no goals
    -/


@[simp]
theorem FiniteMultiplicity.emultiplicity_self {a : α} (hfin : FiniteMultiplicity a a) :
    emultiplicity a a = 1 := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    a : α
    hfin : FiniteMultiplicity a a
    ⊢ Eq (emultiplicity a a) 1
  -/
  simp [hfin.emultiplicity_eq_multiplicity]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.emultiplicity_self := FiniteMultiplicity.emultiplicity_self


theorem multiplicity_mul {p a b : α} (hp : Prime p) (hfin : FiniteMultiplicity p (a * b)) :
    multiplicity p (a * b) = multiplicity p a + multiplicity p b := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    p a b : α
    hp : Prime p
    hfin : FiniteMultiplicity p (HMul.hMul a b)
    ⊢ Eq (multiplicity p (HMul.hMul a b)) (HAdd.hAdd (multiplicity p a) (multiplic …
  -/
  have hdiva : p ^ multiplicity p a ∣ a := pow_multiplicity_dvd ..
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    p a b : α
    hp : Prime p
    hfin : FiniteMultiplicity p (HMul.hMul a b)
    hdiva : Dvd.dvd (HPow.hPow p (multiplicity p a)) a
    ⊢ Eq (multiplicity p (HMul.hMul a b)) (HAdd.hAdd (multiplicity p a) (multiplic …
  -/
  have hdivb : p ^ multiplicity p b ∣ b := pow_multiplicity_dvd ..
  have hdiv : p ^ (multiplicity p a + multiplicity p b) ∣ a * b := by
    rw [pow_add]; apply mul_dvd_mul <;> assumption
  have hsucc : ¬p ^ (multiplicity p a + multiplicity p b + 1) ∣ a * b :=
    fun h =>
    not_or_intro (hfin.mul_left.not_pow_dvd_of_multiplicity_lt (lt_succ_self _))
      (hfin.mul_right.not_pow_dvd_of_multiplicity_lt (lt_succ_self _))
      (_root_.succ_dvd_or_succ_dvd_of_succ_sum_dvd_mul hp hdiva hdivb h)
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    p a b : α
    hp : Prime p
    hfin : FiniteMultiplicity p (HMul.hMul a b)
    hdiva : Dvd.dvd (HPow.hPow p (multiplicity p a)) a
    hdivb : Dvd.dvd (HPow.hPow p (multiplicity p b)) b
    hdiv : Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p a) (multiplicity p b))) …
    hsucc : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd (multiplicity p a) (mu …
    ⊢ Eq (multiplicity p (HMul.hMul a b)) (HAdd.hAdd (multiplicity p a) (multiplic …
  -/
  rw [hfin.multiplicity_eq_iff]
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    p a b : α
    hp : Prime p
    hfin : FiniteMultiplicity p (HMul.hMul a b)
    hdiva : Dvd.dvd (HPow.hPow p (multiplicity p a)) a
    hdivb : Dvd.dvd (HPow.hPow p (multiplicity p b)) b
    hdiv : Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p a) (multiplicity p b))) …
    hsucc : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd (multiplicity p a) (mu …
    ⊢ And (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p a) (multiplicity p b))) …
  -/
  exact ⟨hdiv, hsucc⟩
  /-
    🎉 no goals
  -/


theorem emultiplicity_mul {p a b : α} (hp : Prime p) :
    emultiplicity p (a * b) = emultiplicity p a + emultiplicity p b := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    p a b : α
    hp : Prime p
    ⊢ Eq (emultiplicity p (HMul.hMul a b)) (HAdd.hAdd (emultiplicity p a) (emultip …
  -/
  by_cases hfin : FiniteMultiplicity p (a * b)
  · rw [hfin.emultiplicity_eq_multiplicity, hfin.mul_left.emultiplicity_eq_multiplicity,
      hfin.mul_right.emultiplicity_eq_multiplicity]
    /-
      case pos
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      p a b : α
      hp : Prime p
      hfin : FiniteMultiplicity p (HMul.hMul a b)
      ⊢ Eq (↑(multiplicity p (HMul.hMul a b))) (HAdd.hAdd ↑(multiplicity p a) ↑(mult …
    -/
    norm_cast
    /-
      case pos
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      p a b : α
      hp : Prime p
      hfin : FiniteMultiplicity p (HMul.hMul a b)
      ⊢ Eq (multiplicity p (HMul.hMul a b)) (HAdd.hAdd (multiplicity p a) (multiplic …
    -/
    exact multiplicity_mul hp hfin
    /-
      🎉 no goals
    -/
  · rw [emultiplicity_eq_top.2 hfin, eq_comm, WithTop.add_eq_top, emultiplicity_eq_top,
      emultiplicity_eq_top]
    /-
      case neg
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      p a b : α
      hp : Prime p
      hfin : Not (FiniteMultiplicity p (HMul.hMul a b))
      ⊢ Or (Not (FiniteMultiplicity p a)) (Not (FiniteMultiplicity p b))
    -/
    simpa only [FiniteMultiplicity.mul_iff hp, not_and_or] using hfin
    /-
      🎉 no goals
    -/


theorem Finset.emultiplicity_prod {β : Type*} {p : α} (hp : Prime p) (s : Finset β) (f : β → α) :
    emultiplicity p (∏ x ∈ s, f x) = ∑ x ∈ s, emultiplicity p (f x) := by classical
    induction' s using Finset.induction with a s has ih h
    · simp only [Finset.sum_empty, Finset.prod_empty]
      exact emultiplicity_of_one_right hp.not_unit
    · simpa [has, ← ih] using emultiplicity_mul hp


theorem emultiplicity_pow {p a : α} (hp : Prime p) {k : ℕ} :
    emultiplicity p (a ^ k) = k * emultiplicity p a := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    p a : α
    hp : Prime p
    k : Nat
    ⊢ Eq (emultiplicity p (HPow.hPow a k)) (HMul.hMul (↑k) (emultiplicity p a))
  -/
  induction' k with k hk
    /-
      case zero
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      p a : α
      hp : Prime p
      ⊢ Eq (emultiplicity p (HPow.hPow a 0)) (HMul.hMul (↑0) (emultiplicity p a))
    -/
  · simp [emultiplicity_of_one_right hp.not_unit]
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      p a : α
      hp : Prime p
      k : Nat
      hk : Eq (emultiplicity p (HPow.hPow a k)) (HMul.hMul (↑k) (emultiplicity p a))
      ⊢ Eq (emultiplicity p (HPow.hPow a (HAdd.hAdd k 1))) (HMul.hMul (↑(HAdd.hAdd k …
    -/
  · simp [pow_succ, emultiplicity_mul hp, hk, add_mul]
    /-
      🎉 no goals
    -/


protected theorem FiniteMultiplicity.multiplicity_pow {p a : α} (hp : Prime p)
    (ha : FiniteMultiplicity p a) {k : ℕ} : multiplicity p (a ^ k) = k * multiplicity p a := by
  exact_mod_cast (ha.pow hp).emultiplicity_eq_multiplicity ▸
    ha.emultiplicity_eq_multiplicity ▸ emultiplicity_pow hp


@[deprecated (since := "2024-11-30")]
alias multiplicity.Finite.multiplicity_pow := FiniteMultiplicity.multiplicity_pow


theorem emultiplicity_pow_self {p : α} (h0 : p ≠ 0) (hu : ¬IsUnit p) (n : ℕ) :
    emultiplicity p (p ^ n) = n := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    p : α
    h0 : Ne p 0
    hu : Not (IsUnit p)
    n : Nat
    ⊢ Eq (emultiplicity p (HPow.hPow p n)) ↑n
  -/
  apply emultiplicity_eq_of_dvd_of_not_dvd
    /-
      case hk
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      p : α
      h0 : Ne p 0
      hu : Not (IsUnit p)
      n : Nat
      ⊢ Dvd.dvd (HPow.hPow p n) (HPow.hPow p n)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case hsucc
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      p : α
      h0 : Ne p 0
      hu : Not (IsUnit p)
      n : Nat
      ⊢ Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) (HPow.hPow p n))
    -/
  · rw [pow_dvd_pow_iff h0 hu]
    /-
      case hsucc
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      p : α
      h0 : Ne p 0
      hu : Not (IsUnit p)
      n : Nat
      ⊢ Not (LE.le (HAdd.hAdd n 1) n)
    -/
    apply Nat.not_succ_le_self
    /-
      🎉 no goals
    -/


theorem multiplicity_pow_self {p : α} (h0 : p ≠ 0) (hu : ¬IsUnit p) (n : ℕ) :
    multiplicity p (p ^ n) = n :=
  multiplicity_eq_of_emultiplicity_eq_some (emultiplicity_pow_self h0 hu n)


theorem emultiplicity_pow_self_of_prime {p : α} (hp : Prime p) (n : ℕ) :
    emultiplicity p (p ^ n) = n :=
  emultiplicity_pow_self hp.ne_zero hp.not_unit n


theorem multiplicity_pow_self_of_prime {p : α} (hp : Prime p) (n : ℕ) :
    multiplicity p (p ^ n) = n :=
  multiplicity_pow_self hp.ne_zero hp.not_unit n


theorem multiplicity_eq_zero_of_coprime {p a b : ℕ} (hp : p ≠ 1)
    (hle : multiplicity p a ≤ multiplicity p b) (hab : Nat.Coprime a b) : multiplicity p a = 0 := by
  /-
    p a b : Nat
    hp : Ne p 1
    hle : LE.le (multiplicity p a) (multiplicity p b)
    hab : a.Coprime b
    ⊢ Eq (multiplicity p a) 0
  -/
  apply Nat.eq_zero_of_not_pos
  /-
    case h
    p a b : Nat
    hp : Ne p 1
    hle : LE.le (multiplicity p a) (multiplicity p b)
    hab : a.Coprime b
    ⊢ Not (LT.lt 0 (multiplicity p a))
  -/
  intro nh
  /-
    case h
    p a b : Nat
    hp : Ne p 1
    hle : LE.le (multiplicity p a) (multiplicity p b)
    hab : a.Coprime b
    nh : LT.lt 0 (multiplicity p a)
    ⊢ False
  -/
  have da : p ∣ a := by simpa [multiplicity_eq_zero] using nh.ne.symm
  /-
    case h
    p a b : Nat
    hp : Ne p 1
    hle : LE.le (multiplicity p a) (multiplicity p b)
    hab : a.Coprime b
    nh : LT.lt 0 (multiplicity p a)
    da : Dvd.dvd p a
    ⊢ False
  -/
  have db : p ∣ b := by simpa [multiplicity_eq_zero] using (nh.trans_le hle).ne.symm
  /-
    case h
    p a b : Nat
    hp : Ne p 1
    hle : LE.le (multiplicity p a) (multiplicity p b)
    hab : a.Coprime b
    nh : LT.lt 0 (multiplicity p a)
    da : Dvd.dvd p a
    db : Dvd.dvd p b
    ⊢ False
  -/
  have := Nat.dvd_gcd da db
  /-
    case h
    p a b : Nat
    hp : Ne p 1
    hle : LE.le (multiplicity p a) (multiplicity p b)
    hab : a.Coprime b
    nh : LT.lt 0 (multiplicity p a)
    da : Dvd.dvd p a
    db : Dvd.dvd p b
    this : Dvd.dvd p (a.gcd b)
    ⊢ False
  -/
  rw [Coprime.gcd_eq_one hab, Nat.dvd_one] at this
  /-
    case h
    p a b : Nat
    hp : Ne p 1
    hle : LE.le (multiplicity p a) (multiplicity p b)
    hab : a.Coprime b
    nh : LT.lt 0 (multiplicity p a)
    da : Dvd.dvd p a
    db : Dvd.dvd p b
    this : Eq p 1
    ⊢ False
  -/
  exact hp this
  /-
    🎉 no goals
  -/


theorem Int.finiteMultiplicity_iff_finiteMultiplicity_natAbs {a b : ℤ} :
    FiniteMultiplicity a b ↔ FiniteMultiplicity a.natAbs b.natAbs := by
  /-
    a b : Int
    ⊢ Iff (FiniteMultiplicity a b) (FiniteMultiplicity a.natAbs b.natAbs)
  -/
  simp only [FiniteMultiplicity.def, ← Int.natAbs_dvd_natAbs, Int.natAbs_pow]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias Int.multiplicity_finite_iff_natAbs_finite :=
  Int.finiteMultiplicity_iff_finiteMultiplicity_natAbs


theorem Int.finiteMultiplicity_iff {a b : ℤ} : FiniteMultiplicity a b ↔ a.natAbs ≠ 1 ∧ b ≠ 0 := by
  rw [finiteMultiplicity_iff_finiteMultiplicity_natAbs, Nat.finiteMultiplicity_iff,
    pos_iff_ne_zero, Int.natAbs_ne_zero]


@[deprecated (since := "2024-11-30")]
alias Int.multiplicity_finite_iff := Int.finiteMultiplicity_iff


instance Nat.decidableFiniteMultiplicity : DecidableRel fun a b : ℕ => FiniteMultiplicity a b :=
  fun _ _ ↦ decidable_of_iff' _ Nat.finiteMultiplicity_iff


@[deprecated (since := "2024-11-30")]
alias Nat.decidableMultiplicityFinite := Nat.decidableFiniteMultiplicity


instance Int.decidableMultiplicityFinite : DecidableRel fun a b : ℤ => FiniteMultiplicity a b :=
  fun _ _ ↦ decidable_of_iff' _ Int.finiteMultiplicity_iff


@[deprecated (since := "2024-11-30")]
alias Int.decidableFiniteMultiplicity := Int.decidableMultiplicityFinite

