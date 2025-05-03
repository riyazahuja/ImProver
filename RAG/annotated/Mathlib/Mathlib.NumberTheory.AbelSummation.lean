private theorem sumlocc (n : ℕ) :
    ∀ᵐ t, t ∈ Set.Icc (n : ℝ) (n + 1) → ∑ k ∈ Icc 0 ⌊t⌋₊, c k = ∑ k ∈ Icc 0 n, c k := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    n : Nat
    ⊢ Filter.Eventually (fun t => Membership.mem (Set.Icc (↑n) (HAdd.hAdd (↑n) 1)) …
  -/
  filter_upwards [Ico_ae_eq_Icc] with t h ht
  /-
    case h
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    n : Nat
    t : Real
    h : Eq (Set.Ico ?m.1757 ?m.1758 t) (Set.Icc ?m.1757 ?m.1758 t)
    ht : Membership.mem (Set.Icc (↑n) (HAdd.hAdd (↑n) 1)) t
    ⊢ Eq ((Finset.Icc 0 (Nat.floor t)).sum fun k => c k) ((Finset.Icc 0 n).sum fun …
  -/
  rw [Nat.floor_eq_on_Ico _ _ (h.mpr ht)]
  /-
    🎉 no goals
  -/


private theorem integralmulsum (hf_diff : ∀ t ∈ Set.Icc a b, DifferentiableAt ℝ f t)
              /-
                𝕜 : Type u_1
                inst✝ : RCLike 𝕜
                c : Nat → 𝕜
                f : Real → 𝕜
                a b : Real
                hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
                ⊢ MeasureTheory.Measure Real
              -/
    (hf_int : IntegrableOn (deriv f) (Set.Icc a b)) (t₁ t₂ : ℝ) (n : ℕ) (h : t₁ ≤ t₂)
              /-
                🎉 no goals
              -/
    (h₁ : n ≤ t₁) (h₂ : t₂ ≤ n + 1) (h₃ : a ≤ t₁) (h₄ : t₂ ≤ b) :
    ∫ t in t₁..t₂, deriv f t * ∑ k ∈ Icc 0 ⌊t⌋₊, c k =
      (f t₂ - f t₁) * ∑ k ∈ Icc 0 n, c k := by
  have h_inc₁ : Ι t₁ t₂ ⊆ Set.Icc n (n + 1) :=
    Set.uIoc_of_le h ▸ Set.Ioc_subset_Icc_self.trans <| Set.Icc_subset_Icc h₁ h₂
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    t₁ t₂ : Real
    n : Nat
    h : LE.le t₁ t₂
    h₁ : LE.le (↑n) t₁
    h₂ : LE.le t₂ (HAdd.hAdd (↑n) 1)
    h₃ : LE.le a t₁
    h₄ : LE.le t₂ b
    h_inc₁ : HasSubset.Subset (Set.uIoc t₁ t₂) (Set.Icc (↑n) (HAdd.hAdd (↑n) 1))
    ⊢ Eq (intervalIntegral (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (Nat.flo …
  -/
  have h_inc₂ : Set.uIcc t₁ t₂ ⊆ Set.Icc a b := Set.uIcc_of_le h ▸ Set.Icc_subset_Icc h₃ h₄
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    t₁ t₂ : Real
    n : Nat
    h : LE.le t₁ t₂
    h₁ : LE.le (↑n) t₁
    h₂ : LE.le t₂ (HAdd.hAdd (↑n) 1)
    h₃ : LE.le a t₁
    h₄ : LE.le t₂ b
    h_inc₁ : HasSubset.Subset (Set.uIoc t₁ t₂) (Set.Icc (↑n) (HAdd.hAdd (↑n) 1))
    h_inc₂ : HasSubset.Subset (Set.uIcc t₁ t₂) (Set.Icc a b)
    ⊢ Eq (intervalIntegral (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (Nat.flo …
  -/
  rw [← integral_deriv_eq_sub (fun t ht ↦ hf_diff t (h_inc₂ ht)), ← integral_mul_const]
    /-
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      t₁ t₂ : Real
      n : Nat
      h : LE.le t₁ t₂
      h₁ : LE.le (↑n) t₁
      h₂ : LE.le t₂ (HAdd.hAdd (↑n) 1)
      h₃ : LE.le a t₁
      h₄ : LE.le t₂ b
      h_inc₁ : HasSubset.Subset (Set.uIoc t₁ t₂) (Set.Icc (↑n) (HAdd.hAdd (↑n) 1))
      h_inc₂ : HasSubset.Subset (Set.uIcc t₁ t₂) (Set.Icc a b)
      ⊢ Eq (intervalIntegral (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (Nat.flo …
    -/
  · refine integral_congr_ae ?_
    /-
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      t₁ t₂ : Real
      n : Nat
      h : LE.le t₁ t₂
      h₁ : LE.le (↑n) t₁
      h₂ : LE.le t₂ (HAdd.hAdd (↑n) 1)
      h₃ : LE.le a t₁
      h₄ : LE.le t₂ b
      h_inc₁ : HasSubset.Subset (Set.uIoc t₁ t₂) (Set.Icc (↑n) (HAdd.hAdd (↑n) 1))
      h_inc₂ : HasSubset.Subset (Set.uIcc t₁ t₂) (Set.Icc a b)
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.uIoc t₁ t₂) x → Eq (HMul.hMu …
    -/
    filter_upwards [sumlocc c n] with t h h'
    /-
      case h
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      t₁ t₂ : Real
      n : Nat
      h✝ : LE.le t₁ t₂
      h₁ : LE.le (↑n) t₁
      h₂ : LE.le t₂ (HAdd.hAdd (↑n) 1)
      h₃ : LE.le a t₁
      h₄ : LE.le t₂ b
      h_inc₁ : HasSubset.Subset (Set.uIoc t₁ t₂) (Set.Icc (↑n) (HAdd.hAdd (↑n) 1))
      h_inc₂ : HasSubset.Subset (Set.uIcc t₁ t₂) (Set.Icc a b)
      t : Real
      h : Membership.mem (Set.Icc (↑n) (HAdd.hAdd (↑n) 1)) t → Eq ((Finset.Icc 0 (Na …
      h' : Membership.mem (Set.uIoc t₁ t₂) t
      ⊢ Eq (HMul.hMul (deriv f t) ((Finset.Icc 0 (Nat.floor t)).sum fun k => c k)) ( …
    -/
    rw [h (h_inc₁ h')]
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      t₁ t₂ : Real
      n : Nat
      h : LE.le t₁ t₂
      h₁ : LE.le (↑n) t₁
      h₂ : LE.le t₂ (HAdd.hAdd (↑n) 1)
      h₃ : LE.le a t₁
      h₄ : LE.le t₂ b
      h_inc₁ : HasSubset.Subset (Set.uIoc t₁ t₂) (Set.Icc (↑n) (HAdd.hAdd (↑n) 1))
      h_inc₂ : HasSubset.Subset (Set.uIcc t₁ t₂) (Set.Icc a b)
      ⊢ IntervalIntegrable (deriv f) MeasureTheory.MeasureSpace.volume t₁ t₂
    -/
  · refine (intervalIntegrable_iff_integrableOn_Icc_of_le h).mpr (hf_int.mono_set ?_)
    /-
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      t₁ t₂ : Real
      n : Nat
      h : LE.le t₁ t₂
      h₁ : LE.le (↑n) t₁
      h₂ : LE.le t₂ (HAdd.hAdd (↑n) 1)
      h₃ : LE.le a t₁
      h₄ : LE.le t₂ b
      h_inc₁ : HasSubset.Subset (Set.uIoc t₁ t₂) (Set.Icc (↑n) (HAdd.hAdd (↑n) 1))
      h_inc₂ : HasSubset.Subset (Set.uIcc t₁ t₂) (Set.Icc a b)
      ⊢ HasSubset.Subset (Set.Icc t₁ t₂) (Set.Icc a b)
    -/
    rwa [← Set.uIcc_of_le h]
    /-
      🎉 no goals
    -/


private theorem ineqofmemIco {k : ℕ} (hk : k ∈ Set.Ico (⌊a⌋₊ + 1) ⌊b⌋₊) :
    a ≤ k ∧ k + 1 ≤ b := by
  /-
    a b : Real
    k : Nat
    hk : Membership.mem (Set.Ico (HAdd.hAdd (Nat.floor a) 1) (Nat.floor b)) k
    ⊢ And (LE.le a ↑k) (LE.le (HAdd.hAdd (↑k) 1) b)
  -/
  constructor
    /-
      case left
      a b : Real
      k : Nat
      hk : Membership.mem (Set.Ico (HAdd.hAdd (Nat.floor a) 1) (Nat.floor b)) k
      ⊢ LE.le a ↑k
    -/
  · have := (Set.mem_Ico.mp hk).1
    /-
      case left
      a b : Real
      k : Nat
      hk : Membership.mem (Set.Ico (HAdd.hAdd (Nat.floor a) 1) (Nat.floor b)) k
      this : LE.le (HAdd.hAdd (Nat.floor a) 1) k
      ⊢ LE.le a ↑k
    -/
    exact le_of_lt <| (Nat.floor_lt' (by omega)).mp this
    /-
      🎉 no goals
    -/
    /-
      case right
      a b : Real
      k : Nat
      hk : Membership.mem (Set.Ico (HAdd.hAdd (Nat.floor a) 1) (Nat.floor b)) k
      ⊢ LE.le (HAdd.hAdd (↑k) 1) b
    -/
  · rw [← Nat.cast_add_one, ← Nat.le_floor_iff' (Nat.succ_ne_zero k)]
    /-
      case right
      a b : Real
      k : Nat
      hk : Membership.mem (Set.Ico (HAdd.hAdd (Nat.floor a) 1) (Nat.floor b)) k
      ⊢ LE.le k.succ (Nat.floor b)
    -/
    exact (Set.mem_Ico.mp hk).2
    /-
      🎉 no goals
    -/


private theorem ineqofmemIco' {k : ℕ} (hk : k ∈ Ico (⌊a⌋₊ + 1) ⌊b⌋₊) :
    a ≤ k ∧ k + 1 ≤ b :=
                   /-
                     a b : Real
                     k : Nat
                     hk : Membership.mem (Finset.Ico (HAdd.hAdd (Nat.floor a) 1) (Nat.floor b)) k
                     ⊢ Membership.mem (Set.Ico (HAdd.hAdd (Nat.floor a) 1) (Nat.floor b)) k
                   -/
  ineqofmemIco (by rwa [← Finset.coe_Ico])
                   /-
                     🎉 no goals
                   -/


private theorem integrablemulsum (ha : 0 ≤ a) (hb : ⌊a⌋₊ < ⌊b⌋₊)
              /-
                𝕜 : Type u_1
                inst✝ : RCLike 𝕜
                c : Nat → 𝕜
                f : Real → 𝕜
                a b : Real
                ha : LE.le 0 a
                hb : LT.lt (Nat.floor a) (Nat.floor b)
                ⊢ MeasureTheory.Measure Real
              -/
    (hf_int : IntegrableOn (deriv f) (Set.Icc a b)) :
              /-
                🎉 no goals
              -/
    /-
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      ha : LE.le 0 a
      hb : LT.lt (Nat.floor a) (Nat.floor b)
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn (fun t ↦ deriv f t * (∑ k ∈ Icc 0 ⌊t⌋₊, c k)) (Set.Icc a b) := by
    /-
      🎉 no goals
    -/
  have h_locint {t₁ t₂ : ℝ} {n : ℕ} (h : t₁ ≤ t₂) (h₁ : n ≤ t₁) (h₂ : t₂ ≤ n + 1)
      (h₃ : a ≤ t₁) (h₄ : t₂ ≤ b) :
      IntervalIntegrable (fun t ↦ deriv f t * (∑ k ∈ Icc 0 ⌊t⌋₊, c k)) volume t₁ t₂ := by
    rw [intervalIntegrable_iff_integrableOn_Icc_of_le h]
    exact (IntegrableOn.mono_set (hf_int.mul_const _) (Set.Icc_subset_Icc h₃ h₄)).congr
      <| ae_restrict_of_ae_restrict_of_subset (Set.Icc_subset_Icc h₁ h₂)
        <| (ae_restrict_iff' measurableSet_Icc).mpr
          (by filter_upwards [sumlocc c n] with t h ht using by rw [h ht])
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hb : LT.lt (Nat.floor a) (Nat.floor b)
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    h_locint : ∀ {t₁ t₂ : Real} {n : Nat}, LE.le t₁ t₂ → LE.le (↑n) t₁ → LE.le t₂  …
    ⊢ MeasureTheory.IntegrableOn (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (N …
  -/
  have aux1 : 0 ≤ b := (Nat.pos_of_floor_pos <| (Nat.zero_le _).trans_lt hb).le
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hb : LT.lt (Nat.floor a) (Nat.floor b)
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    h_locint : ∀ {t₁ t₂ : Real} {n : Nat}, LE.le t₁ t₂ → LE.le (↑n) t₁ → LE.le t₂  …
    aux1 : LE.le 0 b
    ⊢ MeasureTheory.IntegrableOn (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (N …
  -/
  have aux2 : ⌊a⌋₊ + 1 ≤ b := by rwa [← Nat.cast_add_one, ← Nat.le_floor_iff aux1]
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hb : LT.lt (Nat.floor a) (Nat.floor b)
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    h_locint : ∀ {t₁ t₂ : Real} {n : Nat}, LE.le t₁ t₂ → LE.le (↑n) t₁ → LE.le t₂  …
    aux1 : LE.le 0 b
    aux2 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
    ⊢ MeasureTheory.IntegrableOn (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (N …
  -/
  have aux3 : a ≤ ⌊a⌋₊ + 1 := (Nat.lt_floor_add_one _).le
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hb : LT.lt (Nat.floor a) (Nat.floor b)
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    h_locint : ∀ {t₁ t₂ : Real} {n : Nat}, LE.le t₁ t₂ → LE.le (↑n) t₁ → LE.le t₂  …
    aux1 : LE.le 0 b
    aux2 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
    aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
    ⊢ MeasureTheory.IntegrableOn (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (N …
  -/
  have aux4 : a ≤ ⌊b⌋₊ := le_of_lt (by rwa [← Nat.floor_lt ha])
  -- now break up into 3 subintervals
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hb : LT.lt (Nat.floor a) (Nat.floor b)
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    h_locint : ∀ {t₁ t₂ : Real} {n : Nat}, LE.le t₁ t₂ → LE.le (↑n) t₁ → LE.le t₂  …
    aux1 : LE.le 0 b
    aux2 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
    aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
    aux4 : LE.le a ↑(Nat.floor b)
    ⊢ MeasureTheory.IntegrableOn (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (N …
  -/
  rw [← intervalIntegrable_iff_integrableOn_Icc_of_le (aux3.trans aux2)]
  have I1 : IntervalIntegrable _ volume a ↑(⌊a⌋₊ + 1) :=
    h_locint (mod_cast aux3) (Nat.floor_le ha) (mod_cast le_rfl) le_rfl (mod_cast aux2)
  have I2 : IntervalIntegrable _ volume ↑(⌊a⌋₊ + 1) ⌊b⌋₊ :=
    trans_iterate_Ico hb fun k hk ↦ h_locint (mod_cast k.le_succ)
      le_rfl (mod_cast le_rfl) (ineqofmemIco hk).1 (mod_cast (ineqofmemIco hk).2)
  have I3 : IntervalIntegrable _ volume ⌊b⌋₊ b :=
    h_locint (Nat.floor_le aux1) le_rfl (Nat.lt_floor_add_one _).le aux4 le_rfl
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hb : LT.lt (Nat.floor a) (Nat.floor b)
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    h_locint : ∀ {t₁ t₂ : Real} {n : Nat}, LE.le t₁ t₂ → LE.le (↑n) t₁ → LE.le t₂  …
    aux1 : LE.le 0 b
    aux2 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
    aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
    aux4 : LE.le a ↑(Nat.floor b)
    I1 : IntervalIntegrable (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (Nat.fl …
    I2 : IntervalIntegrable (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (Nat.fl …
    I3 : IntervalIntegrable (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (Nat.fl …
    ⊢ IntervalIntegrable (fun t => HMul.hMul (deriv f t) ((Finset.Icc 0 (Nat.floor …
  -/
  exact (I1.trans I2).trans I3
  /-
    🎉 no goals
  -/


/-- Abel's summation formula. -/
theorem _root_.sum_mul_eq_sub_sub_integral_mul (ha : 0 ≤ a) (hab : a ≤ b)
    (hf_diff : ∀ t ∈ Set.Icc a b, DifferentiableAt ℝ f t)
              /-
                𝕜 : Type u_1
                inst✝ : RCLike 𝕜
                c : Nat → 𝕜
                f : Real → 𝕜
                a b : Real
                ha : LE.le 0 a
                hab : LE.le a b
                hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
                ⊢ MeasureTheory.Measure Real
              -/
    (hf_int : IntegrableOn (deriv f) (Set.Icc a b)) :
              /-
                🎉 no goals
              -/
    ∑ k ∈ Ioc ⌊a⌋₊ ⌊b⌋₊, f k * c k =
      f b * (∑ k ∈ Icc 0 ⌊b⌋₊, c k) - f a * (∑ k ∈ Icc 0 ⌊a⌋₊, c k) -
        ∫ t in Set.Ioc a b, deriv f t * (∑ k ∈ Icc 0 ⌊t⌋₊, c k) := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hab : LE.le a b
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    ⊢ Eq ((Finset.Ioc (Nat.floor a) (Nat.floor b)).sum fun k => HMul.hMul (f ↑k) ( …
  -/
  rw [← integral_of_le hab]
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hab : LE.le a b
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    ⊢ Eq ((Finset.Ioc (Nat.floor a) (Nat.floor b)).sum fun k => HMul.hMul (f ↑k) ( …
  -/
  have aux1 : ⌊a⌋₊ ≤ a := Nat.floor_le ha
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hab : LE.le a b
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    aux1 : LE.le (↑(Nat.floor a)) a
    ⊢ Eq ((Finset.Ioc (Nat.floor a) (Nat.floor b)).sum fun k => HMul.hMul (f ↑k) ( …
  -/
  have aux2 : b ≤ ⌊b⌋₊ + 1 := (Nat.lt_floor_add_one _).le
  -- We consider two cases depending on whether the sum is empty or not
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hab : LE.le a b
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    aux1 : LE.le (↑(Nat.floor a)) a
    aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
    ⊢ Eq ((Finset.Ioc (Nat.floor a) (Nat.floor b)).sum fun k => HMul.hMul (f ↑k) ( …
  -/
  obtain hb | hb := eq_or_lt_of_le (Nat.floor_le_floor hab)
  · rw [hb, Ioc_eq_empty_of_le le_rfl, sum_empty, ← sub_mul,
      integralmulsum c hf_diff hf_int _ _ ⌊b⌋₊ hab (hb ▸ aux1) aux2 le_rfl le_rfl, sub_self]
  /-
    case inr
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hab : LE.le a b
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    aux1 : LE.le (↑(Nat.floor a)) a
    aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
    hb : LT.lt (Nat.floor a) (Nat.floor b)
    ⊢ Eq ((Finset.Ioc (Nat.floor a) (Nat.floor b)).sum fun k => HMul.hMul (f ↑k) ( …
  -/
  have aux3 : a ≤ ⌊a⌋₊ + 1 := (Nat.lt_floor_add_one _).le
  /-
    case inr
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hab : LE.le a b
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    aux1 : LE.le (↑(Nat.floor a)) a
    aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
    hb : LT.lt (Nat.floor a) (Nat.floor b)
    aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
    ⊢ Eq ((Finset.Ioc (Nat.floor a) (Nat.floor b)).sum fun k => HMul.hMul (f ↑k) ( …
  -/
  have aux4 : ⌊a⌋₊ + 1 ≤ b := by rwa [← Nat.cast_add_one,  ← Nat.le_floor_iff (ha.trans hab)]
  /-
    case inr
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hab : LE.le a b
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    aux1 : LE.le (↑(Nat.floor a)) a
    aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
    hb : LT.lt (Nat.floor a) (Nat.floor b)
    aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
    aux4 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
    ⊢ Eq ((Finset.Ioc (Nat.floor a) (Nat.floor b)).sum fun k => HMul.hMul (f ↑k) ( …
  -/
  have aux5 : ⌊b⌋₊ ≤ b := Nat.floor_le (ha.trans hab)
  /-
    case inr
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    a b : Real
    ha : LE.le 0 a
    hab : LE.le a b
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
    aux1 : LE.le (↑(Nat.floor a)) a
    aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
    hb : LT.lt (Nat.floor a) (Nat.floor b)
    aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
    aux4 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
    aux5 : LE.le (↑(Nat.floor b)) b
    ⊢ Eq ((Finset.Ioc (Nat.floor a) (Nat.floor b)).sum fun k => HMul.hMul (f ↑k) ( …
  -/
  have aux6 : a ≤ ⌊b⌋₊ := Nat.floor_lt ha |>.mp hb |>.le
  simp_rw [← smul_eq_mul, sum_Ioc_by_parts (fun k ↦ f k) _ hb, range_eq_Ico, Nat.Ico_succ_right,
    smul_eq_mul]
  have : ∑ k ∈ Ioc ⌊a⌋₊ (⌊b⌋₊ - 1), (f ↑(k + 1) - f k) * ∑ n ∈ Icc 0 k, c n =
        ∑ k ∈ Ico (⌊a⌋₊ + 1) ⌊b⌋₊, ∫ t in k..↑(k + 1), deriv f t * ∑ n ∈ Icc 0 ⌊t⌋₊, c n := by
    rw [← Nat.Ico_succ_succ, Nat.succ_eq_add_one,  Nat.succ_eq_add_one, Nat.sub_add_cancel
      (by omega), Eq.comm]
    exact sum_congr rfl fun k hk ↦ (integralmulsum c hf_diff hf_int _ _ _  (mod_cast k.le_succ)
      le_rfl (mod_cast le_rfl) (ineqofmemIco' hk).1 <| mod_cast (ineqofmemIco' hk).2)
  rw [this, sum_integral_adjacent_intervals_Ico hb, Nat.cast_add, Nat.cast_one,
    ← integral_interval_sub_left (a := a) (c := ⌊a⌋₊ + 1),
    ← integral_add_adjacent_intervals (b := ⌊b⌋₊) (c := b),
    integralmulsum c hf_diff hf_int _ _ _ aux3 aux1 le_rfl le_rfl aux4,
    integralmulsum c hf_diff hf_int _ _ _ aux5 le_rfl aux2 aux6 le_rfl]
    /-
      case inr
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      ha : LE.le 0 a
      hab : LE.le a b
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      aux1 : LE.le (↑(Nat.floor a)) a
      aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
      hb : LT.lt (Nat.floor a) (Nat.floor b)
      aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
      aux4 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
      aux5 : LE.le (↑(Nat.floor b)) b
      aux6 : LE.le a ↑(Nat.floor b)
      this : Eq ((Finset.Ioc (Nat.floor a) (HSub.hSub (Nat.floor b) 1)).sum fun k => …
      ⊢ Eq (HSub.hSub (HSub.hSub (HMul.hMul (f ↑(Nat.floor b)) ((Finset.Icc 0 (Nat.f …
    -/
  · ring
    /-
      🎉 no goals
    -/
  -- now deal with the integrability side goals
  -- (Note we have 5 goals, but the 1st and 3rd are identical. TODO: find a non-hacky way of dealing
  -- with both at once.)
    /-
      case inr.hab
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      ha : LE.le 0 a
      hab : LE.le a b
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      aux1 : LE.le (↑(Nat.floor a)) a
      aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
      hb : LT.lt (Nat.floor a) (Nat.floor b)
      aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
      aux4 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
      aux5 : LE.le (↑(Nat.floor b)) b
      aux6 : LE.le a ↑(Nat.floor b)
      this : Eq ((Finset.Ioc (Nat.floor a) (HSub.hSub (Nat.floor b) 1)).sum fun k => …
      ⊢ IntervalIntegrable (fun x => HMul.hMul (deriv f x) ((Finset.Icc 0 (Nat.floor …
    -/
  · rw [intervalIntegrable_iff_integrableOn_Icc_of_le aux6]
    /-
      case inr.hab
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      ha : LE.le 0 a
      hab : LE.le a b
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      aux1 : LE.le (↑(Nat.floor a)) a
      aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
      hb : LT.lt (Nat.floor a) (Nat.floor b)
      aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
      aux4 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
      aux5 : LE.le (↑(Nat.floor b)) b
      aux6 : LE.le a ↑(Nat.floor b)
      this : Eq ((Finset.Ioc (Nat.floor a) (HSub.hSub (Nat.floor b) 1)).sum fun k => …
      ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (deriv f x) ((Finset.Icc 0 (N …
    -/
    exact (integrablemulsum c ha hb hf_int).mono_set (Set.Icc_subset_Icc_right aux5)
    /-
      🎉 no goals
    -/
    /-
      case inr.hbc
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      ha : LE.le 0 a
      hab : LE.le a b
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      aux1 : LE.le (↑(Nat.floor a)) a
      aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
      hb : LT.lt (Nat.floor a) (Nat.floor b)
      aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
      aux4 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
      aux5 : LE.le (↑(Nat.floor b)) b
      aux6 : LE.le a ↑(Nat.floor b)
      this : Eq ((Finset.Ioc (Nat.floor a) (HSub.hSub (Nat.floor b) 1)).sum fun k => …
      ⊢ IntervalIntegrable (fun x => HMul.hMul (deriv f x) ((Finset.Icc 0 (Nat.floor …
    -/
  · rw [intervalIntegrable_iff_integrableOn_Icc_of_le aux5]
    /-
      case inr.hbc
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      ha : LE.le 0 a
      hab : LE.le a b
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      aux1 : LE.le (↑(Nat.floor a)) a
      aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
      hb : LT.lt (Nat.floor a) (Nat.floor b)
      aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
      aux4 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
      aux5 : LE.le (↑(Nat.floor b)) b
      aux6 : LE.le a ↑(Nat.floor b)
      this : Eq ((Finset.Ioc (Nat.floor a) (HSub.hSub (Nat.floor b) 1)).sum fun k => …
      ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (deriv f x) ((Finset.Icc 0 (N …
    -/
    exact (integrablemulsum c ha hb hf_int).mono_set (Set.Icc_subset_Icc_left aux6)
    /-
      🎉 no goals
    -/
    /-
      case inr.hab
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      ha : LE.le 0 a
      hab : LE.le a b
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      aux1 : LE.le (↑(Nat.floor a)) a
      aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
      hb : LT.lt (Nat.floor a) (Nat.floor b)
      aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
      aux4 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
      aux5 : LE.le (↑(Nat.floor b)) b
      aux6 : LE.le a ↑(Nat.floor b)
      this : Eq ((Finset.Ioc (Nat.floor a) (HSub.hSub (Nat.floor b) 1)).sum fun k => …
      ⊢ IntervalIntegrable (fun x => HMul.hMul (deriv f x) ((Finset.Icc 0 (Nat.floor …
    -/
  · rw [intervalIntegrable_iff_integrableOn_Icc_of_le aux6]
    /-
      case inr.hab
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      ha : LE.le 0 a
      hab : LE.le a b
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      aux1 : LE.le (↑(Nat.floor a)) a
      aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
      hb : LT.lt (Nat.floor a) (Nat.floor b)
      aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
      aux4 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
      aux5 : LE.le (↑(Nat.floor b)) b
      aux6 : LE.le a ↑(Nat.floor b)
      this : Eq ((Finset.Ioc (Nat.floor a) (HSub.hSub (Nat.floor b) 1)).sum fun k => …
      ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (deriv f x) ((Finset.Icc 0 (N …
    -/
    exact (integrablemulsum c ha hb hf_int).mono_set (Set.Icc_subset_Icc_right aux5)
    /-
      🎉 no goals
    -/
    /-
      case inr.hac
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      ha : LE.le 0 a
      hab : LE.le a b
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      aux1 : LE.le (↑(Nat.floor a)) a
      aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
      hb : LT.lt (Nat.floor a) (Nat.floor b)
      aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
      aux4 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
      aux5 : LE.le (↑(Nat.floor b)) b
      aux6 : LE.le a ↑(Nat.floor b)
      this : Eq ((Finset.Ioc (Nat.floor a) (HSub.hSub (Nat.floor b) 1)).sum fun k => …
      ⊢ IntervalIntegrable (fun x => HMul.hMul (deriv f x) ((Finset.Icc 0 (Nat.floor …
    -/
  · rw [intervalIntegrable_iff_integrableOn_Icc_of_le aux3]
    /-
      case inr.hac
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      a b : Real
      ha : LE.le 0 a
      hab : LE.le a b
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc a b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc a b) MeasureTheory.Meas …
      aux1 : LE.le (↑(Nat.floor a)) a
      aux2 : LE.le b (HAdd.hAdd (↑(Nat.floor b)) 1)
      hb : LT.lt (Nat.floor a) (Nat.floor b)
      aux3 : LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
      aux4 : LE.le (HAdd.hAdd (↑(Nat.floor a)) 1) b
      aux5 : LE.le (↑(Nat.floor b)) b
      aux6 : LE.le a ↑(Nat.floor b)
      this : Eq ((Finset.Ioc (Nat.floor a) (HSub.hSub (Nat.floor b) 1)).sum fun k => …
      ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (deriv f x) ((Finset.Icc 0 (N …
    -/
    exact (integrablemulsum c ha hb hf_int).mono_set (Set.Icc_subset_Icc_right aux4)
    /-
      🎉 no goals
    -/
  · exact fun k hk ↦ (intervalIntegrable_iff_integrableOn_Icc_of_le (mod_cast k.le_succ)).mpr
      <| (integrablemulsum c ha hb hf_int).mono_set
        <| (Set.Icc_subset_Icc_iff (mod_cast k.le_succ)).mpr <| mod_cast (ineqofmemIco hk)


/-- Specialized version of `sum_mul_eq_sub_sub_integral_mul` for the case `a = 0`.-/
theorem sum_mul_eq_sub_integral_mul {b : ℝ} (hb : 0 ≤ b)
    (hf_diff : ∀ t ∈ Set.Icc 0 b, DifferentiableAt ℝ f t)
              /-
                𝕜 : Type u_1
                inst✝ : RCLike 𝕜
                c : Nat → 𝕜
                f : Real → 𝕜
                a b✝ b : Real
                hb : LE.le 0 b
                hf_diff : ∀ (t : Real), Membership.mem (Set.Icc 0 b) t → DifferentiableAt Real …
                ⊢ MeasureTheory.Measure Real
              -/
    (hf_int : IntegrableOn (deriv f) (Set.Icc 0 b)) :
              /-
                🎉 no goals
              -/
    ∑ k ∈ Icc 0 ⌊b⌋₊, f k * c k =
      f b * (∑ k ∈ Icc 0 ⌊b⌋₊, c k) - ∫ t in Set.Ioc 0 b, deriv f t * (∑ k ∈ Icc 0 ⌊t⌋₊, c k) := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    b : Real
    hb : LE.le 0 b
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc 0 b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc 0 b) MeasureTheory.Meas …
    ⊢ Eq ((Finset.Icc 0 (Nat.floor b)).sum fun k => HMul.hMul (f ↑k) (c k)) (HSub. …
  -/
  nth_rewrite 1 [Finset.Icc_eq_cons_Ioc (Nat.zero_le _)]
  rw [sum_cons, ← Nat.floor_zero (α := ℝ), sum_mul_eq_sub_sub_integral_mul c le_rfl hb hf_diff
    hf_int, Nat.floor_zero, Nat.cast_zero, Icc_self, sum_singleton]
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    b : Real
    hb : LE.le 0 b
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc 0 b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc 0 b) MeasureTheory.Meas …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (f 0) (c 0)) (HSub.hSub (HSub.hSub (HMul.hMul (f b) …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Specialized version of `sum_mul_eq_sub_integral_mul` when the first coefficient of the sequence
`c` is equal to `0`. -/
theorem sum_mul_eq_sub_integral_mul' (hc : c 0 = 0) (b : ℝ)
    (hf_diff : ∀ t ∈ Set.Icc 1 b, DifferentiableAt ℝ f t)
              /-
                𝕜 : Type u_1
                inst✝ : RCLike 𝕜
                c : Nat → 𝕜
                f : Real → 𝕜
                a b✝ : Real
                hc : Eq (c 0) 0
                b : Real
                hf_diff : ∀ (t : Real), Membership.mem (Set.Icc 1 b) t → DifferentiableAt Real …
                ⊢ MeasureTheory.Measure Real
              -/
    (hf_int : IntegrableOn (deriv f) (Set.Icc 1 b)) :
              /-
                🎉 no goals
              -/
    ∑ k ∈ Icc 0 ⌊b⌋₊, f k * c k =
      f b * (∑ k ∈ Icc 0 ⌊b⌋₊, c k) - ∫ t in Set.Ioc 1 b, deriv f t * (∑ k ∈ Icc 0 ⌊t⌋₊, c k) := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    c : Nat → 𝕜
    f : Real → 𝕜
    hc : Eq (c 0) 0
    b : Real
    hf_diff : ∀ (t : Real), Membership.mem (Set.Icc 1 b) t → DifferentiableAt Real …
    hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc 1 b) MeasureTheory.Meas …
    ⊢ Eq ((Finset.Icc 0 (Nat.floor b)).sum fun k => HMul.hMul (f ↑k) (c k)) (HSub. …
  -/
  obtain hb | hb := le_or_gt 1 b
    /-
      case inl
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      hc : Eq (c 0) 0
      b : Real
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc 1 b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc 1 b) MeasureTheory.Meas …
      hb : LE.le 1 b
      ⊢ Eq ((Finset.Icc 0 (Nat.floor b)).sum fun k => HMul.hMul (f ↑k) (c k)) (HSub. …
    -/
  · have : 1 ≤ ⌊b⌋₊ := (Nat.one_le_floor_iff _).mpr hb
    nth_rewrite 1 [Finset.Icc_eq_cons_Ioc (by omega), sum_cons, ← Nat.Icc_succ_left,
      Finset.Icc_eq_cons_Ioc (by omega), sum_cons]
    rw [Nat.succ_eq_add_one, zero_add, ← Nat.floor_one (α := ℝ),
      sum_mul_eq_sub_sub_integral_mul c zero_le_one hb hf_diff hf_int, Nat.floor_one, Nat.cast_one,
      Finset.Icc_eq_cons_Ioc zero_le_one, sum_cons, show 1 = 0 + 1 by rfl, Nat.Ioc_succ_singleton,
      zero_add, sum_singleton, hc, mul_zero, zero_add]
    /-
      case inl
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      c : Nat → 𝕜
      f : Real → 𝕜
      hc : Eq (c 0) 0
      b : Real
      hf_diff : ∀ (t : Real), Membership.mem (Set.Icc 1 b) t → DifferentiableAt Real …
      hf_int : MeasureTheory.IntegrableOn (deriv f) (Set.Icc 1 b) MeasureTheory.Meas …
      hb : LE.le 1 b
      this : LE.le 1 (Nat.floor b)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (f 1) (c 1)) (HSub.hSub (HSub.hSub (HMul.hMul (f b) …
    -/
    ring
    /-
      🎉 no goals
    -/
  · simp_rw [Nat.floor_eq_zero.mpr hb, Icc_self, sum_singleton, Nat.cast_zero, hc, mul_zero,
      Set.Ioc_eq_empty_of_le hb.le, Measure.restrict_empty, integral_zero_measure, sub_self]

