/-- The unit interval `[0,1]` in ℝ. -/
abbrev unitInterval : Set ℝ :=
  Set.Icc 0 1


@[inherit_doc]
scoped[unitInterval] notation "I" => unitInterval


theorem zero_mem : (0 : ℝ) ∈ I :=
  ⟨le_rfl, zero_le_one⟩


theorem one_mem : (1 : ℝ) ∈ I :=
  ⟨zero_le_one, le_rfl⟩


theorem mul_mem {x y : ℝ} (hx : x ∈ I) (hy : y ∈ I) : x * y ∈ I :=
  ⟨mul_nonneg hx.1 hy.1, mul_le_one₀ hx.2 hy.1 hy.2⟩


theorem div_mem {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x ≤ y) : x / y ∈ I :=
  ⟨div_nonneg hx hy, div_le_one_of_le₀ hxy hy⟩


theorem fract_mem (x : ℝ) : fract x ∈ I :=
  ⟨fract_nonneg _, (fract_lt_one _).le⟩


theorem mem_iff_one_sub_mem {t : ℝ} : t ∈ I ↔ 1 - t ∈ I := by
  /-
    t : Real
    ⊢ Iff (Membership.mem unitInterval t) (Membership.mem unitInterval (HSub.hSub  …
  -/
  rw [mem_Icc, mem_Icc]
  /-
    t : Real
    ⊢ Iff (And (LE.le 0 t) (LE.le t 1)) (And (LE.le 0 (HSub.hSub 1 t)) (LE.le (HSu …
  -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  constructor <;> intro <;> constructor <;> linarith
                                            /-
                                              🎉 no goals
                                            -/


instance hasZero : Zero I :=
  ⟨⟨0, zero_mem⟩⟩


instance hasOne : One I :=
          /-
            ⊢ Membership.mem unitInterval 1
          -/
                          /-
                            🎉 no goals
                          -/
  ⟨⟨1, by constructor <;> norm_num⟩⟩
                          /-
                            🎉 no goals
                          -/


instance : ZeroLEOneClass I := ⟨zero_le_one (α := ℝ)⟩


instance : BoundedOrder I := have : Fact ((0 : ℝ) ≤ 1) := ⟨zero_le_one⟩; inferInstance


lemma univ_eq_Icc : (univ : Set I) = Icc (0 : I) (1 : I) := Icc_bot_top.symm


@[norm_cast] theorem coe_ne_zero {x : I} : (x : ℝ) ≠ 0 ↔ x ≠ 0 := coe_eq_zero.not

@[norm_cast] theorem coe_ne_one {x : I} : (x : ℝ) ≠ 1 ↔ x ≠ 1 := coe_eq_one.not

@[simp, norm_cast] theorem coe_pos {x : I} : (0 : ℝ) < x ↔ 0 < x := Iff.rfl

@[simp, norm_cast] theorem coe_lt_one {x : I} : (x : ℝ) < 1 ↔ x < 1 := Iff.rfl


instance : Nonempty I :=
  ⟨0⟩


instance : Mul I :=
  ⟨fun x y => ⟨x * y, mul_mem x.2 y.2⟩⟩


theorem mul_le_left {x y : I} : x * y ≤ x :=
  Subtype.coe_le_coe.mp <| mul_le_of_le_one_right x.2.1 y.2.2


theorem mul_le_right {x y : I} : x * y ≤ y :=
  Subtype.coe_le_coe.mp <| mul_le_of_le_one_left y.2.1 x.2.2


/-- Unit interval central symmetry. -/
def symm : I → I := fun t => ⟨1 - t, mem_iff_one_sub_mem.mp t.prop⟩


@[inherit_doc]
scoped notation "σ" => unitInterval.symm


@[simp]
theorem symm_zero : σ 0 = 1 :=
                    /-
                      ⊢ Eq ↑(unitInterval.symm 0) ↑1
                    -/
  Subtype.ext <| by simp [symm]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem symm_one : σ 1 = 0 :=
                    /-
                      ⊢ Eq ↑(unitInterval.symm 1) ↑0
                    -/
  Subtype.ext <| by simp [symm]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem symm_symm (x : I) : σ (σ x) = x :=
                    /-
                      x : ↑unitInterval
                      ⊢ Eq ↑(unitInterval.symm (unitInterval.symm x)) ↑x
                    -/
  Subtype.ext <| by simp [symm]
                    /-
                      🎉 no goals
                    -/


theorem symm_involutive : Function.Involutive (symm : I → I) := symm_symm


theorem symm_bijective : Function.Bijective (symm : I → I) := symm_involutive.bijective


@[simp]
theorem coe_symm_eq (x : I) : (σ x : ℝ) = 1 - x :=
  rfl

-- Porting note: Proof used to be `by continuity!`

@[continuity, fun_prop]
theorem continuous_symm : Continuous σ := by
  /-
    ⊢ Continuous unitInterval.symm
  -/
  apply Continuous.subtype_mk (by fun_prop)
  /-
    🎉 no goals
  -/


/-- `unitInterval.symm` as a `Homeomorph`. -/
@[simps]
def symmHomeomorph : I ≃ₜ I where
  toFun := symm
  invFun := symm
  left_inv := symm_symm
  right_inv := symm_symm


theorem strictAnti_symm : StrictAnti σ := fun _ _ h ↦ sub_lt_sub_left (α := ℝ) h _


@[deprecated (since := "2024-02-27")] alias involutive_symm := symm_involutive

@[deprecated (since := "2024-02-27")] alias bijective_symm := symm_bijective


@[simp]
theorem symm_inj {i j : I} : σ i = σ j ↔ i = j := symm_bijective.injective.eq_iff


theorem half_le_symm_iff (t : I) : 1 / 2 ≤ (σ t : ℝ) ↔ (t : ℝ) ≤ 1 / 2 := by
  /-
    t : ↑unitInterval
    ⊢ Iff (LE.le (1 / 2) ↑(unitInterval.symm t)) (LE.le (↑t) (1 / 2))
  -/
  rw [coe_symm_eq, le_sub_iff_add_le, add_comm, ← le_sub_iff_add_le, sub_half]
  /-
    🎉 no goals
  -/


@[simp]
lemma symm_eq_one {i : I} : σ i = 1 ↔ i = 0 := by
  /-
    i : ↑unitInterval
    ⊢ Iff (Eq (unitInterval.symm i) 1) (Eq i 0)
  -/
  rw [← symm_zero, symm_inj]
  /-
    🎉 no goals
  -/


@[simp]
lemma symm_eq_zero {i : I} : σ i = 0 ↔ i = 1 := by
  /-
    i : ↑unitInterval
    ⊢ Iff (Eq (unitInterval.symm i) 0) (Eq i 1)
  -/
  rw [← symm_one, symm_inj]
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_le_symm {i j : I} : σ i ≤ σ j ↔ j ≤ i := by
  /-
    i j : ↑unitInterval
    ⊢ Iff (LE.le (unitInterval.symm i) (unitInterval.symm j)) (LE.le j i)
  -/
  simp only [symm, Subtype.mk_le_mk, sub_le_sub_iff, add_le_add_iff_left, Subtype.coe_le_coe]
  /-
    🎉 no goals
  -/


theorem le_symm_comm {i j : I} : i ≤ σ j ↔ j ≤ σ i := by
  /-
    i j : ↑unitInterval
    ⊢ Iff (LE.le i (unitInterval.symm j)) (LE.le j (unitInterval.symm i))
  -/
  rw [← symm_le_symm, symm_symm]
  /-
    🎉 no goals
  -/


theorem symm_le_comm {i j : I} : σ i ≤ j ↔ σ j ≤ i := by
  /-
    i j : ↑unitInterval
    ⊢ Iff (LE.le (unitInterval.symm i) j) (LE.le (unitInterval.symm j) i)
  -/
  rw [← symm_le_symm, symm_symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_lt_symm {i j : I} : σ i < σ j ↔ j < i := by
  /-
    i j : ↑unitInterval
    ⊢ Iff (LT.lt (unitInterval.symm i) (unitInterval.symm j)) (LT.lt j i)
  -/
  simp only [symm, Subtype.mk_lt_mk, sub_lt_sub_iff_left, Subtype.coe_lt_coe]
  /-
    🎉 no goals
  -/


theorem lt_symm_comm {i j : I} : i < σ j ↔ j < σ i := by
  /-
    i j : ↑unitInterval
    ⊢ Iff (LT.lt i (unitInterval.symm j)) (LT.lt j (unitInterval.symm i))
  -/
  rw [← symm_lt_symm, symm_symm]
  /-
    🎉 no goals
  -/


theorem symm_lt_comm {i j : I} : σ i < j ↔ σ j < i := by
  /-
    i j : ↑unitInterval
    ⊢ Iff (LT.lt (unitInterval.symm i) j) (LT.lt (unitInterval.symm j) i)
  -/
  rw [← symm_lt_symm, symm_symm]
  /-
    🎉 no goals
  -/


instance : ConnectedSpace I :=
  Subtype.connectedSpace ⟨nonempty_Icc.mpr zero_le_one, isPreconnected_Icc⟩


theorem nonneg (x : I) : 0 ≤ (x : ℝ) :=
  x.2.1


                                                         /-
                                                           x : ↑unitInterval
                                                           ⊢ LE.le 0 (HSub.hSub 1 ↑x)
                                                         -/
theorem one_minus_nonneg (x : I) : 0 ≤ 1 - (x : ℝ) := by simpa using x.2.2
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem le_one (x : I) : (x : ℝ) ≤ 1 :=
  x.2.2


                                                         /-
                                                           x : ↑unitInterval
                                                           ⊢ LE.le (HSub.hSub 1 ↑x) 1
                                                         -/
theorem one_minus_le_one (x : I) : 1 - (x : ℝ) ≤ 1 := by simpa using x.2.1
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem add_pos {t : I} {x : ℝ} (hx : 0 < x) : 0 < (x + t : ℝ) :=
  add_pos_of_pos_of_nonneg hx <| nonneg _


/-- like `unitInterval.nonneg`, but with the inequality in `I`. -/
theorem nonneg' {t : I} : 0 ≤ t :=
  t.2.1


/-- like `unitInterval.le_one`, but with the inequality in `I`. -/
theorem le_one' {t : I} : t ≤ 1 :=
  t.2.2


protected lemma pos_iff_ne_zero {x : I} : 0 < x ↔ x ≠ 0 := bot_lt_iff_ne_bot


protected lemma lt_one_iff_ne_one {x : I} : x < 1 ↔ x ≠ 1 := lt_top_iff_ne_top


lemma eq_one_or_eq_zero_of_le_mul {i j : I} (h : i ≤ j * i) : i = 0 ∨ j = 1 := by
  /-
    i j : ↑unitInterval
    h : LE.le i (HMul.hMul j i)
    ⊢ Or (Eq i 0) (Eq j 1)
  -/
  contrapose! h
  rw [← unitInterval.lt_one_iff_ne_one, ← coe_lt_one, ← unitInterval.pos_iff_ne_zero,
    ← coe_pos] at h
  /-
    i j : ↑unitInterval
    h : And (LT.lt 0 ↑i) (LT.lt (↑j) 1)
    ⊢ LT.lt (HMul.hMul j i) i
  -/
  rw [← Subtype.coe_lt_coe, coe_mul]
  /-
    i j : ↑unitInterval
    h : And (LT.lt 0 ↑i) (LT.lt (↑j) 1)
    ⊢ LT.lt (HMul.hMul ↑j ↑i) ↑i
  -/
  simpa using mul_lt_mul_of_pos_right h.right h.left
  /-
    🎉 no goals
  -/


instance : Nontrivial I := ⟨⟨1, 0, (one_ne_zero <| congrArg Subtype.val ·)⟩⟩


theorem mul_pos_mem_iff {a t : ℝ} (ha : 0 < a) : a * t ∈ I ↔ t ∈ Set.Icc (0 : ℝ) (1 / a) := by
  /-
    a t : Real
    ha : LT.lt 0 a
    ⊢ Iff (Membership.mem unitInterval (HMul.hMul a t)) (Membership.mem (Set.Icc 0 …
  -/
  constructor <;> rintro ⟨h₁, h₂⟩ <;> constructor
    /-
      case mp.intro.left
      a t : Real
      ha : LT.lt 0 a
      h₁ : LE.le 0 (HMul.hMul a t)
      h₂ : LE.le (HMul.hMul a t) 1
      ⊢ LE.le 0 t
    -/
  · exact nonneg_of_mul_nonneg_right h₁ ha
    /-
      🎉 no goals
    -/
    /-
      case mp.intro.right
      a t : Real
      ha : LT.lt 0 a
      h₁ : LE.le 0 (HMul.hMul a t)
      h₂ : LE.le (HMul.hMul a t) 1
      ⊢ LE.le t (HDiv.hDiv 1 a)
    -/
  · rwa [le_div_iff₀ ha, mul_comm]
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro.left
      a t : Real
      ha : LT.lt 0 a
      h₁ : LE.le 0 t
      h₂ : LE.le t (HDiv.hDiv 1 a)
      ⊢ LE.le 0 (HMul.hMul a t)
    -/
  · exact mul_nonneg ha.le h₁
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro.right
      a t : Real
      ha : LT.lt 0 a
      h₁ : LE.le 0 t
      h₂ : LE.le t (HDiv.hDiv 1 a)
      ⊢ LE.le (HMul.hMul a t) 1
    -/
  · rwa [le_div_iff₀ ha, mul_comm] at h₂
    /-
      🎉 no goals
    -/


theorem two_mul_sub_one_mem_iff {t : ℝ} : 2 * t - 1 ∈ I ↔ t ∈ Set.Icc (1 / 2 : ℝ) 1 := by
  /-
    t : Real
    ⊢ Iff (Membership.mem unitInterval (HSub.hSub (HMul.hMul 2 t) 1)) (Membership. …
  -/
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  constructor <;> rintro ⟨h₁, h₂⟩ <;> constructor <;> linarith
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- The unit interval as a submonoid of ℝ. -/
def submonoid : Submonoid ℝ where
  carrier := unitInterval
  one_mem' := unitInterval.one_mem
  mul_mem' := unitInterval.mul_mem


@[simp] theorem coe_unitIntervalSubmonoid : submonoid = unitInterval := rfl

@[simp] theorem mem_unitIntervalSubmonoid {x} : x ∈ submonoid ↔ x ∈ unitInterval :=
  Iff.rfl


protected theorem prod_mem {ι : Type*} {t : Finset ι} {f : ι → ℝ}
    (h : ∀ c ∈ t, f c ∈ unitInterval) :
    ∏ c ∈ t, f c ∈ unitInterval := _root_.prod_mem (S := unitInterval.submonoid) h


instance : LinearOrderedCommMonoidWithZero I where
  zero_mul i := zero_mul i
  mul_zero i := mul_zero i
  zero_le_one := nonneg'
  mul_le_mul_left i j h_ij k := by
    /-
      i j : ↑unitInterval
      h_ij : LE.le i j
      k : ↑unitInterval
      ⊢ LE.le (HMul.hMul k i) (HMul.hMul k j)
    -/
    simp only [← Subtype.coe_le_coe, coe_mul]
    /-
      i j : ↑unitInterval
      h_ij : LE.le i j
      k : ↑unitInterval
      ⊢ LE.le (HMul.hMul ↑k ↑i) (HMul.hMul ↑k ↑j)
    -/
    apply mul_le_mul le_rfl ?_ (nonneg i) (nonneg k)
    /-
      i j : ↑unitInterval
      h_ij : LE.le i j
      k : ↑unitInterval
      ⊢ LE.le ↑i ↑j
    -/
    simp [h_ij]
    /-
      🎉 no goals
    -/
  __ := inferInstanceAs (LinearOrder I)


/-- `Set.projIcc` is a contraction. -/
lemma _root_.Set.abs_projIcc_sub_projIcc : (|projIcc a b h c - projIcc a b h d| : α) ≤ |c - d| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c d : α
    h : LE.le a b
    ⊢ LE.le (abs (HSub.hSub ↑(Set.projIcc a b h c) ↑(Set.projIcc a b h d))) (abs ( …
  -/
  wlog hdc : d ≤ c generalizing c d
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a b c d : α
      h : LE.le a b
      this : ∀ {c d : α}, LE.le d c → LE.le (abs (HSub.hSub ↑(Set.projIcc a b h c) ↑ …
      hdc : Not (LE.le d c)
      ⊢ LE.le (abs (HSub.hSub ↑(Set.projIcc a b h c) ↑(Set.projIcc a b h d))) (abs ( …
    -/
  · rw [abs_sub_comm, abs_sub_comm c]; exact this (le_of_not_le hdc)
                                       /-
                                         🎉 no goals
                                       -/
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c✝ d✝ : α
    h : LE.le a b
    c d : α
    hdc : LE.le d c
    ⊢ LE.le (abs (HSub.hSub ↑(Set.projIcc a b h c) ↑(Set.projIcc a b h d))) (abs ( …
  -/
  rw [abs_eq_self.2 (sub_nonneg.2 hdc), abs_eq_self.2 (sub_nonneg.2 <| monotone_projIcc h hdc)]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c✝ d✝ : α
    h : LE.le a b
    c d : α
    hdc : LE.le d c
    ⊢ LE.le (HSub.hSub ((fun a_1 => ↑a_1) (Set.projIcc a b h c)) ((fun a_1 => ↑a_1 …
  -/
  rw [← sub_nonneg] at hdc
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c✝ d✝ : α
    h : LE.le a b
    c d : α
    hdc : LE.le 0 (HSub.hSub c d)
    ⊢ LE.le (HSub.hSub ((fun a_1 => ↑a_1) (Set.projIcc a b h c)) ((fun a_1 => ↑a_1 …
  -/
  refine (max_sub_max_le_max _ _ _ _).trans (max_le (by rwa [sub_self]) ?_)
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c✝ d✝ : α
    h : LE.le a b
    c d : α
    hdc : LE.le 0 (HSub.hSub c d)
    ⊢ LE.le (HSub.hSub (Min.min b c) (Min.min b d)) (HSub.hSub c d)
  -/
  refine ((le_abs_self _).trans <| abs_min_sub_min_le_max _ _ _ _).trans (max_le ?_ ?_)
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a b c✝ d✝ : α
      h : LE.le a b
      c d : α
      hdc : LE.le 0 (HSub.hSub c d)
      ⊢ LE.le (abs (HSub.hSub b b)) (HSub.hSub c d)
    -/
  · rwa [sub_self, abs_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a b c✝ d✝ : α
      h : LE.le a b
      c d : α
      hdc : LE.le 0 (HSub.hSub c d)
      ⊢ LE.le (abs (HSub.hSub c d)) (HSub.hSub c d)
    -/
  · exact (abs_eq_self.mpr hdc).le
    /-
      🎉 no goals
    -/


/-- When `h : a ≤ b` and `δ > 0`, `addNSMul h δ` is a sequence of points in the closed interval
  `[a,b]`, which is initially equally spaced but eventually stays at the right endpoint `b`. -/
def addNSMul (δ : α) (n : ℕ) : Icc a b := projIcc a b h (a + n • δ)


lemma addNSMul_zero : addNSMul h δ 0 = a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b : α
    h : LE.le a b
    δ : α
    ⊢ Eq (↑(Set.Icc.addNSMul h δ 0)) a
  -/
  rw [addNSMul, zero_smul, add_zero, projIcc_left]
  /-
    🎉 no goals
  -/


lemma addNSMul_eq_right [Archimedean α] (hδ : 0 < δ) :
    ∃ m, ∀ n ≥ m, addNSMul h δ n = b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    a b : α
    h : LE.le a b
    δ : α
    inst✝ : Archimedean α
    hδ : LT.lt 0 δ
    ⊢ Exists fun m => ∀ (n : Nat), GE.ge n m → Eq (↑(Set.Icc.addNSMul h δ n)) b
  -/
  obtain ⟨m, hm⟩ := Archimedean.arch (b - a) hδ
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    a b : α
    h : LE.le a b
    δ : α
    inst✝ : Archimedean α
    hδ : LT.lt 0 δ
    m : Nat
    hm : LE.le (HSub.hSub b a) (HSMul.hSMul m δ)
    ⊢ Exists fun m => ∀ (n : Nat), GE.ge n m → Eq (↑(Set.Icc.addNSMul h δ n)) b
  -/
  refine ⟨m, fun n hn ↦ ?_⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    a b : α
    h : LE.le a b
    δ : α
    inst✝ : Archimedean α
    hδ : LT.lt 0 δ
    m : Nat
    hm : LE.le (HSub.hSub b a) (HSMul.hSMul m δ)
    n : Nat
    hn : GE.ge n m
    ⊢ Eq (↑(Set.Icc.addNSMul h δ n)) b
  -/
  rw [addNSMul, coe_projIcc, add_comm, min_eq_left_iff.mpr, max_eq_right h]
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    a b : α
    h : LE.le a b
    δ : α
    inst✝ : Archimedean α
    hδ : LT.lt 0 δ
    m : Nat
    hm : LE.le (HSub.hSub b a) (HSMul.hSMul m δ)
    n : Nat
    hn : GE.ge n m
    ⊢ LE.le b (HAdd.hAdd (HSMul.hSMul n δ) a)
  -/
  exact sub_le_iff_le_add.mp (hm.trans <| nsmul_le_nsmul_left hδ.le hn)
  /-
    🎉 no goals
  -/


lemma monotone_addNSMul (hδ : 0 ≤ δ) : Monotone (addNSMul h δ) :=
  fun _ _ hnm ↦ monotone_projIcc h <| (add_le_add_iff_left _).mpr (nsmul_le_nsmul_left hδ hnm)


lemma abs_sub_addNSMul_le (hδ : 0 ≤ δ) {t : Icc a b} (n : ℕ)
    (ht : t ∈ Icc (addNSMul h δ n) (addNSMul h δ (n+1))) :
    (|t - addNSMul h δ n| : α) ≤ δ :=
  calc
    (|t - addNSMul h δ n| : α) = t - addNSMul h δ n            := abs_eq_self.2 <| sub_nonneg.2 ht.1
                                                             /-
                                                               α : Type u_1
                                                               inst✝ : LinearOrderedAddCommGroup α
                                                               a b : α
                                                               h : LE.le a b
                                                               δ : α
                                                               hδ : LE.le 0 δ
                                                               t : ↑(Set.Icc a b)
                                                               n : Nat
                                                               ht : Membership.mem (Set.Icc (Set.Icc.addNSMul h δ n) (Set.Icc.addNSMul h δ (H …
                                                               ⊢ LE.le (HSub.hSub ↑t ↑(Set.Icc.addNSMul h δ n)) (HSub.hSub ↑(Set.projIcc a b  …
                                                             -/
    _ ≤ projIcc a b h (a + (n+1) • δ) - addNSMul h δ n := by apply sub_le_sub_right; exact ht.2
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
    _ ≤ (|projIcc a b h (a + (n+1) • δ) - addNSMul h δ n| : α) := le_abs_self _
    _ ≤ |a + (n+1) • δ - (a + n • δ)|                          := abs_projIcc_sub_projIcc h
    _ ≤ δ := by
          /-
            α : Type u_1
            inst✝ : LinearOrderedAddCommGroup α
            a b : α
            h : LE.le a b
            δ : α
            hδ : LE.le 0 δ
            t : ↑(Set.Icc a b)
            n : Nat
            ht : Membership.mem (Set.Icc (Set.Icc.addNSMul h δ n) (Set.Icc.addNSMul h δ (H …
            ⊢ LE.le (abs (HSub.hSub (HAdd.hAdd a (HSMul.hSMul (HAdd.hAdd n 1) δ)) (HAdd.hA …
          -/
          rw [add_sub_add_comm, sub_self, zero_add, succ_nsmul', add_sub_cancel_right]
          /-
            α : Type u_1
            inst✝ : LinearOrderedAddCommGroup α
            a b : α
            h : LE.le a b
            δ : α
            hδ : LE.le 0 δ
            t : ↑(Set.Icc a b)
            n : Nat
            ht : Membership.mem (Set.Icc (Set.Icc.addNSMul h δ n) (Set.Icc.addNSMul h δ (H …
            ⊢ LE.le (abs δ) δ
          -/
          exact (abs_eq_self.mpr hδ).le
          /-
            🎉 no goals
          -/


/-- Any open cover `c` of a closed interval `[a, b]` in ℝ can be refined to
  a finite partition into subintervals. -/
lemma exists_monotone_Icc_subset_open_cover_Icc {ι} {a b : ℝ} (h : a ≤ b) {c : ι → Set (Icc a b)}
    (hc₁ : ∀ i, IsOpen (c i)) (hc₂ : univ ⊆ ⋃ i, c i) : ∃ t : ℕ → Icc a b, t 0 = a ∧
      Monotone t ∧ (∃ m, ∀ n ≥ m, t n = b) ∧ ∀ n, ∃ i, Icc (t n) (t (n + 1)) ⊆ c i := by
  /-
    ι : Sort u_1
    a b : Real
    h : LE.le a b
    c : ι → Set ↑(Set.Icc a b)
    hc₁ : ∀ (i : ι), IsOpen (c i)
    hc₂ : HasSubset.Subset Set.univ (Set.iUnion fun i => c i)
    ⊢ Exists fun t => And (Eq (↑(t 0)) a) (And (Monotone t) (And (Exists fun m =>  …
  -/
  obtain ⟨δ, δ_pos, ball_subset⟩ := lebesgue_number_lemma_of_metric isCompact_univ hc₁ hc₂
  /-
    case intro.intro
    ι : Sort u_1
    a b : Real
    h : LE.le a b
    c : ι → Set ↑(Set.Icc a b)
    hc₁ : ∀ (i : ι), IsOpen (c i)
    hc₂ : HasSubset.Subset Set.univ (Set.iUnion fun i => c i)
    δ : Real
    δ_pos : GT.gt δ 0
    ball_subset : ∀ (x : ↑(Set.Icc a b)), Membership.mem Set.univ x → Exists fun i …
    ⊢ Exists fun t => And (Eq (↑(t 0)) a) (And (Monotone t) (And (Exists fun m =>  …
  -/
  have hδ := half_pos δ_pos
  refine ⟨addNSMul h (δ/2), addNSMul_zero h,
    monotone_addNSMul h hδ.le, addNSMul_eq_right h hδ, fun n ↦ ?_⟩
  /-
    case intro.intro
    ι : Sort u_1
    a b : Real
    h : LE.le a b
    c : ι → Set ↑(Set.Icc a b)
    hc₁ : ∀ (i : ι), IsOpen (c i)
    hc₂ : HasSubset.Subset Set.univ (Set.iUnion fun i => c i)
    δ : Real
    δ_pos : GT.gt δ 0
    ball_subset : ∀ (x : ↑(Set.Icc a b)), Membership.mem Set.univ x → Exists fun i …
    hδ : LT.lt 0 (HDiv.hDiv δ 2)
    n : Nat
    ⊢ Exists fun i => HasSubset.Subset (Set.Icc (Set.Icc.addNSMul h (HDiv.hDiv δ 2 …
  -/
  obtain ⟨i, hsub⟩ := ball_subset (addNSMul h (δ/2) n) trivial
  /-
    case intro.intro.intro
    ι : Sort u_1
    a b : Real
    h : LE.le a b
    c : ι → Set ↑(Set.Icc a b)
    hc₁ : ∀ (i : ι), IsOpen (c i)
    hc₂ : HasSubset.Subset Set.univ (Set.iUnion fun i => c i)
    δ : Real
    δ_pos : GT.gt δ 0
    ball_subset : ∀ (x : ↑(Set.Icc a b)), Membership.mem Set.univ x → Exists fun i …
    hδ : LT.lt 0 (HDiv.hDiv δ 2)
    n : Nat
    i : ι
    hsub : HasSubset.Subset (Metric.ball (Set.Icc.addNSMul h (HDiv.hDiv δ 2) n) δ) …
    ⊢ Exists fun i => HasSubset.Subset (Set.Icc (Set.Icc.addNSMul h (HDiv.hDiv δ 2 …
  -/
  exact ⟨i, fun t ht ↦ hsub ((abs_sub_addNSMul_le h hδ.le n ht).trans_lt <| half_lt_self δ_pos)⟩
  /-
    🎉 no goals
  -/


/-- Any open cover of the unit interval can be refined to a finite partition into subintervals. -/
lemma exists_monotone_Icc_subset_open_cover_unitInterval {ι} {c : ι → Set I}
    (hc₁ : ∀ i, IsOpen (c i)) (hc₂ : univ ⊆ ⋃ i, c i) : ∃ t : ℕ → I, t 0 = 0 ∧
      Monotone t ∧ (∃ n, ∀ m ≥ n, t m = 1) ∧ ∀ n, ∃ i, Icc (t n) (t (n + 1)) ⊆ c i := by
  /-
    ι : Sort u_1
    c : ι → Set ↑unitInterval
    hc₁ : ∀ (i : ι), IsOpen (c i)
    hc₂ : HasSubset.Subset Set.univ (Set.iUnion fun i => c i)
    ⊢ Exists fun t => And (Eq (t 0) 0) (And (Monotone t) (And (Exists fun n => ∀ ( …
  -/
  simp_rw [← Subtype.coe_inj]
  /-
    ι : Sort u_1
    c : ι → Set ↑unitInterval
    hc₁ : ∀ (i : ι), IsOpen (c i)
    hc₂ : HasSubset.Subset Set.univ (Set.iUnion fun i => c i)
    ⊢ Exists fun t => And (Eq ↑(t 0) ↑0) (And (Monotone t) (And (Exists fun n => ∀ …
  -/
  exact exists_monotone_Icc_subset_open_cover_Icc zero_le_one hc₁ hc₂
  /-
    🎉 no goals
  -/


lemma exists_monotone_Icc_subset_open_cover_unitInterval_prod_self {ι} {c : ι → Set (I × I)}
    (hc₁ : ∀ i, IsOpen (c i)) (hc₂ : univ ⊆ ⋃ i, c i) :
    ∃ t : ℕ → I, t 0 = 0 ∧ Monotone t ∧ (∃ n, ∀ m ≥ n, t m = 1) ∧
      ∀ n m, ∃ i, Icc (t n) (t (n + 1)) ×ˢ Icc (t m) (t (m + 1)) ⊆ c i := by
  /-
    ι : Sort u_1
    c : ι → Set (Prod ↑unitInterval ↑unitInterval)
    hc₁ : ∀ (i : ι), IsOpen (c i)
    hc₂ : HasSubset.Subset Set.univ (Set.iUnion fun i => c i)
    ⊢ Exists fun t => And (Eq (t 0) 0) (And (Monotone t) (And (Exists fun n => ∀ ( …
  -/
  obtain ⟨δ, δ_pos, ball_subset⟩ := lebesgue_number_lemma_of_metric isCompact_univ hc₁ hc₂
  /-
    case intro.intro
    ι : Sort u_1
    c : ι → Set (Prod ↑unitInterval ↑unitInterval)
    hc₁ : ∀ (i : ι), IsOpen (c i)
    hc₂ : HasSubset.Subset Set.univ (Set.iUnion fun i => c i)
    δ : Real
    δ_pos : GT.gt δ 0
    ball_subset : ∀ (x : Prod ↑unitInterval ↑unitInterval), Membership.mem Set.uni …
    ⊢ Exists fun t => And (Eq (t 0) 0) (And (Monotone t) (And (Exists fun n => ∀ ( …
  -/
  have hδ := half_pos δ_pos
  /-
    case intro.intro
    ι : Sort u_1
    c : ι → Set (Prod ↑unitInterval ↑unitInterval)
    hc₁ : ∀ (i : ι), IsOpen (c i)
    hc₂ : HasSubset.Subset Set.univ (Set.iUnion fun i => c i)
    δ : Real
    δ_pos : GT.gt δ 0
    ball_subset : ∀ (x : Prod ↑unitInterval ↑unitInterval), Membership.mem Set.uni …
    hδ : LT.lt 0 (HDiv.hDiv δ 2)
    ⊢ Exists fun t => And (Eq (t 0) 0) (And (Monotone t) (And (Exists fun n => ∀ ( …
  -/
  simp_rw [Subtype.ext_iff]
  /-
    case intro.intro
    ι : Sort u_1
    c : ι → Set (Prod ↑unitInterval ↑unitInterval)
    hc₁ : ∀ (i : ι), IsOpen (c i)
    hc₂ : HasSubset.Subset Set.univ (Set.iUnion fun i => c i)
    δ : Real
    δ_pos : GT.gt δ 0
    ball_subset : ∀ (x : Prod ↑unitInterval ↑unitInterval), Membership.mem Set.uni …
    hδ : LT.lt 0 (HDiv.hDiv δ 2)
    ⊢ Exists fun t => And (Eq ↑(t 0) ↑0) (And (Monotone t) (And (Exists fun n => ∀ …
  -/
  have h : (0 : ℝ) ≤ 1 := zero_le_one
  refine ⟨addNSMul h (δ/2), addNSMul_zero h,
    monotone_addNSMul h hδ.le, addNSMul_eq_right h hδ, fun n m ↦ ?_⟩
  /-
    case intro.intro
    ι : Sort u_1
    c : ι → Set (Prod ↑unitInterval ↑unitInterval)
    hc₁ : ∀ (i : ι), IsOpen (c i)
    hc₂ : HasSubset.Subset Set.univ (Set.iUnion fun i => c i)
    δ : Real
    δ_pos : GT.gt δ 0
    ball_subset : ∀ (x : Prod ↑unitInterval ↑unitInterval), Membership.mem Set.uni …
    hδ : LT.lt 0 (HDiv.hDiv δ 2)
    h : LE.le 0 1
    n m : Nat
    ⊢ Exists fun i => HasSubset.Subset (SProd.sprod (Set.Icc (Set.Icc.addNSMul h ( …
  -/
  obtain ⟨i, hsub⟩ := ball_subset (addNSMul h (δ/2) n, addNSMul h (δ/2) m) trivial
  exact ⟨i, fun t ht ↦ hsub (Metric.mem_ball.mpr <| (max_le (abs_sub_addNSMul_le h hδ.le n ht.1) <|
    abs_sub_addNSMul_le h hδ.le m ht.2).trans_lt <| half_lt_self δ_pos)⟩


@[simp]
theorem projIcc_eq_zero {x : ℝ} : projIcc (0 : ℝ) 1 zero_le_one x = 0 ↔ x ≤ 0 :=
  projIcc_eq_left zero_lt_one


@[simp]
theorem projIcc_eq_one {x : ℝ} : projIcc (0 : ℝ) 1 zero_le_one x = 1 ↔ 1 ≤ x :=
  projIcc_eq_right zero_lt_one


/-- A tactic that solves `0 ≤ ↑x`, `0 ≤ 1 - ↑x`, `↑x ≤ 1`, and `1 - ↑x ≤ 1` for `x : I`. -/
macro "unit_interval" : tactic =>
  `(tactic| (first
  | apply unitInterval.nonneg
  | apply unitInterval.one_minus_nonneg
  | apply unitInterval.le_one
  | apply unitInterval.one_minus_le_one))


/-- The image of `[0,1]` under the homeomorphism `fun x ↦ a * x + b` is `[b, a+b]`.
-/
theorem affineHomeomorph_image_I (a b : 𝕜) (h : 0 < a) :
                                                                            /-
                                                                              𝕜 : Type u_1
                                                                              inst✝² : LinearOrderedField 𝕜
                                                                              inst✝¹ : TopologicalSpace 𝕜
                                                                              inst✝ : TopologicalRing 𝕜
                                                                              a b : 𝕜
                                                                              h : LT.lt 0 a
                                                                              ⊢ Eq (Set.image (⇑(affineHomeomorph a b ⋯)) (Set.Icc 0 1)) (Set.Icc b (HAdd.hA …
                                                                            -/
    affineHomeomorph a b h.ne.symm '' Set.Icc 0 1 = Set.Icc b (a + b) := by simp [h]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- The affine homeomorphism from a nontrivial interval `[a,b]` to `[0,1]`.
-/
def iccHomeoI (a b : 𝕜) (h : a < b) : Set.Icc a b ≃ₜ Set.Icc (0 : 𝕜) (1 : 𝕜) := by
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : TopologicalRing 𝕜
    a b : 𝕜
    h : LT.lt a b
    ⊢ Homeomorph ↑(Set.Icc a b) ↑(Set.Icc 0 1)
  -/
  let e := Homeomorph.image (affineHomeomorph (b - a) a (sub_pos.mpr h).ne.symm) (Set.Icc 0 1)
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : TopologicalRing 𝕜
    a b : 𝕜
    h : LT.lt a b
    e : Homeomorph ↑(Set.Icc 0 1) ↑(Set.image (⇑(affineHomeomorph (HSub.hSub b a)  …
    ⊢ Homeomorph ↑(Set.Icc a b) ↑(Set.Icc 0 1)
  -/
  refine (e.trans ?_).symm
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : TopologicalRing 𝕜
    a b : 𝕜
    h : LT.lt a b
    e : Homeomorph ↑(Set.Icc 0 1) ↑(Set.image (⇑(affineHomeomorph (HSub.hSub b a)  …
    ⊢ Homeomorph ↑(Set.image (⇑(affineHomeomorph (HSub.hSub b a) a ⋯)) (Set.Icc 0  …
  -/
  apply Homeomorph.setCongr
  /-
    case h
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : TopologicalRing 𝕜
    a b : 𝕜
    h : LT.lt a b
    e : Homeomorph ↑(Set.Icc 0 1) ↑(Set.image (⇑(affineHomeomorph (HSub.hSub b a)  …
    ⊢ Eq (Set.image (⇑(affineHomeomorph (HSub.hSub b a) a ⋯)) (Set.Icc 0 1)) (Set. …
  -/
  rw [affineHomeomorph_image_I _ _ (sub_pos.2 h)]
  /-
    case h
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : TopologicalRing 𝕜
    a b : 𝕜
    h : LT.lt a b
    e : Homeomorph ↑(Set.Icc 0 1) ↑(Set.image (⇑(affineHomeomorph (HSub.hSub b a)  …
    ⊢ Eq (Set.Icc a (HAdd.hAdd (HSub.hSub b a) a)) (Set.Icc a b)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem iccHomeoI_apply_coe (a b : 𝕜) (h : a < b) (x : Set.Icc a b) :
    ((iccHomeoI a b h) x : 𝕜) = (x - a) / (b - a) :=
  rfl


@[simp]
theorem iccHomeoI_symm_apply_coe (a b : 𝕜) (h : a < b) (x : Set.Icc (0 : 𝕜) (1 : 𝕜)) :
    ((iccHomeoI a b h).symm x : 𝕜) = (b - a) * x + a :=
  rfl


