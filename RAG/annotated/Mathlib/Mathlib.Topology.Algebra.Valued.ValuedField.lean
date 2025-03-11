theorem Valuation.inversion_estimate {x y : K} {γ : Γ₀ˣ} (y_ne : y ≠ 0)
    (h : v (x - y) < min (γ * (v y * v y)) (v y)) : v (x⁻¹ - y⁻¹) < γ := by
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    x y : K
    γ : Units Γ₀
    y_ne : Ne y 0
    h : LT.lt (v (HSub.hSub x y)) (Min.min (HMul.hMul (↑γ) (HMul.hMul (v y) (v y)) …
    ⊢ LT.lt (v (HSub.hSub (Inv.inv x) (Inv.inv y))) ↑γ
  -/
  have hyp1 : v (x - y) < γ * (v y * v y) := lt_of_lt_of_le h (min_le_left _ _)
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    x y : K
    γ : Units Γ₀
    y_ne : Ne y 0
    h : LT.lt (v (HSub.hSub x y)) (Min.min (HMul.hMul (↑γ) (HMul.hMul (v y) (v y)) …
    hyp1 : LT.lt (v (HSub.hSub x y)) (HMul.hMul (↑γ) (HMul.hMul (v y) (v y)))
    ⊢ LT.lt (v (HSub.hSub (Inv.inv x) (Inv.inv y))) ↑γ
  -/
  have hyp1' : v (x - y) * (v y * v y)⁻¹ < γ := mul_inv_lt_of_lt_mul₀ hyp1
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    x y : K
    γ : Units Γ₀
    y_ne : Ne y 0
    h : LT.lt (v (HSub.hSub x y)) (Min.min (HMul.hMul (↑γ) (HMul.hMul (v y) (v y)) …
    hyp1 : LT.lt (v (HSub.hSub x y)) (HMul.hMul (↑γ) (HMul.hMul (v y) (v y)))
    hyp1' : LT.lt (HMul.hMul (v (HSub.hSub x y)) (Inv.inv (HMul.hMul (v y) (v y))) …
    ⊢ LT.lt (v (HSub.hSub (Inv.inv x) (Inv.inv y))) ↑γ
  -/
  have hyp2 : v (x - y) < v y := lt_of_lt_of_le h (min_le_right _ _)
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    x y : K
    γ : Units Γ₀
    y_ne : Ne y 0
    h : LT.lt (v (HSub.hSub x y)) (Min.min (HMul.hMul (↑γ) (HMul.hMul (v y) (v y)) …
    hyp1 : LT.lt (v (HSub.hSub x y)) (HMul.hMul (↑γ) (HMul.hMul (v y) (v y)))
    hyp1' : LT.lt (HMul.hMul (v (HSub.hSub x y)) (Inv.inv (HMul.hMul (v y) (v y))) …
    hyp2 : LT.lt (v (HSub.hSub x y)) (v y)
    ⊢ LT.lt (v (HSub.hSub (Inv.inv x) (Inv.inv y))) ↑γ
  -/
  have key : v x = v y := Valuation.map_eq_of_sub_lt v hyp2
  have x_ne : x ≠ 0 := by
    intro h
    apply y_ne
    rw [h, v.map_zero] at key
    exact v.zero_iff.1 key.symm
  have decomp : x⁻¹ - y⁻¹ = x⁻¹ * (y - x) * y⁻¹ := by
    rw [mul_sub_left_distrib, sub_mul, mul_assoc, show y * y⁻¹ = 1 from mul_inv_cancel₀ y_ne,
      show x⁻¹ * x = 1 from inv_mul_cancel₀ x_ne, mul_one, one_mul]
  calc
    v (x⁻¹ - y⁻¹) = v (x⁻¹ * (y - x) * y⁻¹) := by rw [decomp]
    _ = v x⁻¹ * (v <| y - x) * v y⁻¹ := by repeat' rw [Valuation.map_mul]
    _ = (v x)⁻¹ * (v <| y - x) * (v y)⁻¹ := by rw [map_inv₀, map_inv₀]
    _ = (v <| y - x) * (v y * v y)⁻¹ := by rw [mul_assoc, mul_comm, key, mul_assoc, mul_inv_rev]
    _ = (v <| y - x) * (v y * v y)⁻¹ := rfl
    _ = (v <| x - y) * (v y * v y)⁻¹ := by rw [Valuation.map_sub_swap]
    _ < γ := hyp1'


/-- The topology coming from a valuation on a division ring makes it a topological division ring
    [BouAC, VI.5.1 middle of Proposition 1] -/
instance (priority := 100) Valued.topologicalDivisionRing [Valued K Γ₀] :
    TopologicalDivisionRing K :=
        /-
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_2
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : Valued K Γ₀
          ⊢ TopologicalRing K
        -/
  { (by infer_instance : TopologicalRing K) with
        /-
          🎉 no goals
        -/
    continuousAt_inv₀ := by
      /-
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        ⊢ ∀ ⦃x : K⦄, Ne x 0 → ContinuousAt Inv.inv x
      -/
      intro x x_ne s s_in
      /-
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        x : K
        x_ne : Ne x 0
        s : Set K
        s_in : Membership.mem (nhds (Inv.inv x)) s
        ⊢ Membership.mem (Filter.map Inv.inv (nhds x)) s
      -/
      cases' Valued.mem_nhds.mp s_in with γ hs; clear s_in
      /-
        case intro
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        x : K
        x_ne : Ne x 0
        s : Set K
        γ : Units Γ₀
        hs : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x) …
        ⊢ Membership.mem (Filter.map Inv.inv (nhds x)) s
      -/
      rw [mem_map, Valued.mem_nhds]
      /-
        case intro
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        x : K
        x_ne : Ne x 0
        s : Set K
        γ : Units Γ₀
        hs : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x) …
        ⊢ Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub  …
      -/
      change ∃ γ : Γ₀ˣ, { y : K | (v (y - x) : Γ₀) < γ } ⊆ { x : K | x⁻¹ ∈ s }
      /-
        case intro
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        x : K
        x_ne : Ne x 0
        s : Set K
        γ : Units Γ₀
        hs : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x) …
        ⊢ Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub  …
      -/
      have vx_ne := (Valuation.ne_zero_iff <| v).mpr x_ne
      /-
        case intro
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        x : K
        x_ne : Ne x 0
        s : Set K
        γ : Units Γ₀
        hs : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x) …
        vx_ne : Ne (Valued.v x) 0
        ⊢ Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub  …
      -/
      let γ' := Units.mk0 _ vx_ne
      /-
        case intro
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        x : K
        x_ne : Ne x 0
        s : Set K
        γ : Units Γ₀
        hs : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x) …
        vx_ne : Ne (Valued.v x) 0
        γ' : Units Γ₀ := Units.mk0 (Valued.v x) vx_ne
        ⊢ Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub  …
      -/
      use min (γ * (γ' * γ')) γ'
      /-
        case h
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        x : K
        x_ne : Ne x 0
        s : Set K
        γ : Units Γ₀
        hs : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x) …
        vx_ne : Ne (Valued.v x) 0
        γ' : Units Γ₀ := Units.mk0 (Valued.v x) vx_ne
        ⊢ HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min  …
      -/
      intro y y_in
      /-
        case h
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        x : K
        x_ne : Ne x 0
        s : Set K
        γ : Units Γ₀
        hs : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x) …
        vx_ne : Ne (Valued.v x) 0
        γ' : Units Γ₀ := Units.mk0 (Valued.v x) vx_ne
        y : K
        y_in : Membership.mem (setOf fun y => LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.m …
        ⊢ Membership.mem (setOf fun x => Membership.mem s (Inv.inv x)) y
      -/
      apply hs
      /-
        case h.a
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        x : K
        x_ne : Ne x 0
        s : Set K
        γ : Units Γ₀
        hs : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x) …
        vx_ne : Ne (Valued.v x) 0
        γ' : Units Γ₀ := Units.mk0 (Valued.v x) vx_ne
        y : K
        y_in : Membership.mem (setOf fun y => LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.m …
        ⊢ Membership.mem (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x))) ↑γ …
      -/
      simp only [mem_setOf_eq] at y_in
      /-
        case h.a
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        x : K
        x_ne : Ne x 0
        s : Set K
        γ : Units Γ₀
        hs : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x) …
        vx_ne : Ne (Valued.v x) 0
        γ' : Units Γ₀ := Units.mk0 (Valued.v x) vx_ne
        y : K
        y_in : LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min (HMul.hMul γ (HMul.hMul γ' γ …
        ⊢ Membership.mem (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x))) ↑γ …
      -/
      rw [Units.min_val, Units.val_mul, Units.val_mul] at y_in
      /-
        case h.a
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_2
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : Valued K Γ₀
        x : K
        x_ne : Ne x 0
        s : Set K
        γ : Units Γ₀
        hs : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x) …
        vx_ne : Ne (Valued.v x) 0
        γ' : Units Γ₀ := Units.mk0 (Valued.v x) vx_ne
        y : K
        y_in : LT.lt (Valued.v (HSub.hSub y x)) (Min.min (HMul.hMul (↑γ) (HMul.hMul ↑γ …
        ⊢ Membership.mem (setOf fun y => LT.lt (Valued.v (HSub.hSub y (Inv.inv x))) ↑γ …
      -/
      exact Valuation.inversion_estimate _ x_ne y_in }
      /-
        🎉 no goals
      -/


/-- A valued division ring is separated. -/
instance (priority := 100) ValuedRing.separated [Valued K Γ₀] : T0Space K := by
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : Valued K Γ₀
    ⊢ T0Space K
  -/
  suffices T2Space K by infer_instance
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : Valued K Γ₀
    ⊢ T2Space K
  -/
  apply TopologicalAddGroup.t2Space_of_zero_sep
  /-
    case H
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : Valued K Γ₀
    ⊢ ∀ (x : K), Ne x 0 → Exists fun U => And (Membership.mem (nhds 0) U) (Not (Me …
  -/
  intro x x_ne
  /-
    case H
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : Valued K Γ₀
    x : K
    x_ne : Ne x 0
    ⊢ Exists fun U => And (Membership.mem (nhds 0) U) (Not (Membership.mem U x))
  -/
  refine ⟨{ k | v k < v x }, ?_, fun h => lt_irrefl _ h⟩
  /-
    case H
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : Valued K Γ₀
    x : K
    x_ne : Ne x 0
    ⊢ Membership.mem (nhds 0) (setOf fun k => LT.lt (Valued.v k) (Valued.v x))
  -/
  rw [Valued.mem_nhds]
  /-
    case H
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : Valued K Γ₀
    x : K
    x_ne : Ne x 0
    ⊢ Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub  …
  -/
  have vx_ne := (Valuation.ne_zero_iff <| v).mpr x_ne
  /-
    case H
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : Valued K Γ₀
    x : K
    x_ne : Ne x 0
    vx_ne : Ne (Valued.v x) 0
    ⊢ Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub  …
  -/
  let γ' := Units.mk0 _ vx_ne
  /-
    case H
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : Valued K Γ₀
    x : K
    x_ne : Ne x 0
    vx_ne : Ne (Valued.v x) 0
    γ' : Units Γ₀ := Units.mk0 (Valued.v x) vx_ne
    ⊢ Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub  …
  -/
  exact ⟨γ', fun y hy => by simpa using hy⟩
  /-
    🎉 no goals
  -/


theorem Valued.continuous_valuation [Valued K Γ₀] : Continuous (v : K → Γ₀) := by
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : Valued K Γ₀
    ⊢ Continuous ⇑Valued.v
  -/
  rw [continuous_iff_continuousAt]
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : Valued K Γ₀
    ⊢ ∀ (x : K), ContinuousAt (⇑Valued.v) x
  -/
  intro x
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : Valued K Γ₀
    x : K
    ⊢ ContinuousAt (⇑Valued.v) x
  -/
  rcases eq_or_ne x 0 with (rfl | h)
    /-
      case inl
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_2
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : Valued K Γ₀
      ⊢ ContinuousAt (⇑Valued.v) 0
    -/
  · rw [ContinuousAt, map_zero, WithZeroTopology.tendsto_zero]
    /-
      case inl
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_2
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : Valued K Γ₀
      ⊢ ∀ (γ₀ : Γ₀), Ne γ₀ 0 → Filter.Eventually (fun x => LT.lt (Valued.v x) γ₀) (n …
    -/
    intro γ hγ
    /-
      case inl
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_2
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : Valued K Γ₀
      γ : Γ₀
      hγ : Ne γ 0
      ⊢ Filter.Eventually (fun x => LT.lt (Valued.v x) γ) (nhds 0)
    -/
    rw [Filter.Eventually, Valued.mem_nhds_zero]
    /-
      case inl
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_2
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : Valued K Γ₀
      γ : Γ₀
      hγ : Ne γ 0
      ⊢ Exists fun γ_1 => HasSubset.Subset (setOf fun x => LT.lt (Valued.v x) ↑γ_1)  …
    -/
    use Units.mk0 γ hγ; rfl
                        /-
                          🎉 no goals
                        -/
    /-
      case inr
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_2
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : Valued K Γ₀
      x : K
      h : Ne x 0
      ⊢ ContinuousAt (⇑Valued.v) x
    -/
  · have v_ne : (v x : Γ₀) ≠ 0 := (Valuation.ne_zero_iff _).mpr h
    /-
      case inr
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_2
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : Valued K Γ₀
      x : K
      h : Ne x 0
      v_ne : Ne (Valued.v x) 0
      ⊢ ContinuousAt (⇑Valued.v) x
    -/
    rw [ContinuousAt, WithZeroTopology.tendsto_of_ne_zero v_ne]
    /-
      case inr
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_2
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : Valued K Γ₀
      x : K
      h : Ne x 0
      v_ne : Ne (Valued.v x) 0
      ⊢ Filter.Eventually (fun x_1 => Eq (Valued.v x_1) (Valued.v x)) (nhds x)
    -/
    apply Valued.loc_const v_ne
    /-
      🎉 no goals
    -/


local notation "hat " => Completion


/-- A valued field is completable. -/
instance (priority := 100) completable : CompletableTopField K :=
  { ValuedRing.separated with
    nice := by
      /-
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        ⊢ ∀ (F : Filter K), Cauchy F → Eq (Min.min (nhds 0) F) Bot.bot → Cauchy (Filte …
      -/
      rintro F hF h0
      have : ∃ γ₀ : Γ₀ˣ, ∃ M ∈ F, ∀ x ∈ M, (γ₀ : Γ₀) ≤ v x := by
        rcases Filter.inf_eq_bot_iff.mp h0 with ⟨U, U_in, M, M_in, H⟩
        rcases Valued.mem_nhds_zero.mp U_in with ⟨γ₀, hU⟩
        exists γ₀, M, M_in
        intro x xM
        apply le_of_not_lt _
        intro hyp
        have : x ∈ U ∩ M := ⟨hU hyp, xM⟩
        rwa [H] at this
      /-
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        F : Filter K
        hF : Cauchy F
        h0 : Eq (Min.min (nhds 0) F) Bot.bot
        this : Exists fun γ₀ => Exists fun M => And (Membership.mem F M) (∀ (x : K), M …
        ⊢ Cauchy (Filter.map (fun x => Inv.inv x) F)
      -/
      rcases this with ⟨γ₀, M₀, M₀_in, H₀⟩
      /-
        case intro.intro.intro
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        F : Filter K
        hF : Cauchy F
        h0 : Eq (Min.min (nhds 0) F) Bot.bot
        γ₀ : Units Γ₀
        M₀ : Set K
        M₀_in : Membership.mem F M₀
        H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
        ⊢ Cauchy (Filter.map (fun x => Inv.inv x) F)
      -/
      rw [Valued.cauchy_iff] at hF ⊢
      /-
        case intro.intro.intro
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        F : Filter K
        hF : And F.NeBot (∀ (γ : Units Γ₀), Exists fun M => And (Membership.mem F M) ( …
        h0 : Eq (Min.min (nhds 0) F) Bot.bot
        γ₀ : Units Γ₀
        M₀ : Set K
        M₀_in : Membership.mem F M₀
        H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
        ⊢ And (Filter.map (fun x => Inv.inv x) F).NeBot (∀ (γ : Units Γ₀), Exists fun  …
      -/
      refine ⟨hF.1.map _, ?_⟩
      /-
        case intro.intro.intro
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        F : Filter K
        hF : And F.NeBot (∀ (γ : Units Γ₀), Exists fun M => And (Membership.mem F M) ( …
        h0 : Eq (Min.min (nhds 0) F) Bot.bot
        γ₀ : Units Γ₀
        M₀ : Set K
        M₀_in : Membership.mem F M₀
        H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
        ⊢ ∀ (γ : Units Γ₀), Exists fun M => And (Membership.mem (Filter.map (fun x =>  …
      -/
      replace hF := hF.2
      /-
        case intro.intro.intro
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        F : Filter K
        h0 : Eq (Min.min (nhds 0) F) Bot.bot
        γ₀ : Units Γ₀
        M₀ : Set K
        M₀_in : Membership.mem F M₀
        H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
        hF : ∀ (γ : Units Γ₀), Exists fun M => And (Membership.mem F M) (∀ (x : K), Me …
        ⊢ ∀ (γ : Units Γ₀), Exists fun M => And (Membership.mem (Filter.map (fun x =>  …
      -/
      intro γ
      /-
        case intro.intro.intro
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        F : Filter K
        h0 : Eq (Min.min (nhds 0) F) Bot.bot
        γ₀ : Units Γ₀
        M₀ : Set K
        M₀_in : Membership.mem F M₀
        H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
        hF : ∀ (γ : Units Γ₀), Exists fun M => And (Membership.mem F M) (∀ (x : K), Me …
        γ : Units Γ₀
        ⊢ Exists fun M => And (Membership.mem (Filter.map (fun x => Inv.inv x) F) M) ( …
      -/
      rcases hF (min (γ * γ₀ * γ₀) γ₀) with ⟨M₁, M₁_in, H₁⟩
      /-
        case intro.intro.intro.intro.intro
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        F : Filter K
        h0 : Eq (Min.min (nhds 0) F) Bot.bot
        γ₀ : Units Γ₀
        M₀ : Set K
        M₀_in : Membership.mem F M₀
        H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
        hF : ∀ (γ : Units Γ₀), Exists fun M => And (Membership.mem F M) (∀ (x : K), Me …
        γ : Units Γ₀
        M₁ : Set K
        M₁_in : Membership.mem F M₁
        H₁ : ∀ (x : K), Membership.mem M₁ x → ∀ (y : K), Membership.mem M₁ y → LT.lt ( …
        ⊢ Exists fun M => And (Membership.mem (Filter.map (fun x => Inv.inv x) F) M) ( …
      -/
      clear hF
      /-
        case intro.intro.intro.intro.intro
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        F : Filter K
        h0 : Eq (Min.min (nhds 0) F) Bot.bot
        γ₀ : Units Γ₀
        M₀ : Set K
        M₀_in : Membership.mem F M₀
        H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
        γ : Units Γ₀
        M₁ : Set K
        M₁_in : Membership.mem F M₁
        H₁ : ∀ (x : K), Membership.mem M₁ x → ∀ (y : K), Membership.mem M₁ y → LT.lt ( …
        ⊢ Exists fun M => And (Membership.mem (Filter.map (fun x => Inv.inv x) F) M) ( …
      -/
      use (fun x : K => x⁻¹) '' (M₀ ∩ M₁)
      /-
        case h
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        F : Filter K
        h0 : Eq (Min.min (nhds 0) F) Bot.bot
        γ₀ : Units Γ₀
        M₀ : Set K
        M₀_in : Membership.mem F M₀
        H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
        γ : Units Γ₀
        M₁ : Set K
        M₁_in : Membership.mem F M₁
        H₁ : ∀ (x : K), Membership.mem M₁ x → ∀ (y : K), Membership.mem M₁ y → LT.lt ( …
        ⊢ And (Membership.mem (Filter.map (fun x => Inv.inv x) F) (Set.image (fun x => …
      -/
      constructor
        /-
          case h.left
          K : Type u_1
          inst✝¹ : Field K
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          hv : Valued K Γ₀
          F : Filter K
          h0 : Eq (Min.min (nhds 0) F) Bot.bot
          γ₀ : Units Γ₀
          M₀ : Set K
          M₀_in : Membership.mem F M₀
          H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
          γ : Units Γ₀
          M₁ : Set K
          M₁_in : Membership.mem F M₁
          H₁ : ∀ (x : K), Membership.mem M₁ x → ∀ (y : K), Membership.mem M₁ y → LT.lt ( …
          ⊢ Membership.mem (Filter.map (fun x => Inv.inv x) F) (Set.image (fun x => Inv. …
        -/
      · rw [mem_map]
        /-
          case h.left
          K : Type u_1
          inst✝¹ : Field K
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          hv : Valued K Γ₀
          F : Filter K
          h0 : Eq (Min.min (nhds 0) F) Bot.bot
          γ₀ : Units Γ₀
          M₀ : Set K
          M₀_in : Membership.mem F M₀
          H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
          γ : Units Γ₀
          M₁ : Set K
          M₁_in : Membership.mem F M₁
          H₁ : ∀ (x : K), Membership.mem M₁ x → ∀ (y : K), Membership.mem M₁ y → LT.lt ( …
          ⊢ Membership.mem F (Set.preimage (fun x => Inv.inv x) (Set.image (fun x => Inv …
        -/
        apply mem_of_superset (Filter.inter_mem M₀_in M₁_in)
        /-
          case h.left
          K : Type u_1
          inst✝¹ : Field K
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          hv : Valued K Γ₀
          F : Filter K
          h0 : Eq (Min.min (nhds 0) F) Bot.bot
          γ₀ : Units Γ₀
          M₀ : Set K
          M₀_in : Membership.mem F M₀
          H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
          γ : Units Γ₀
          M₁ : Set K
          M₁_in : Membership.mem F M₁
          H₁ : ∀ (x : K), Membership.mem M₁ x → ∀ (y : K), Membership.mem M₁ y → LT.lt ( …
          ⊢ HasSubset.Subset (Inter.inter M₀ M₁) (Set.preimage (fun x => Inv.inv x) (Set …
        -/
        exact subset_preimage_image _ _
        /-
          🎉 no goals
        -/
        /-
          case h.right
          K : Type u_1
          inst✝¹ : Field K
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          hv : Valued K Γ₀
          F : Filter K
          h0 : Eq (Min.min (nhds 0) F) Bot.bot
          γ₀ : Units Γ₀
          M₀ : Set K
          M₀_in : Membership.mem F M₀
          H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
          γ : Units Γ₀
          M₁ : Set K
          M₁_in : Membership.mem F M₁
          H₁ : ∀ (x : K), Membership.mem M₁ x → ∀ (y : K), Membership.mem M₁ y → LT.lt ( …
          ⊢ ∀ (x : K), Membership.mem (Set.image (fun x => Inv.inv x) (Inter.inter M₀ M₁ …
        -/
      · rintro _ ⟨x, ⟨x_in₀, x_in₁⟩, rfl⟩ _ ⟨y, ⟨_, y_in₁⟩, rfl⟩
        /-
          case h.right.intro.intro.intro.intro.intro.intro
          K : Type u_1
          inst✝¹ : Field K
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          hv : Valued K Γ₀
          F : Filter K
          h0 : Eq (Min.min (nhds 0) F) Bot.bot
          γ₀ : Units Γ₀
          M₀ : Set K
          M₀_in : Membership.mem F M₀
          H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
          γ : Units Γ₀
          M₁ : Set K
          M₁_in : Membership.mem F M₁
          H₁ : ∀ (x : K), Membership.mem M₁ x → ∀ (y : K), Membership.mem M₁ y → LT.lt ( …
          x : K
          x_in₀ : Membership.mem M₀ x
          x_in₁ : Membership.mem M₁ x
          y : K
          left✝ : Membership.mem M₀ y
          y_in₁ : Membership.mem M₁ y
          ⊢ LT.lt (Valued.v (HSub.hSub ((fun x => Inv.inv x) y) ((fun x => Inv.inv x) x) …
        -/
        simp only [mem_setOf_eq]
        /-
          case h.right.intro.intro.intro.intro.intro.intro
          K : Type u_1
          inst✝¹ : Field K
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          hv : Valued K Γ₀
          F : Filter K
          h0 : Eq (Min.min (nhds 0) F) Bot.bot
          γ₀ : Units Γ₀
          M₀ : Set K
          M₀_in : Membership.mem F M₀
          H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
          γ : Units Γ₀
          M₁ : Set K
          M₁_in : Membership.mem F M₁
          H₁ : ∀ (x : K), Membership.mem M₁ x → ∀ (y : K), Membership.mem M₁ y → LT.lt ( …
          x : K
          x_in₀ : Membership.mem M₀ x
          x_in₁ : Membership.mem M₁ x
          y : K
          left✝ : Membership.mem M₀ y
          y_in₁ : Membership.mem M₁ y
          ⊢ LT.lt (Valued.v (HSub.hSub (Inv.inv y) (Inv.inv x))) ↑γ
        -/
        specialize H₁ x x_in₁ y y_in₁
        /-
          case h.right.intro.intro.intro.intro.intro.intro
          K : Type u_1
          inst✝¹ : Field K
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          hv : Valued K Γ₀
          F : Filter K
          h0 : Eq (Min.min (nhds 0) F) Bot.bot
          γ₀ : Units Γ₀
          M₀ : Set K
          M₀_in : Membership.mem F M₀
          H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
          γ : Units Γ₀
          M₁ : Set K
          M₁_in : Membership.mem F M₁
          x : K
          x_in₀ : Membership.mem M₀ x
          x_in₁ : Membership.mem M₁ x
          y : K
          left✝ : Membership.mem M₀ y
          y_in₁ : Membership.mem M₁ y
          H₁ : LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min (HMul.hMul (HMul.hMul γ γ₀) γ₀ …
          ⊢ LT.lt (Valued.v (HSub.hSub (Inv.inv y) (Inv.inv x))) ↑γ
        -/
        replace x_in₀ := H₀ x x_in₀
        /-
          case h.right.intro.intro.intro.intro.intro.intro
          K : Type u_1
          inst✝¹ : Field K
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          hv : Valued K Γ₀
          F : Filter K
          h0 : Eq (Min.min (nhds 0) F) Bot.bot
          γ₀ : Units Γ₀
          M₀ : Set K
          M₀_in : Membership.mem F M₀
          H₀ : ∀ (x : K), Membership.mem M₀ x → LE.le (↑γ₀) (Valued.v x)
          γ : Units Γ₀
          M₁ : Set K
          M₁_in : Membership.mem F M₁
          x : K
          x_in₁ : Membership.mem M₁ x
          y : K
          left✝ : Membership.mem M₀ y
          y_in₁ : Membership.mem M₁ y
          H₁ : LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min (HMul.hMul (HMul.hMul γ γ₀) γ₀ …
          x_in₀ : LE.le (↑γ₀) (Valued.v x)
          ⊢ LT.lt (Valued.v (HSub.hSub (Inv.inv y) (Inv.inv x))) ↑γ
        -/
        clear H₀
        /-
          case h.right.intro.intro.intro.intro.intro.intro
          K : Type u_1
          inst✝¹ : Field K
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          hv : Valued K Γ₀
          F : Filter K
          h0 : Eq (Min.min (nhds 0) F) Bot.bot
          γ₀ : Units Γ₀
          M₀ : Set K
          M₀_in : Membership.mem F M₀
          γ : Units Γ₀
          M₁ : Set K
          M₁_in : Membership.mem F M₁
          x : K
          x_in₁ : Membership.mem M₁ x
          y : K
          left✝ : Membership.mem M₀ y
          y_in₁ : Membership.mem M₁ y
          H₁ : LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min (HMul.hMul (HMul.hMul γ γ₀) γ₀ …
          x_in₀ : LE.le (↑γ₀) (Valued.v x)
          ⊢ LT.lt (Valued.v (HSub.hSub (Inv.inv y) (Inv.inv x))) ↑γ
        -/
        apply Valuation.inversion_estimate
        · have : (v x : Γ₀) ≠ 0 := by
            intro h
            rw [h] at x_in₀
            simp at x_in₀
          /-
            case h.right.intro.intro.intro.intro.intro.intro.y_ne
            K : Type u_1
            inst✝¹ : Field K
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            hv : Valued K Γ₀
            F : Filter K
            h0 : Eq (Min.min (nhds 0) F) Bot.bot
            γ₀ : Units Γ₀
            M₀ : Set K
            M₀_in : Membership.mem F M₀
            γ : Units Γ₀
            M₁ : Set K
            M₁_in : Membership.mem F M₁
            x : K
            x_in₁ : Membership.mem M₁ x
            y : K
            left✝ : Membership.mem M₀ y
            y_in₁ : Membership.mem M₁ y
            H₁ : LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min (HMul.hMul (HMul.hMul γ γ₀) γ₀ …
            x_in₀ : LE.le (↑γ₀) (Valued.v x)
            this : Ne (Valued.v x) 0
            ⊢ Ne x 0
          -/
          exact (Valuation.ne_zero_iff _).mp this
          /-
            🎉 no goals
          -/
          /-
            case h.right.intro.intro.intro.intro.intro.intro.h
            K : Type u_1
            inst✝¹ : Field K
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            hv : Valued K Γ₀
            F : Filter K
            h0 : Eq (Min.min (nhds 0) F) Bot.bot
            γ₀ : Units Γ₀
            M₀ : Set K
            M₀_in : Membership.mem F M₀
            γ : Units Γ₀
            M₁ : Set K
            M₁_in : Membership.mem F M₁
            x : K
            x_in₁ : Membership.mem M₁ x
            y : K
            left✝ : Membership.mem M₀ y
            y_in₁ : Membership.mem M₁ y
            H₁ : LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min (HMul.hMul (HMul.hMul γ γ₀) γ₀ …
            x_in₀ : LE.le (↑γ₀) (Valued.v x)
            ⊢ LT.lt (Valued.v (HSub.hSub y x)) (Min.min (HMul.hMul (↑γ) (HMul.hMul (Valued …
          -/
        · refine lt_of_lt_of_le H₁ ?_
          /-
            case h.right.intro.intro.intro.intro.intro.intro.h
            K : Type u_1
            inst✝¹ : Field K
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            hv : Valued K Γ₀
            F : Filter K
            h0 : Eq (Min.min (nhds 0) F) Bot.bot
            γ₀ : Units Γ₀
            M₀ : Set K
            M₀_in : Membership.mem F M₀
            γ : Units Γ₀
            M₁ : Set K
            M₁_in : Membership.mem F M₁
            x : K
            x_in₁ : Membership.mem M₁ x
            y : K
            left✝ : Membership.mem M₀ y
            y_in₁ : Membership.mem M₁ y
            H₁ : LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min (HMul.hMul (HMul.hMul γ γ₀) γ₀ …
            x_in₀ : LE.le (↑γ₀) (Valued.v x)
            ⊢ LE.le (↑(Min.min (HMul.hMul (HMul.hMul γ γ₀) γ₀) γ₀)) (Min.min (HMul.hMul (↑ …
          -/
          rw [Units.min_val]
          /-
            case h.right.intro.intro.intro.intro.intro.intro.h
            K : Type u_1
            inst✝¹ : Field K
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            hv : Valued K Γ₀
            F : Filter K
            h0 : Eq (Min.min (nhds 0) F) Bot.bot
            γ₀ : Units Γ₀
            M₀ : Set K
            M₀_in : Membership.mem F M₀
            γ : Units Γ₀
            M₁ : Set K
            M₁_in : Membership.mem F M₁
            x : K
            x_in₁ : Membership.mem M₁ x
            y : K
            left✝ : Membership.mem M₀ y
            y_in₁ : Membership.mem M₁ y
            H₁ : LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min (HMul.hMul (HMul.hMul γ γ₀) γ₀ …
            x_in₀ : LE.le (↑γ₀) (Valued.v x)
            ⊢ LE.le (Min.min ↑(HMul.hMul (HMul.hMul γ γ₀) γ₀) ↑γ₀) (Min.min (HMul.hMul (↑γ …
          -/
          apply min_le_min _ x_in₀
          /-
            K : Type u_1
            inst✝¹ : Field K
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            hv : Valued K Γ₀
            F : Filter K
            h0 : Eq (Min.min (nhds 0) F) Bot.bot
            γ₀ : Units Γ₀
            M₀ : Set K
            M₀_in : Membership.mem F M₀
            γ : Units Γ₀
            M₁ : Set K
            M₁_in : Membership.mem F M₁
            x : K
            x_in₁ : Membership.mem M₁ x
            y : K
            left✝ : Membership.mem M₀ y
            y_in₁ : Membership.mem M₁ y
            H₁ : LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min (HMul.hMul (HMul.hMul γ γ₀) γ₀ …
            x_in₀ : LE.le (↑γ₀) (Valued.v x)
            ⊢ LE.le (↑(HMul.hMul (HMul.hMul γ γ₀) γ₀)) (HMul.hMul (↑γ) (HMul.hMul (Valued. …
          -/
          rw [mul_assoc]
          have : ((γ₀ * γ₀ : Γ₀ˣ) : Γ₀) ≤ v x * v x :=
            calc
              ↑γ₀ * ↑γ₀ ≤ ↑γ₀ * v x := mul_le_mul_left' x_in₀ ↑γ₀
              _ ≤ _ := mul_le_mul_right' x_in₀ (v x)
          /-
            K : Type u_1
            inst✝¹ : Field K
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            hv : Valued K Γ₀
            F : Filter K
            h0 : Eq (Min.min (nhds 0) F) Bot.bot
            γ₀ : Units Γ₀
            M₀ : Set K
            M₀_in : Membership.mem F M₀
            γ : Units Γ₀
            M₁ : Set K
            M₁_in : Membership.mem F M₁
            x : K
            x_in₁ : Membership.mem M₁ x
            y : K
            left✝ : Membership.mem M₀ y
            y_in₁ : Membership.mem M₁ y
            H₁ : LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min (HMul.hMul (HMul.hMul γ γ₀) γ₀ …
            x_in₀ : LE.le (↑γ₀) (Valued.v x)
            this : LE.le (↑(HMul.hMul γ₀ γ₀)) (HMul.hMul (Valued.v x) (Valued.v x))
            ⊢ LE.le (↑(HMul.hMul γ (HMul.hMul γ₀ γ₀))) (HMul.hMul (↑γ) (HMul.hMul (Valued. …
          -/
          rw [Units.val_mul]
          /-
            K : Type u_1
            inst✝¹ : Field K
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            hv : Valued K Γ₀
            F : Filter K
            h0 : Eq (Min.min (nhds 0) F) Bot.bot
            γ₀ : Units Γ₀
            M₀ : Set K
            M₀_in : Membership.mem F M₀
            γ : Units Γ₀
            M₁ : Set K
            M₁_in : Membership.mem F M₁
            x : K
            x_in₁ : Membership.mem M₁ x
            y : K
            left✝ : Membership.mem M₀ y
            y_in₁ : Membership.mem M₁ y
            H₁ : LT.lt (Valued.v (HSub.hSub y x)) ↑(Min.min (HMul.hMul (HMul.hMul γ γ₀) γ₀ …
            x_in₀ : LE.le (↑γ₀) (Valued.v x)
            this : LE.le (↑(HMul.hMul γ₀ γ₀)) (HMul.hMul (Valued.v x) (Valued.v x))
            ⊢ LE.le (HMul.hMul ↑γ ↑(HMul.hMul γ₀ γ₀)) (HMul.hMul (↑γ) (HMul.hMul (Valued.v …
          -/
          exact mul_le_mul_left' this γ }
          /-
            🎉 no goals
          -/


/-- The extension of the valuation of a valued field to the completion of the field. -/
noncomputable def extension : hat K → Γ₀ :=
  Completion.isDenseInducing_coe.extend (v : K → Γ₀)


theorem continuous_extension : Continuous (Valued.extension : hat K → Γ₀) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    hv : Valued K Γ₀
    ⊢ Continuous Valued.extension
  -/
  refine Completion.isDenseInducing_coe.continuous_extend ?_
  /-
    K : Type u_1
    inst✝¹ : Field K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    hv : Valued K Γ₀
    ⊢ ∀ (b : UniformSpace.Completion K), Exists fun c => Filter.Tendsto (⇑Valued.v …
  -/
  intro x₀
  /-
    K : Type u_1
    inst✝¹ : Field K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    hv : Valued K Γ₀
    x₀ : UniformSpace.Completion K
    ⊢ Exists fun c => Filter.Tendsto (⇑Valued.v) (Filter.comap (↑K) (nhds x₀)) (nh …
  -/
  rcases eq_or_ne x₀ 0 with (rfl | h)
    /-
      case inl
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      ⊢ Exists fun c => Filter.Tendsto (⇑Valued.v) (Filter.comap (↑K) (nhds 0)) (nhd …
    -/
  · refine ⟨0, ?_⟩
    /-
      case inl
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      ⊢ Filter.Tendsto (⇑Valued.v) (Filter.comap (↑K) (nhds 0)) (nhds 0)
    -/
    erw [← Completion.isDenseInducing_coe.isInducing.nhds_eq_comap]
    /-
      case inl
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      ⊢ Filter.Tendsto (⇑Valued.v) (nhds 0) (nhds 0)
    -/
    exact Valued.continuous_valuation.tendsto' 0 0 (map_zero v)
    /-
      🎉 no goals
    -/
  · have preimage_one : v ⁻¹' {(1 : Γ₀)} ∈ 𝓝 (1 : K) := by
      have : (v (1 : K) : Γ₀) ≠ 0 := by
        rw [Valuation.map_one]
        exact zero_ne_one.symm
      convert Valued.loc_const this
      ext x
      rw [Valuation.map_one, mem_preimage, mem_singleton_iff, mem_setOf_eq]
    obtain ⟨V, V_in, hV⟩ : ∃ V ∈ 𝓝 (1 : hat K), ∀ x : K, (x : hat K) ∈ V → (v x : Γ₀) = 1 := by
      rwa [Completion.isDenseInducing_coe.nhds_eq_comap, mem_comap] at preimage_one
    have : ∃ V' ∈ 𝓝 (1 : hat K), (0 : hat K) ∉ V' ∧ ∀ (x) (_ : x ∈ V') (y) (_ : y ∈ V'),
      x * y⁻¹ ∈ V := by
      have : Tendsto (fun p : hat K × hat K => p.1 * p.2⁻¹) ((𝓝 1) ×ˢ (𝓝 1)) (𝓝 1) := by
        rw [← nhds_prod_eq]
        conv =>
          congr
          rfl
          rfl
          rw [← one_mul (1 : hat K)]
        refine
          Tendsto.mul continuous_fst.continuousAt (Tendsto.comp ?_ continuous_snd.continuousAt)
        -- Porting note: Added `ContinuousAt.tendsto`
        convert (continuousAt_inv₀ (zero_ne_one.symm : 1 ≠ (0 : hat K))).tendsto
        exact inv_one.symm
      rcases tendsto_prod_self_iff.mp this V V_in with ⟨U, U_in, hU⟩
      let hatKstar := ({0}ᶜ : Set <| hat K)
      have : hatKstar ∈ 𝓝 (1 : hat K) := compl_singleton_mem_nhds zero_ne_one.symm
      use U ∩ hatKstar, Filter.inter_mem U_in this
      constructor
      · rintro ⟨_, h'⟩
        rw [mem_compl_singleton_iff] at h'
        exact h' rfl
      · rintro x ⟨hx, _⟩ y ⟨hy, _⟩
        apply hU <;> assumption
    /-
      case inr.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      x₀ : UniformSpace.Completion K
      h : Ne x₀ 0
      preimage_one : Membership.mem (nhds 1) (Set.preimage (⇑Valued.v) (Singleton.si …
      V : Set (UniformSpace.Completion K)
      V_in : Membership.mem (nhds 1) V
      hV : ∀ (x : K), Membership.mem V (↑K x) → Eq (Valued.v x) 1
      this : Exists fun V' => And (Membership.mem (nhds 1) V') (And (Not (Membership …
      ⊢ Exists fun c => Filter.Tendsto (⇑Valued.v) (Filter.comap (↑K) (nhds x₀)) (nh …
    -/
    rcases this with ⟨V', V'_in, zeroV', hV'⟩
    have nhds_right : (fun x => x * x₀) '' V' ∈ 𝓝 x₀ := by
      have l : Function.LeftInverse (fun x : hat K => x * x₀⁻¹) fun x : hat K => x * x₀ := by
        intro x
        simp only [mul_assoc, mul_inv_cancel₀ h, mul_one]
      have r : Function.RightInverse (fun x : hat K => x * x₀⁻¹) fun x : hat K => x * x₀ := by
        intro x
        simp only [mul_assoc, inv_mul_cancel₀ h, mul_one]
      have c : Continuous fun x : hat K => x * x₀⁻¹ := continuous_id.mul continuous_const
      rw [image_eq_preimage_of_inverse l r]
      rw [← mul_inv_cancel₀ h] at V'_in
      exact c.continuousAt V'_in
    have : ∃ z₀ : K, ∃ y₀ ∈ V', ↑z₀ = y₀ * x₀ ∧ z₀ ≠ 0 := by
      rcases Completion.denseRange_coe.mem_nhds nhds_right with ⟨z₀, y₀, y₀_in, H : y₀ * x₀ = z₀⟩
      refine ⟨z₀, y₀, y₀_in, ⟨H.symm, ?_⟩⟩
      rintro rfl
      exact mul_ne_zero (ne_of_mem_of_not_mem y₀_in zeroV') h H
    /-
      case inr.intro.intro.intro.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      x₀ : UniformSpace.Completion K
      h : Ne x₀ 0
      preimage_one : Membership.mem (nhds 1) (Set.preimage (⇑Valued.v) (Singleton.si …
      V : Set (UniformSpace.Completion K)
      V_in : Membership.mem (nhds 1) V
      hV : ∀ (x : K), Membership.mem V (↑K x) → Eq (Valued.v x) 1
      V' : Set (UniformSpace.Completion K)
      V'_in : Membership.mem (nhds 1) V'
      zeroV' : Not (Membership.mem V' 0)
      hV' : ∀ (x : UniformSpace.Completion K), Membership.mem V' x → ∀ (y : UniformS …
      nhds_right : Membership.mem (nhds x₀) (Set.image (fun x => HMul.hMul x x₀) V')
      this : Exists fun z₀ => Exists fun y₀ => And (Membership.mem V' y₀) (And (Eq ( …
      ⊢ Exists fun c => Filter.Tendsto (⇑Valued.v) (Filter.comap (↑K) (nhds x₀)) (nh …
    -/
    rcases this with ⟨z₀, y₀, y₀_in, hz₀, z₀_ne⟩
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      x₀ : UniformSpace.Completion K
      h : Ne x₀ 0
      preimage_one : Membership.mem (nhds 1) (Set.preimage (⇑Valued.v) (Singleton.si …
      V : Set (UniformSpace.Completion K)
      V_in : Membership.mem (nhds 1) V
      hV : ∀ (x : K), Membership.mem V (↑K x) → Eq (Valued.v x) 1
      V' : Set (UniformSpace.Completion K)
      V'_in : Membership.mem (nhds 1) V'
      zeroV' : Not (Membership.mem V' 0)
      hV' : ∀ (x : UniformSpace.Completion K), Membership.mem V' x → ∀ (y : UniformS …
      nhds_right : Membership.mem (nhds x₀) (Set.image (fun x => HMul.hMul x x₀) V')
      z₀ : K
      y₀ : UniformSpace.Completion K
      y₀_in : Membership.mem V' y₀
      hz₀ : Eq (↑K z₀) (HMul.hMul y₀ x₀)
      z₀_ne : Ne z₀ 0
      ⊢ Exists fun c => Filter.Tendsto (⇑Valued.v) (Filter.comap (↑K) (nhds x₀)) (nh …
    -/
    have vz₀_ne : (v z₀ : Γ₀) ≠ 0 := by rwa [Valuation.ne_zero_iff]
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      x₀ : UniformSpace.Completion K
      h : Ne x₀ 0
      preimage_one : Membership.mem (nhds 1) (Set.preimage (⇑Valued.v) (Singleton.si …
      V : Set (UniformSpace.Completion K)
      V_in : Membership.mem (nhds 1) V
      hV : ∀ (x : K), Membership.mem V (↑K x) → Eq (Valued.v x) 1
      V' : Set (UniformSpace.Completion K)
      V'_in : Membership.mem (nhds 1) V'
      zeroV' : Not (Membership.mem V' 0)
      hV' : ∀ (x : UniformSpace.Completion K), Membership.mem V' x → ∀ (y : UniformS …
      nhds_right : Membership.mem (nhds x₀) (Set.image (fun x => HMul.hMul x x₀) V')
      z₀ : K
      y₀ : UniformSpace.Completion K
      y₀_in : Membership.mem V' y₀
      hz₀ : Eq (↑K z₀) (HMul.hMul y₀ x₀)
      z₀_ne : Ne z₀ 0
      vz₀_ne : Ne (Valued.v z₀) 0
      ⊢ Exists fun c => Filter.Tendsto (⇑Valued.v) (Filter.comap (↑K) (nhds x₀)) (nh …
    -/
    refine ⟨v z₀, ?_⟩
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      x₀ : UniformSpace.Completion K
      h : Ne x₀ 0
      preimage_one : Membership.mem (nhds 1) (Set.preimage (⇑Valued.v) (Singleton.si …
      V : Set (UniformSpace.Completion K)
      V_in : Membership.mem (nhds 1) V
      hV : ∀ (x : K), Membership.mem V (↑K x) → Eq (Valued.v x) 1
      V' : Set (UniformSpace.Completion K)
      V'_in : Membership.mem (nhds 1) V'
      zeroV' : Not (Membership.mem V' 0)
      hV' : ∀ (x : UniformSpace.Completion K), Membership.mem V' x → ∀ (y : UniformS …
      nhds_right : Membership.mem (nhds x₀) (Set.image (fun x => HMul.hMul x x₀) V')
      z₀ : K
      y₀ : UniformSpace.Completion K
      y₀_in : Membership.mem V' y₀
      hz₀ : Eq (↑K z₀) (HMul.hMul y₀ x₀)
      z₀_ne : Ne z₀ 0
      vz₀_ne : Ne (Valued.v z₀) 0
      ⊢ Filter.Tendsto (⇑Valued.v) (Filter.comap (↑K) (nhds x₀)) (nhds (Valued.v z₀))
    -/
    rw [WithZeroTopology.tendsto_of_ne_zero vz₀_ne, eventually_comap]
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      x₀ : UniformSpace.Completion K
      h : Ne x₀ 0
      preimage_one : Membership.mem (nhds 1) (Set.preimage (⇑Valued.v) (Singleton.si …
      V : Set (UniformSpace.Completion K)
      V_in : Membership.mem (nhds 1) V
      hV : ∀ (x : K), Membership.mem V (↑K x) → Eq (Valued.v x) 1
      V' : Set (UniformSpace.Completion K)
      V'_in : Membership.mem (nhds 1) V'
      zeroV' : Not (Membership.mem V' 0)
      hV' : ∀ (x : UniformSpace.Completion K), Membership.mem V' x → ∀ (y : UniformS …
      nhds_right : Membership.mem (nhds x₀) (Set.image (fun x => HMul.hMul x x₀) V')
      z₀ : K
      y₀ : UniformSpace.Completion K
      y₀_in : Membership.mem V' y₀
      hz₀ : Eq (↑K z₀) (HMul.hMul y₀ x₀)
      z₀_ne : Ne z₀ 0
      vz₀_ne : Ne (Valued.v z₀) 0
      ⊢ Filter.Eventually (fun b => ∀ (a : K), Eq (↑K a) b → Eq (Valued.v a) (Valued …
    -/
    filter_upwards [nhds_right] with x x_in a ha
    /-
      case h
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      x₀ : UniformSpace.Completion K
      h : Ne x₀ 0
      preimage_one : Membership.mem (nhds 1) (Set.preimage (⇑Valued.v) (Singleton.si …
      V : Set (UniformSpace.Completion K)
      V_in : Membership.mem (nhds 1) V
      hV : ∀ (x : K), Membership.mem V (↑K x) → Eq (Valued.v x) 1
      V' : Set (UniformSpace.Completion K)
      V'_in : Membership.mem (nhds 1) V'
      zeroV' : Not (Membership.mem V' 0)
      hV' : ∀ (x : UniformSpace.Completion K), Membership.mem V' x → ∀ (y : UniformS …
      nhds_right : Membership.mem (nhds x₀) (Set.image (fun x => HMul.hMul x x₀) V')
      z₀ : K
      y₀ : UniformSpace.Completion K
      y₀_in : Membership.mem V' y₀
      hz₀ : Eq (↑K z₀) (HMul.hMul y₀ x₀)
      z₀_ne : Ne z₀ 0
      vz₀_ne : Ne (Valued.v z₀) 0
      x : UniformSpace.Completion K
      x_in : Membership.mem (Set.image (fun x => HMul.hMul x x₀) V') x
      a : K
      ha : Eq (↑K a) x
      ⊢ Eq (Valued.v a) (Valued.v z₀)
    -/
    rcases x_in with ⟨y, y_in, rfl⟩
    have : (v (a * z₀⁻¹) : Γ₀) = 1 := by
      apply hV
      have : (z₀⁻¹ : K) = (z₀ : hat K)⁻¹ := map_inv₀ (Completion.coeRingHom : K →+* hat K) z₀
      rw [Completion.coe_mul, this, ha, hz₀, mul_inv, mul_comm y₀⁻¹, ← mul_assoc, mul_assoc y,
        mul_inv_cancel₀ h, mul_one]
      solve_by_elim
    calc
      v a = v (a * z₀⁻¹ * z₀) := by rw [mul_assoc, inv_mul_cancel₀ z₀_ne, mul_one]
      _ = v (a * z₀⁻¹) * v z₀ := Valuation.map_mul _ _ _
      _ = v z₀ := by rw [this, one_mul]


@[simp, norm_cast]
theorem extension_extends (x : K) : extension (x : hat K) = v x := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    hv : Valued K Γ₀
    x : K
    ⊢ Eq (Valued.extension (↑K x)) (Valued.v x)
  -/
  refine Completion.isDenseInducing_coe.extend_eq_of_tendsto ?_
  /-
    K : Type u_1
    inst✝¹ : Field K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    hv : Valued K Γ₀
    x : K
    ⊢ Filter.Tendsto (⇑Valued.v) (Filter.comap (↑K) (nhds (↑K x))) (nhds (Valued.v …
  -/
  rw [← Completion.isDenseInducing_coe.nhds_eq_comap]
  /-
    K : Type u_1
    inst✝¹ : Field K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    hv : Valued K Γ₀
    x : K
    ⊢ Filter.Tendsto (⇑Valued.v) (nhds x) (nhds (Valued.v x))
  -/
  exact Valued.continuous_valuation.continuousAt
  /-
    🎉 no goals
  -/


/-- the extension of a valuation on a division ring to its completion. -/
noncomputable def extensionValuation : Valuation (hat K) Γ₀ where
  toFun := Valued.extension
  map_zero' := by
    /-
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      ⊢ Eq (Valued.extension 0) 0
    -/
    rw [← v.map_zero (R := K), ← Valued.extension_extends (0 : K)]
    /-
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      ⊢ Eq (Valued.extension 0) (Valued.extension (↑K 0))
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_one' := by
    /-
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      ⊢ Eq ({ toFun := Valued.extension, map_zero' := ⋯ }.toFun 1) 1
    -/
    simp only
    /-
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      ⊢ Eq (Valued.extension 1) 1
    -/
    rw [← Completion.coe_one, Valued.extension_extends (1 : K)]
    /-
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      ⊢ Eq (Valued.v 1) 1
    -/
    exact Valuation.map_one _
    /-
      🎉 no goals
    -/
  map_mul' x y := by
    apply Completion.induction_on₂ x y
      (p := fun x y => extension (x * y) = extension x * extension y)
    · have c1 : Continuous fun x : hat K × hat K => Valued.extension (x.1 * x.2) :=
        Valued.continuous_extension.comp (continuous_fst.mul continuous_snd)
      have c2 : Continuous fun x : hat K × hat K => Valued.extension x.1 * Valued.extension x.2 :=
        (Valued.continuous_extension.comp continuous_fst).mul
          (Valued.continuous_extension.comp continuous_snd)
      /-
        case hp
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        x y : UniformSpace.Completion K
        c1 : Continuous fun x => Valued.extension (HMul.hMul x.1 x.2)
        c2 : Continuous fun x => HMul.hMul (Valued.extension x.1) (Valued.extension x.2)
        ⊢ IsClosed (setOf fun x => Eq (Valued.extension (HMul.hMul x.1 x.2)) (HMul.hMu …
      -/
      exact isClosed_eq c1 c2
      /-
        🎉 no goals
      -/
      /-
        case ih
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        x y : UniformSpace.Completion K
        ⊢ ∀ (a b : K), Eq (Valued.extension (HMul.hMul (↑K a) (↑K b))) (HMul.hMul (Val …
      -/
    · intro x y
      /-
        case ih
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        x✝ y✝ : UniformSpace.Completion K
        x y : K
        ⊢ Eq (Valued.extension (HMul.hMul (↑K x) (↑K y))) (HMul.hMul (Valued.extension …
      -/
      norm_cast
      /-
        case ih
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        x✝ y✝ : UniformSpace.Completion K
        x y : K
        ⊢ Eq (Valued.v (HMul.hMul x y)) (HMul.hMul (Valued.v x) (Valued.v y))
      -/
      exact Valuation.map_mul _ _ _
      /-
        🎉 no goals
      -/
  map_add_le_max' x y := by
    /-
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      x y : UniformSpace.Completion K
      ⊢ LE.le ((↑{ toFun := Valued.extension, map_zero' := ⋯, map_one' := ⋯, map_mul …
    -/
    rw [le_max_iff]
    apply Completion.induction_on₂ x y
      (p := fun x y => extension (x + y) ≤ extension x ∨ extension (x + y) ≤ extension y)
      /-
        case hp
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        x y : UniformSpace.Completion K
        ⊢ IsClosed (setOf fun x => Or (LE.le (Valued.extension (HAdd.hAdd x.1 x.2)) (V …
      -/
    · have cont : Continuous (Valued.extension : hat K → Γ₀) := Valued.continuous_extension
      exact
        (isClosed_le (cont.comp continuous_add) <| cont.comp continuous_fst).union
          (isClosed_le (cont.comp continuous_add) <| cont.comp continuous_snd)
      /-
        case ih
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        x y : UniformSpace.Completion K
        ⊢ ∀ (a b : K), Or (LE.le (Valued.extension (HAdd.hAdd (↑K a) (↑K b))) (Valued. …
      -/
    · intro x y
      /-
        case ih
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        x✝ y✝ : UniformSpace.Completion K
        x y : K
        ⊢ Or (LE.le (Valued.extension (HAdd.hAdd (↑K x) (↑K y))) (Valued.extension (↑K …
      -/
      norm_cast
      /-
        case ih
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        x✝ y✝ : UniformSpace.Completion K
        x y : K
        ⊢ Or (LE.le (Valued.v (HAdd.hAdd x y)) (Valued.v x)) (LE.le (Valued.v (HAdd.hA …
      -/
      rw [← le_max_iff]
      /-
        case ih
        K : Type u_1
        inst✝¹ : Field K
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        hv : Valued K Γ₀
        x✝ y✝ : UniformSpace.Completion K
        x y : K
        ⊢ LE.le (Valued.v (HAdd.hAdd x y)) (Max.max (Valued.v x) (Valued.v y))
      -/
      exact v.map_add x y
      /-
        🎉 no goals
      -/

-- Bourbaki CA VI §5 no.3 Proposition 5 (d)

theorem closure_coe_completion_v_lt {γ : Γ₀ˣ} :
    closure ((↑) '' { x : K | v x < (γ : Γ₀) }) =
    { x : hat K | extensionValuation x < (γ : Γ₀) } := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    hv : Valued K Γ₀
    γ : Units Γ₀
    ⊢ Eq (closure (Set.image (↑K) (setOf fun x => LT.lt (Valued.v x) ↑γ))) (setOf  …
  -/
  ext x
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    hv : Valued K Γ₀
    γ : Units Γ₀
    x : UniformSpace.Completion K
    ⊢ Iff (Membership.mem (closure (Set.image (↑K) (setOf fun x => LT.lt (Valued.v …
  -/
  let γ₀ := extensionValuation x
  suffices γ₀ ≠ 0 → (x ∈ closure ((↑) '' { x : K | v x < (γ : Γ₀) }) ↔ γ₀ < (γ : Γ₀)) by
    rcases eq_or_ne γ₀ 0 with h | h
    · simp only [h, (Valuation.zero_iff _).mp h, mem_setOf_eq, Valuation.map_zero, Units.zero_lt,
        iff_true]
      apply subset_closure
      exact ⟨0, by simp only [mem_setOf_eq, Valuation.map_zero, Units.zero_lt, true_and]; rfl⟩
    · exact this h
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    hv : Valued K Γ₀
    γ : Units Γ₀
    x : UniformSpace.Completion K
    γ₀ : Γ₀ := Valued.extensionValuation x
    ⊢ Ne γ₀ 0 → Iff (Membership.mem (closure (Set.image (↑K) (setOf fun x => LT.lt …
  -/
  intro h
  have hγ₀ : extension ⁻¹' {γ₀} ∈ 𝓝 x :=
    continuous_extension.continuousAt.preimage_mem_nhds
      (WithZeroTopology.singleton_mem_nhds_of_ne_zero h)
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    hv : Valued K Γ₀
    γ : Units Γ₀
    x : UniformSpace.Completion K
    γ₀ : Γ₀ := Valued.extensionValuation x
    h : Ne γ₀ 0
    hγ₀ : Membership.mem (nhds x) (Set.preimage Valued.extension (Singleton.single …
    ⊢ Iff (Membership.mem (closure (Set.image (↑K) (setOf fun x => LT.lt (Valued.v …
  -/
  rw [mem_closure_iff_nhds']
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    hv : Valued K Γ₀
    γ : Units Γ₀
    x : UniformSpace.Completion K
    γ₀ : Γ₀ := Valued.extensionValuation x
    h : Ne γ₀ 0
    hγ₀ : Membership.mem (nhds x) (Set.preimage Valued.extension (Singleton.single …
    ⊢ Iff (∀ (t : Set (UniformSpace.Completion K)), Membership.mem (nhds x) t → Ex …
  -/
  refine ⟨fun hx => ?_, fun hx s hs => ?_⟩
    /-
      case h.refine_1
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      γ : Units Γ₀
      x : UniformSpace.Completion K
      γ₀ : Γ₀ := Valued.extensionValuation x
      h : Ne γ₀ 0
      hγ₀ : Membership.mem (nhds x) (Set.preimage Valued.extension (Singleton.single …
      hx : ∀ (t : Set (UniformSpace.Completion K)), Membership.mem (nhds x) t → Exis …
      ⊢ LT.lt γ₀ ↑γ
    -/
  · obtain ⟨⟨-, y, hy₁ : v y < (γ : Γ₀), rfl⟩, hy₂⟩ := hx _ hγ₀
    /-
      case h.refine_1.intro.mk.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      γ : Units Γ₀
      x : UniformSpace.Completion K
      γ₀ : Γ₀ := Valued.extensionValuation x
      h : Ne γ₀ 0
      hγ₀ : Membership.mem (nhds x) (Set.preimage Valued.extension (Singleton.single …
      hx : ∀ (t : Set (UniformSpace.Completion K)), Membership.mem (nhds x) t → Exis …
      y : K
      hy₁ : LT.lt (Valued.v y) ↑γ
      hy₂ : Membership.mem (Set.preimage Valued.extension (Singleton.singleton γ₀))  …
      ⊢ LT.lt γ₀ ↑γ
    -/
    replace hy₂ : v y = γ₀ := by simpa using hy₂
    /-
      case h.refine_1.intro.mk.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      γ : Units Γ₀
      x : UniformSpace.Completion K
      γ₀ : Γ₀ := Valued.extensionValuation x
      h : Ne γ₀ 0
      hγ₀ : Membership.mem (nhds x) (Set.preimage Valued.extension (Singleton.single …
      hx : ∀ (t : Set (UniformSpace.Completion K)), Membership.mem (nhds x) t → Exis …
      y : K
      hy₁ : LT.lt (Valued.v y) ↑γ
      hy₂ : Eq (Valued.v y) γ₀
      ⊢ LT.lt γ₀ ↑γ
    -/
    rwa [← hy₂]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      γ : Units Γ₀
      x : UniformSpace.Completion K
      γ₀ : Γ₀ := Valued.extensionValuation x
      h : Ne γ₀ 0
      hγ₀ : Membership.mem (nhds x) (Set.preimage Valued.extension (Singleton.single …
      hx : LT.lt γ₀ ↑γ
      s : Set (UniformSpace.Completion K)
      hs : Membership.mem (nhds x) s
      ⊢ Exists fun y => Membership.mem s ↑y
    -/
  · obtain ⟨y, hy₁, hy₂⟩ := Completion.denseRange_coe.mem_nhds (inter_mem hγ₀ hs)
    /-
      case h.refine_2.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      γ : Units Γ₀
      x : UniformSpace.Completion K
      γ₀ : Γ₀ := Valued.extensionValuation x
      h : Ne γ₀ 0
      hγ₀ : Membership.mem (nhds x) (Set.preimage Valued.extension (Singleton.single …
      hx : LT.lt γ₀ ↑γ
      s : Set (UniformSpace.Completion K)
      hs : Membership.mem (nhds x) s
      y : K
      hy₁ : Membership.mem (Set.preimage Valued.extension (Singleton.singleton γ₀))  …
      hy₂ : Membership.mem s (↑K y)
      ⊢ Exists fun y => Membership.mem s ↑y
    -/
    replace hy₁ : v y = γ₀ := by simpa using hy₁
    /-
      case h.refine_2.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      γ : Units Γ₀
      x : UniformSpace.Completion K
      γ₀ : Γ₀ := Valued.extensionValuation x
      h : Ne γ₀ 0
      hγ₀ : Membership.mem (nhds x) (Set.preimage Valued.extension (Singleton.single …
      hx : LT.lt γ₀ ↑γ
      s : Set (UniformSpace.Completion K)
      hs : Membership.mem (nhds x) s
      y : K
      hy₂ : Membership.mem s (↑K y)
      hy₁ : Eq (Valued.v y) γ₀
      ⊢ Exists fun y => Membership.mem s ↑y
    -/
    rw [← hy₁] at hx
    /-
      case h.refine_2.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      γ : Units Γ₀
      x : UniformSpace.Completion K
      γ₀ : Γ₀ := Valued.extensionValuation x
      h : Ne γ₀ 0
      hγ₀ : Membership.mem (nhds x) (Set.preimage Valued.extension (Singleton.single …
      s : Set (UniformSpace.Completion K)
      hs : Membership.mem (nhds x) s
      y : K
      hx : LT.lt (Valued.v y) ↑γ
      hy₂ : Membership.mem s (↑K y)
      hy₁ : Eq (Valued.v y) γ₀
      ⊢ Exists fun y => Membership.mem s ↑y
    -/
    exact ⟨⟨y, ⟨y, hx, rfl⟩⟩, hy₂⟩
    /-
      🎉 no goals
    -/


noncomputable instance valuedCompletion : Valued (hat K) Γ₀ where
  v := extensionValuation
  is_topological_valuation s := by
    suffices
      HasBasis (𝓝 (0 : hat K)) (fun _ => True) fun γ : Γ₀ˣ => { x | extensionValuation x < γ } by
      rw [this.mem_iff]
      exact exists_congr fun γ => by simp
    /-
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      s : Set (UniformSpace.Completion K)
      ⊢ (nhds 0).HasBasis (fun x => True) fun γ => setOf fun x => LT.lt (Valued.exte …
    -/
    simp_rw [← closure_coe_completion_v_lt]
    /-
      K : Type u_1
      inst✝¹ : Field K
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      hv : Valued K Γ₀
      s : Set (UniformSpace.Completion K)
      ⊢ (nhds 0).HasBasis (fun x => True) fun γ => closure (Set.image (↑K) (setOf fu …
    -/
    exact (hasBasis_nhds_zero K Γ₀).hasBasis_of_isDenseInducing Completion.isDenseInducing_coe
    /-
      🎉 no goals
    -/

-- Porting note: removed @[norm_cast] attribute due to error:
-- norm_cast: badly shaped lemma, rhs can't start with coe

@[simp]
theorem valuedCompletion_apply (x : K) : Valued.v (x : hat K) = v x :=
  extension_extends x


/-- A `Valued` version of `Valuation.integer`, enabling the notation `𝒪[K]` for the
valuation integers of a valued field `K`. -/
@[reducible]
def integer : Subring K := (vK.v).integer


@[inherit_doc]
scoped notation "𝒪[" K "]" => Valued.integer K


/-- An abbreviation for `IsLocalRing.maximalIdeal 𝒪[K]` of a valued field `K`, enabling the notation
`𝓂[K]` for the maximal ideal in `𝒪[K]` of a valued field `K`. -/
@[reducible]
def maximalIdeal : Ideal 𝒪[K] := IsLocalRing.maximalIdeal 𝒪[K]


@[inherit_doc]
scoped notation "𝓂[" K "]" => maximalIdeal K


/-- An abbreviation for `IsLocalRing.ResidueField 𝒪[K]` of a `Valued` instance, enabling the
notation `𝓀[K]` for the residue field of a valued field `K`. -/
@[reducible]
def ResidueField := IsLocalRing.ResidueField (𝒪[K])


@[inherit_doc]
scoped notation "𝓀[" K "]" => ResidueField K


