/-- A topological space `X` is an H-space if it behaves like a (potentially non-associative)
topological group, but where the axioms for a group only hold up to homotopy.
-/
class HSpace (X : Type u) [TopologicalSpace X] where
  hmul : C(X × X, X)
  e : X
  hmul_e_e : hmul (e, e) = e
  eHmul :
    (hmul.comp <| (const X e).prodMk <| ContinuousMap.id X).HomotopyRel (ContinuousMap.id X) {e}
  hmulE :
    (hmul.comp <| (ContinuousMap.id X).prodMk <| const X e).HomotopyRel (ContinuousMap.id X) {e}


/-- The binary operation `hmul` on an `H`-space -/
scoped[HSpaces] notation x "⋀" y => HSpace.hmul (x, y)

-- Porting note: opening `HSpaces` so that the above notation works

instance HSpace.prod (X : Type u) (Y : Type v) [TopologicalSpace X] [TopologicalSpace Y] [HSpace X]
    [HSpace Y] : HSpace (X × Y) where
  hmul := ⟨fun p => (p.1.1 ⋀ p.2.1, p.1.2 ⋀ p.2.2), by
    -- Porting note: was `continuity`
    exact ((map_continuous HSpace.hmul).comp ((continuous_fst.comp continuous_fst).prod_mk
        (continuous_fst.comp continuous_snd))).prod_mk ((map_continuous HSpace.hmul).comp
        ((continuous_snd.comp continuous_fst).prod_mk (continuous_snd.comp continuous_snd)))
  ⟩
  e := (HSpace.e, HSpace.e)
  hmul_e_e := by
    /-
      X : Type u
      Y : Type v
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : HSpace X
      inst✝ : HSpace Y
      ⊢ Eq ({ toFun := fun p => { fst := HSpace.hmul { fst := p.1.1, snd := p.2.1 }, …
    -/
    simp only [ContinuousMap.coe_mk, Prod.mk.inj_iff]
    /-
      X : Type u
      Y : Type v
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : HSpace X
      inst✝ : HSpace Y
      ⊢ And (Eq (HSpace.hmul { fst := HSpace.e, snd := HSpace.e }) HSpace.e) (Eq (HS …
    -/
    exact ⟨HSpace.hmul_e_e, HSpace.hmul_e_e⟩
    /-
      🎉 no goals
    -/
  eHmul := by
    /-
      X : Type u
      Y : Type v
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : HSpace X
      inst✝ : HSpace Y
      ⊢ ({ toFun := fun p => { fst := HSpace.hmul { fst := p.1.1, snd := p.2.1 }, sn …
    -/
    let G : I × X × Y → X × Y := fun p => (HSpace.eHmul (p.1, p.2.1), HSpace.eHmul (p.1, p.2.2))
    have hG : Continuous G :=
      (Continuous.comp HSpace.eHmul.1.1.2
          (continuous_fst.prod_mk (continuous_fst.comp continuous_snd))).prod_mk
        (Continuous.comp HSpace.eHmul.1.1.2
          (continuous_fst.prod_mk (continuous_snd.comp continuous_snd)))
    /-
      X : Type u
      Y : Type v
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : HSpace X
      inst✝ : HSpace Y
      G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.eHm …
      hG : Continuous G
      ⊢ ({ toFun := fun p => { fst := HSpace.hmul { fst := p.1.1, snd := p.2.1 }, sn …
    -/
    use! ⟨G, hG⟩
      /-
        case map_zero_left
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.eHm …
        hG : Continuous G
        ⊢ ∀ (x : Prod X Y), Eq ({ toFun := G, continuous_toFun := hG }.toFun { fst :=  …
      -/
    · rintro ⟨x, y⟩
      /-
        case map_zero_left.mk
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.eHm …
        hG : Continuous G
        x : X
        y : Y
        ⊢ Eq ({ toFun := G, continuous_toFun := hG }.toFun { fst := 0, snd := { fst := …
      -/
      exact Prod.ext (HSpace.eHmul.1.2 x) (HSpace.eHmul.1.2 y)
      /-
        🎉 no goals
      -/
      /-
        case map_one_left
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.eHm …
        hG : Continuous G
        ⊢ ∀ (x : Prod X Y), Eq ({ toFun := G, continuous_toFun := hG }.toFun { fst :=  …
      -/
    · rintro ⟨x, y⟩
      /-
        case map_one_left.mk
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.eHm …
        hG : Continuous G
        x : X
        y : Y
        ⊢ Eq ({ toFun := G, continuous_toFun := hG }.toFun { fst := 1, snd := { fst := …
      -/
      exact Prod.ext (HSpace.eHmul.1.3 x) (HSpace.eHmul.1.3 y)
      /-
        🎉 no goals
      -/
      /-
        case prop'
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.eHm …
        hG : Continuous G
        ⊢ ∀ (t : ↑unitInterval) (x : Prod X Y), Membership.mem (Singleton.singleton {  …
      -/
    · rintro t ⟨x, y⟩ h
      /-
        case prop'.mk
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.eHm …
        hG : Continuous G
        t : ↑unitInterval
        x : X
        y : Y
        h : Membership.mem (Singleton.singleton { fst := HSpace.e, snd := HSpace.e })  …
        ⊢ Eq ({ toFun := fun x => { toFun := G, continuous_toFun := hG, map_zero_left  …
      -/
      replace h := Prod.mk.inj_iff.mp h
      /-
        case prop'.mk
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.eHm …
        hG : Continuous G
        t : ↑unitInterval
        x : X
        y : Y
        h : And (Eq x HSpace.e) (Eq y HSpace.e)
        ⊢ Eq ({ toFun := fun x => { toFun := G, continuous_toFun := hG, map_zero_left  …
      -/
      exact Prod.ext (HSpace.eHmul.2 t x h.1) (HSpace.eHmul.2 t y h.2)
      /-
        🎉 no goals
      -/
  hmulE := by
    /-
      X : Type u
      Y : Type v
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : HSpace X
      inst✝ : HSpace Y
      ⊢ ({ toFun := fun p => { fst := HSpace.hmul { fst := p.1.1, snd := p.2.1 }, sn …
    -/
    let G : I × X × Y → X × Y := fun p => (HSpace.hmulE (p.1, p.2.1), HSpace.hmulE (p.1, p.2.2))
    have hG : Continuous G :=
      (Continuous.comp HSpace.hmulE.1.1.2
            (continuous_fst.prod_mk (continuous_fst.comp continuous_snd))).prod_mk
        (Continuous.comp HSpace.hmulE.1.1.2
          (continuous_fst.prod_mk (continuous_snd.comp continuous_snd)))
    /-
      X : Type u
      Y : Type v
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : HSpace X
      inst✝ : HSpace Y
      G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.hmu …
      hG : Continuous G
      ⊢ ({ toFun := fun p => { fst := HSpace.hmul { fst := p.1.1, snd := p.2.1 }, sn …
    -/
    use! ⟨G, hG⟩
      /-
        case map_zero_left
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.hmu …
        hG : Continuous G
        ⊢ ∀ (x : Prod X Y), Eq ({ toFun := G, continuous_toFun := hG }.toFun { fst :=  …
      -/
    · rintro ⟨x, y⟩
      /-
        case map_zero_left.mk
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.hmu …
        hG : Continuous G
        x : X
        y : Y
        ⊢ Eq ({ toFun := G, continuous_toFun := hG }.toFun { fst := 0, snd := { fst := …
      -/
      exact Prod.ext (HSpace.hmulE.1.2 x) (HSpace.hmulE.1.2 y)
      /-
        🎉 no goals
      -/
      /-
        case map_one_left
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.hmu …
        hG : Continuous G
        ⊢ ∀ (x : Prod X Y), Eq ({ toFun := G, continuous_toFun := hG }.toFun { fst :=  …
      -/
    · rintro ⟨x, y⟩
      /-
        case map_one_left.mk
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.hmu …
        hG : Continuous G
        x : X
        y : Y
        ⊢ Eq ({ toFun := G, continuous_toFun := hG }.toFun { fst := 1, snd := { fst := …
      -/
      exact Prod.ext (HSpace.hmulE.1.3 x) (HSpace.hmulE.1.3 y)
      /-
        🎉 no goals
      -/
      /-
        case prop'
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.hmu …
        hG : Continuous G
        ⊢ ∀ (t : ↑unitInterval) (x : Prod X Y), Membership.mem (Singleton.singleton {  …
      -/
    · rintro t ⟨x, y⟩ h
      /-
        case prop'.mk
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.hmu …
        hG : Continuous G
        t : ↑unitInterval
        x : X
        y : Y
        h : Membership.mem (Singleton.singleton { fst := HSpace.e, snd := HSpace.e })  …
        ⊢ Eq ({ toFun := fun x => { toFun := G, continuous_toFun := hG, map_zero_left  …
      -/
      replace h := Prod.mk.inj_iff.mp h
      /-
        case prop'.mk
        X : Type u
        Y : Type v
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : HSpace X
        inst✝ : HSpace Y
        G : Prod (↑unitInterval) (Prod X Y) → Prod X Y := fun p => { fst := HSpace.hmu …
        hG : Continuous G
        t : ↑unitInterval
        x : X
        y : Y
        h : And (Eq x HSpace.e) (Eq y HSpace.e)
        ⊢ Eq ({ toFun := fun x => { toFun := G, continuous_toFun := hG, map_zero_left  …
      -/
      exact Prod.ext (HSpace.hmulE.2 t x h.1) (HSpace.hmulE.2 t y h.2)
      /-
        🎉 no goals
      -/



/-- The definition `toHSpace` is not an instance because its additive version would
lead to a diamond since a topological field would inherit two `HSpace` structures, one from the
`MulOneClass` and one from the `AddZeroClass`. In the case of a group, we make
`TopologicalGroup.hSpace` an instance."-/
@[to_additive
      "The definition `toHSpace` is not an instance because it comes together with a
      multiplicative version which would lead to a diamond since a topological field would inherit
      two `HSpace` structures, one from the `MulOneClass` and one from the `AddZeroClass`.
      In the case of an additive group, we make `TopologicalAddGroup.hSpace` an instance."]
def toHSpace (M : Type u) [MulOneClass M] [TopologicalSpace M] [ContinuousMul M] : HSpace M where
  hmul := ⟨Function.uncurry Mul.mul, continuous_mul⟩
  e := 1
  hmul_e_e := one_mul 1
                                               /-
                                                 M : Type u
                                                 inst✝² : MulOneClass M
                                                 inst✝¹ : TopologicalSpace M
                                                 inst✝ : ContinuousMul M
                                                 ⊢ Eq ({ toFun := Function.uncurry Mul.mul, continuous_toFun := ⋯ }.comp ((Cont …
                                               -/
  eHmul := (HomotopyRel.refl _ _).cast rfl (by ext1; apply one_mul)
                                                     /-
                                                       🎉 no goals
                                                     -/
                                               /-
                                                 M : Type u
                                                 inst✝² : MulOneClass M
                                                 inst✝¹ : TopologicalSpace M
                                                 inst✝ : ContinuousMul M
                                                 ⊢ Eq ({ toFun := Function.uncurry Mul.mul, continuous_toFun := ⋯ }.comp ((Cont …
                                               -/
  hmulE := (HomotopyRel.refl _ _).cast rfl (by ext1; apply mul_one)
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive]
instance (priority := 600) hSpace (G : Type u) [TopologicalSpace G] [Group G] [TopologicalGroup G] :
    HSpace G :=
  toHSpace G


theorem one_eq_hSpace_e {G : Type u} [TopologicalSpace G] [Group G] [TopologicalGroup G] :
    (1 : G) = HSpace.e :=
  rfl

/- In the following example we see that the H-space structure on the product of two topological
groups is definitionally equally to the product H-space-structure of the two groups. -/

/-- `qRight` is analogous to the function `Q` defined on p. 475 of [serre1951] that helps proving
continuity of `delayReflRight`. -/
def qRight (p : I × I) : I :=
  Set.projIcc 0 1 zero_le_one (2 * p.1 / (1 + p.2))


theorem continuous_qRight : Continuous qRight :=
  continuous_projIcc.comp <|
                       /-
                         ⊢ Continuous fun p => HMul.hMul 2 ↑p.1
                       -/
                       /-
                         🎉 no goals
                       -/
    Continuous.div (by fun_prop) (by fun_prop) fun _ ↦ (add_pos zero_lt_one).ne'
                                     /-
                                       🎉 no goals
                                     -/


theorem qRight_zero_left (θ : I) : qRight (0, θ) = 0 :=
                                 /-
                                   θ : ↑unitInterval
                                   ⊢ LE.le (HDiv.hDiv (HMul.hMul 2 ↑{ fst := 0, snd := θ }.1) (HAdd.hAdd 1 ↑{ fst …
                                 -/
  Set.projIcc_of_le_left _ <| by simp only [coe_zero, mul_zero, zero_div, le_refl]
                                 /-
                                   🎉 no goals
                                 -/


theorem qRight_one_left (θ : I) : qRight (1, θ) = 1 :=
  Set.projIcc_of_right_le _ <|
    (le_div_iff₀ <| add_pos zero_lt_one).2 <| by
      /-
        θ : ↑unitInterval
        ⊢ LE.le (HMul.hMul 1 (HAdd.hAdd 1 ↑{ fst := 1, snd := θ }.2)) (HMul.hMul 2 ↑{  …
      -/
      dsimp only
      /-
        θ : ↑unitInterval
        ⊢ LE.le (HMul.hMul 1 (HAdd.hAdd 1 ↑θ)) (HMul.hMul 2 ↑1)
      -/
      rw [coe_one, one_mul, mul_one, add_comm, ← one_add_one_eq_two]
      /-
        θ : ↑unitInterval
        ⊢ LE.le (HAdd.hAdd (↑θ) 1) (HAdd.hAdd 1 1)
      -/
      simp only [add_le_add_iff_right]
      /-
        θ : ↑unitInterval
        ⊢ LE.le (↑θ) 1
      -/
      exact le_one _
      /-
        🎉 no goals
      -/


theorem qRight_zero_right (t : I) :
    (qRight (t, 0) : ℝ) = if (t : ℝ) ≤ 1 / 2 then (2 : ℝ) * t else 1 := by
  /-
    t : ↑unitInterval
    ⊢ Eq (↑(unitInterval.qRight { fst := t, snd := 0 })) (ite (LE.le (↑t) (1 / 2)) …
  -/
  simp only [qRight, coe_zero, add_zero, div_one]
  /-
    t : ↑unitInterval
    ⊢ Eq (↑(Set.projIcc 0 1 unitInterval.qRight.proof_1 (HMul.hMul 2 ↑t))) (ite (L …
  -/
  split_ifs
    /-
      case pos
      t : ↑unitInterval
      h✝ : LE.le (↑t) (1 / 2)
      ⊢ Eq (↑(Set.projIcc 0 1 unitInterval.qRight.proof_1 (HMul.hMul 2 ↑t))) (HMul.h …
    -/
  · rw [Set.projIcc_of_mem _ ((mul_pos_mem_iff zero_lt_two).2 _)]
    /-
      t : ↑unitInterval
      h✝ : LE.le (↑t) (1 / 2)
      ⊢ Membership.mem (Set.Icc 0 (1 / 2)) ↑t
    -/
    refine ⟨t.2.1, ?_⟩
    /-
      t : ↑unitInterval
      h✝ : LE.le (↑t) (1 / 2)
      ⊢ LE.le (↑t) (1 / 2)
    -/
    tauto
    /-
      🎉 no goals
    -/
    /-
      case neg
      t : ↑unitInterval
      h✝ : Not (LE.le (↑t) (1 / 2))
      ⊢ Eq (↑(Set.projIcc 0 1 unitInterval.qRight.proof_1 (HMul.hMul 2 ↑t))) 1
    -/
  · rw [(Set.projIcc_eq_right _).2]
      /-
        case neg
        t : ↑unitInterval
        h✝ : Not (LE.le (↑t) (1 / 2))
        ⊢ LE.le 1 (HMul.hMul 2 ↑t)
      -/
    · linarith
      /-
        🎉 no goals
      -/
      /-
        t : ↑unitInterval
        h✝ : Not (LE.le (↑t) (1 / 2))
        ⊢ LT.lt 0 1
      -/
    · exact zero_lt_one
      /-
        🎉 no goals
      -/


theorem qRight_one_right (t : I) : qRight (t, 1) = t :=
               /-
                 t : ↑unitInterval
                 ⊢ Eq (unitInterval.qRight { fst := t, snd := 1 }) (Set.projIcc 0 1 ⋯ ↑t)
               -/
  Eq.trans (by rw [qRight]; norm_num) <| Set.projIcc_val zero_le_one _
                            /-
                              🎉 no goals
                            -/


/-- This is the function analogous to the one on p. 475 of [serre1951], defining a homotopy from
the product path `γ ∧ e` to `γ`. -/
def delayReflRight (θ : I) (γ : Path x y) : Path x y where
  toFun t := γ (qRight (t, θ))
  continuous_toFun := γ.continuous.comp (continuous_qRight.comp <| Continuous.Prod.mk_left θ)
  source' := by
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x y : X
      θ : ↑unitInterval
      γ : Path x y
      ⊢ Eq ({ toFun := fun t => γ (unitInterval.qRight { fst := t, snd := θ }), cont …
    -/
    dsimp only
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x y : X
      θ : ↑unitInterval
      γ : Path x y
      ⊢ Eq (γ (unitInterval.qRight { fst := 0, snd := θ })) x
    -/
    rw [qRight_zero_left, γ.source]
    /-
      🎉 no goals
    -/
  target' := by
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x y : X
      θ : ↑unitInterval
      γ : Path x y
      ⊢ Eq ({ toFun := fun t => γ (unitInterval.qRight { fst := t, snd := θ }), cont …
    -/
    dsimp only
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x y : X
      θ : ↑unitInterval
      γ : Path x y
      ⊢ Eq (γ (unitInterval.qRight { fst := 1, snd := θ })) y
    -/
    rw [qRight_one_left, γ.target]
    /-
      🎉 no goals
    -/


theorem continuous_delayReflRight : Continuous fun p : I × Path x y => delayReflRight p.1 p.2 :=
  continuous_uncurry_iff.mp <|
    (continuous_snd.comp continuous_fst).eval <|
      continuous_qRight.comp <| continuous_snd.prod_mk <| continuous_fst.comp continuous_fst


theorem delayReflRight_zero (γ : Path x y) : delayReflRight 0 γ = γ.trans (Path.refl y) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    ⊢ Eq (Path.delayReflRight 0 γ) (γ.trans (Path.refl y))
  -/
  ext t
  simp only [delayReflRight, trans_apply, refl_extend, Path.coe_mk_mk, Function.comp_apply,
    refl_apply]
  /-
    case a.h
    X : Type u
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    t : ↑unitInterval
    ⊢ Eq (γ (unitInterval.qRight { fst := t, snd := 0 })) (dite (LE.le (↑t) (1 / 2 …
  -/
  split_ifs with h; swap
  /-
    case neg
    X : Type u
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    t : ↑unitInterval
    h : Not (LE.le (↑t) (1 / 2))
    ⊢ Eq (γ (unitInterval.qRight { fst := t, snd := 0 })) y
  -/
  on_goal 1 => conv_rhs => rw [← γ.target]
  /-
    case neg
    X : Type u
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    t : ↑unitInterval
    h : Not (LE.le (↑t) (1 / 2))
    ⊢ Eq (γ (unitInterval.qRight { fst := t, snd := 0 })) (γ 1)
  -/
  all_goals apply congr_arg γ; ext1; rw [qRight_zero_right]
  /-
    case neg.a
    X : Type u
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    t : ↑unitInterval
    h : Not (LE.le (↑t) (1 / 2))
    ⊢ Eq (ite (LE.le (↑t) (1 / 2)) (HMul.hMul 2 ↑t) 1) ↑1
  -/
  exacts [if_neg h, if_pos h]
  /-
    🎉 no goals
  -/


theorem delayReflRight_one (γ : Path x y) : delayReflRight 1 γ = γ := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    ⊢ Eq (Path.delayReflRight 1 γ) γ
  -/
  ext t
  /-
    case a.h
    X : Type u
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    t : ↑unitInterval
    ⊢ Eq ((Path.delayReflRight 1 γ) t) (γ t)
  -/
  exact congr_arg γ (qRight_one_right t)
  /-
    🎉 no goals
  -/


/-- This is the function on p. 475 of [serre1951], defining a homotopy from a path `γ` to the
product path `e ∧ γ`. -/
def delayReflLeft (θ : I) (γ : Path x y) : Path x y :=
  (delayReflRight θ γ.symm).symm


theorem continuous_delayReflLeft : Continuous fun p : I × Path x y => delayReflLeft p.1 p.2 :=
  Path.continuous_symm.comp <|
    continuous_delayReflRight.comp <|
      continuous_fst.prod_mk <| Path.continuous_symm.comp continuous_snd


theorem delayReflLeft_zero (γ : Path x y) : delayReflLeft 0 γ = (Path.refl x).trans γ := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    ⊢ Eq (Path.delayReflLeft 0 γ) ((Path.refl x).trans γ)
  -/
  simp only [delayReflLeft, delayReflRight_zero, trans_symm, refl_symm, Path.symm_symm]
  /-
    🎉 no goals
  -/


theorem delayReflLeft_one (γ : Path x y) : delayReflLeft 1 γ = γ := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    ⊢ Eq (Path.delayReflLeft 1 γ) γ
  -/
  simp only [delayReflLeft, delayReflRight_one, Path.symm_symm]
  /-
    🎉 no goals
  -/


/-- The loop space at x carries a structure of an H-space. Note that the field `eHmul`
(resp. `hmulE`) neither implies nor is implied by `Path.Homotopy.reflTrans`
(resp. `Path.Homotopy.transRefl`).
-/
instance (x : X) : HSpace (Path x x) where
  hmul := ⟨fun ρ => ρ.1.trans ρ.2, continuous_trans⟩
  e := refl x
  hmul_e_e := refl_trans_refl
  eHmul :=
    { toHomotopy :=
        ⟨⟨fun p : I × Path x x ↦ delayReflLeft p.1 p.2, continuous_delayReflLeft⟩,
          delayReflLeft_zero, delayReflLeft_one⟩
                  /-
                    X : Type u
                    inst✝ : TopologicalSpace X
                    x✝ y x : X
                    ⊢ ∀ (t : ↑unitInterval) (x_1 : Path x x), Membership.mem (Singleton.singleton  …
                  -/
      prop' := by rintro t _ rfl; exact refl_trans_refl.symm }
                                  /-
                                    🎉 no goals
                                  -/
  hmulE :=
    { toHomotopy :=
        ⟨⟨fun p : I × Path x x ↦ delayReflRight p.1 p.2, continuous_delayReflRight⟩,
          delayReflRight_zero, delayReflRight_one⟩
                  /-
                    X : Type u
                    inst✝ : TopologicalSpace X
                    x✝ y x : X
                    ⊢ ∀ (t : ↑unitInterval) (x_1 : Path x x), Membership.mem (Singleton.singleton  …
                  -/
      prop' := by rintro t _ rfl; exact refl_trans_refl.symm }
                                  /-
                                    🎉 no goals
                                  -/


