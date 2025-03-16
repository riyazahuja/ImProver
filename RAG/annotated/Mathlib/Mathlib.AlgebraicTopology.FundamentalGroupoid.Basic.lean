/-- Auxiliary function for `reflTransSymm`. -/
def reflTransSymmAux (x : I × I) : ℝ :=
  if (x.2 : ℝ) ≤ 1 / 2 then x.1 * 2 * x.2 else x.1 * (2 - 2 * x.2)


@[continuity]
theorem continuous_reflTransSymmAux : Continuous reflTransSymmAux := by
  /-
    ⊢ Continuous Path.Homotopy.reflTransSymmAux
  -/
  refine continuous_if_le ?_ ?_ (Continuous.continuousOn ?_) (Continuous.continuousOn ?_) ?_
    /-
      case refine_1
      ⊢ Continuous fun x => ↑x.2
    -/
  · continuity
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ⊢ Continuous fun x => 1 / 2
    -/
  · continuity
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      ⊢ Continuous fun x => HMul.hMul (HMul.hMul (↑x.1) 2) ↑x.2
    -/
  · continuity
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      ⊢ Continuous fun x => HMul.hMul (↑x.1) (HSub.hSub 2 (HMul.hMul 2 ↑x.2))
    -/
  · continuity
    /-
      🎉 no goals
    -/
  /-
    case refine_5
    ⊢ ∀ (x : Prod ↑unitInterval ↑unitInterval), Eq (↑x.2) (1 / 2) → Eq (HMul.hMul  …
  -/
  intro x hx
  /-
    case refine_5
    x : Prod ↑unitInterval ↑unitInterval
    hx : Eq (↑x.2) (1 / 2)
    ⊢ Eq (HMul.hMul (HMul.hMul (↑x.1) 2) ↑x.2) (HMul.hMul (↑x.1) (HSub.hSub 2 (HMu …
  -/
  norm_num [hx, mul_assoc]
  /-
    🎉 no goals
  -/


theorem reflTransSymmAux_mem_I (x : I × I) : reflTransSymmAux x ∈ I := by
  /-
    x : Prod ↑unitInterval ↑unitInterval
    ⊢ Membership.mem unitInterval (Path.Homotopy.reflTransSymmAux x)
  -/
  dsimp only [reflTransSymmAux]
  /-
    x : Prod ↑unitInterval ↑unitInterval
    ⊢ Membership.mem unitInterval (ite (LE.le (↑x.2) (1 / 2)) (HMul.hMul (HMul.hMu …
  -/
  split_ifs
    /-
      case pos
      x : Prod ↑unitInterval ↑unitInterval
      h✝ : LE.le (↑x.2) (1 / 2)
      ⊢ Membership.mem unitInterval (HMul.hMul (HMul.hMul (↑x.1) 2) ↑x.2)
    -/
  · constructor
      /-
        case pos.left
        x : Prod ↑unitInterval ↑unitInterval
        h✝ : LE.le (↑x.2) (1 / 2)
        ⊢ LE.le 0 (HMul.hMul (HMul.hMul (↑x.1) 2) ↑x.2)
      -/
    · apply mul_nonneg
        /-
          case pos.left.ha
          x : Prod ↑unitInterval ↑unitInterval
          h✝ : LE.le (↑x.2) (1 / 2)
          ⊢ LE.le 0 (HMul.hMul (↑x.1) 2)
        -/
      · apply mul_nonneg
          /-
            case pos.left.ha.ha
            x : Prod ↑unitInterval ↑unitInterval
            h✝ : LE.le (↑x.2) (1 / 2)
            ⊢ LE.le 0 ↑x.1
          -/
        · unit_interval
          /-
            🎉 no goals
          -/
          /-
            case pos.left.ha.hb
            x : Prod ↑unitInterval ↑unitInterval
            h✝ : LE.le (↑x.2) (1 / 2)
            ⊢ LE.le 0 2
          -/
        · norm_num
          /-
            🎉 no goals
          -/
        /-
          case pos.left.hb
          x : Prod ↑unitInterval ↑unitInterval
          h✝ : LE.le (↑x.2) (1 / 2)
          ⊢ LE.le 0 ↑x.2
        -/
      · unit_interval
        /-
          🎉 no goals
        -/
      /-
        case pos.right
        x : Prod ↑unitInterval ↑unitInterval
        h✝ : LE.le (↑x.2) (1 / 2)
        ⊢ LE.le (HMul.hMul (HMul.hMul (↑x.1) 2) ↑x.2) 1
      -/
    · rw [mul_assoc]
      /-
        case pos.right
        x : Prod ↑unitInterval ↑unitInterval
        h✝ : LE.le (↑x.2) (1 / 2)
        ⊢ LE.le (HMul.hMul (↑x.1) (HMul.hMul 2 ↑x.2)) 1
      -/
      apply mul_le_one₀
        /-
          case pos.right.ha
          x : Prod ↑unitInterval ↑unitInterval
          h✝ : LE.le (↑x.2) (1 / 2)
          ⊢ LE.le (↑x.1) 1
        -/
      · unit_interval
        /-
          🎉 no goals
        -/
        /-
          case pos.right.hb₀
          x : Prod ↑unitInterval ↑unitInterval
          h✝ : LE.le (↑x.2) (1 / 2)
          ⊢ LE.le 0 (HMul.hMul 2 ↑x.2)
        -/
      · apply mul_nonneg
          /-
            case pos.right.hb₀.ha
            x : Prod ↑unitInterval ↑unitInterval
            h✝ : LE.le (↑x.2) (1 / 2)
            ⊢ LE.le 0 2
          -/
        · norm_num
          /-
            🎉 no goals
          -/
          /-
            case pos.right.hb₀.hb
            x : Prod ↑unitInterval ↑unitInterval
            h✝ : LE.le (↑x.2) (1 / 2)
            ⊢ LE.le 0 ↑x.2
          -/
        · unit_interval
          /-
            🎉 no goals
          -/
        /-
          case pos.right.hb
          x : Prod ↑unitInterval ↑unitInterval
          h✝ : LE.le (↑x.2) (1 / 2)
          ⊢ LE.le (HMul.hMul 2 ↑x.2) 1
        -/
      · linarith
        /-
          🎉 no goals
        -/
    /-
      case neg
      x : Prod ↑unitInterval ↑unitInterval
      h✝ : Not (LE.le (↑x.2) (1 / 2))
      ⊢ Membership.mem unitInterval (HMul.hMul (↑x.1) (HSub.hSub 2 (HMul.hMul 2 ↑x.2 …
    -/
  · constructor
      /-
        case neg.left
        x : Prod ↑unitInterval ↑unitInterval
        h✝ : Not (LE.le (↑x.2) (1 / 2))
        ⊢ LE.le 0 (HMul.hMul (↑x.1) (HSub.hSub 2 (HMul.hMul 2 ↑x.2)))
      -/
    · apply mul_nonneg
        /-
          case neg.left.ha
          x : Prod ↑unitInterval ↑unitInterval
          h✝ : Not (LE.le (↑x.2) (1 / 2))
          ⊢ LE.le 0 ↑x.1
        -/
      · unit_interval
        /-
          🎉 no goals
        -/
      /-
        case neg.left.hb
        x : Prod ↑unitInterval ↑unitInterval
        h✝ : Not (LE.le (↑x.2) (1 / 2))
        ⊢ LE.le 0 (HSub.hSub 2 (HMul.hMul 2 ↑x.2))
      -/
      linarith [unitInterval.nonneg x.2, unitInterval.le_one x.2]
      /-
        🎉 no goals
      -/
      /-
        case neg.right
        x : Prod ↑unitInterval ↑unitInterval
        h✝ : Not (LE.le (↑x.2) (1 / 2))
        ⊢ LE.le (HMul.hMul (↑x.1) (HSub.hSub 2 (HMul.hMul 2 ↑x.2))) 1
      -/
    · apply mul_le_one₀
        /-
          case neg.right.ha
          x : Prod ↑unitInterval ↑unitInterval
          h✝ : Not (LE.le (↑x.2) (1 / 2))
          ⊢ LE.le (↑x.1) 1
        -/
      · unit_interval
        /-
          🎉 no goals
        -/
        /-
          case neg.right.hb₀
          x : Prod ↑unitInterval ↑unitInterval
          h✝ : Not (LE.le (↑x.2) (1 / 2))
          ⊢ LE.le 0 (HSub.hSub 2 (HMul.hMul 2 ↑x.2))
        -/
      · linarith [unitInterval.nonneg x.2, unitInterval.le_one x.2]
        /-
          🎉 no goals
        -/
        /-
          case neg.right.hb
          x : Prod ↑unitInterval ↑unitInterval
          h✝ : Not (LE.le (↑x.2) (1 / 2))
          ⊢ LE.le (HSub.hSub 2 (HMul.hMul 2 ↑x.2)) 1
        -/
      · linarith [unitInterval.nonneg x.2, unitInterval.le_one x.2]
        /-
          🎉 no goals
        -/


/-- For any path `p` from `x₀` to `x₁`, we have a homotopy from the constant path based at `x₀` to
  `p.trans p.symm`. -/
def reflTransSymm (p : Path x₀ x₁) : Homotopy (Path.refl x₀) (p.trans p.symm) where
  toFun x := p ⟨reflTransSymmAux x, reflTransSymmAux_mem_I x⟩
                         /-
                           X : Type u
                           inst✝ : TopologicalSpace X
                           x₀ x₁ : X
                           p : Path x₀ x₁
                           ⊢ Continuous fun x => p ⟨Path.Homotopy.reflTransSymmAux x, ⋯⟩
                         -/
  continuous_toFun := by continuity
                         /-
                           🎉 no goals
                         -/
                      /-
                        X : Type u
                        inst✝ : TopologicalSpace X
                        x₀ x₁ : X
                        p : Path x₀ x₁
                        ⊢ ∀ (x : ↑unitInterval), Eq ({ toFun := fun x => p ⟨Path.Homotopy.reflTransSym …
                      -/
  map_zero_left := by simp [reflTransSymmAux]
                      /-
                        🎉 no goals
                      -/
  map_one_left x := by
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      p : Path x₀ x₁
      x : ↑unitInterval
      ⊢ Eq ({ toFun := fun x => p ⟨Path.Homotopy.reflTransSymmAux x, ⋯⟩, continuous_ …
    -/
    dsimp only [reflTransSymmAux, Path.coe_toContinuousMap, Path.trans]
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      p : Path x₀ x₁
      x : ↑unitInterval
      ⊢ Eq (p ⟨ite (LE.le (↑x) (1 / 2)) (HMul.hMul (HMul.hMul (↑1) 2) ↑x) (HMul.hMul …
    -/
    change _ = ite _ _ _
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      p : Path x₀ x₁
      x : ↑unitInterval
      ⊢ Eq (p ⟨ite (LE.le (↑x) (1 / 2)) (HMul.hMul (HMul.hMul (↑1) 2) ↑x) (HMul.hMul …
    -/
    split_ifs with h
      /-
        case pos
        X : Type u
        inst✝ : TopologicalSpace X
        x₀ x₁ : X
        p : Path x₀ x₁
        x : ↑unitInterval
        h : LE.le (↑x) (1 / 2)
        ⊢ Eq (p ⟨HMul.hMul (HMul.hMul (↑1) 2) ↑x, ⋯⟩) (p.extend (HMul.hMul 2 ↑x))
      -/
    · rw [Path.extend, Set.IccExtend_of_mem]
        /-
          case pos
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          p : Path x₀ x₁
          x : ↑unitInterval
          h : LE.le (↑x) (1 / 2)
          ⊢ Eq (p ⟨HMul.hMul (HMul.hMul (↑1) 2) ↑x, ⋯⟩) (p ⟨HMul.hMul 2 ↑x, ?pos.hx✝⟩)
        -/
      · norm_num
        /-
          🎉 no goals
        -/
        /-
          case pos.hx
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          p : Path x₀ x₁
          x : ↑unitInterval
          h : LE.le (↑x) (1 / 2)
          ⊢ Membership.mem (Set.Icc 0 1) (HMul.hMul 2 ↑x)
        -/
      · rw [unitInterval.mul_pos_mem_iff zero_lt_two]
        /-
          case pos.hx
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          p : Path x₀ x₁
          x : ↑unitInterval
          h : LE.le (↑x) (1 / 2)
          ⊢ Membership.mem (Set.Icc 0 (1 / 2)) ↑x
        -/
        exact ⟨unitInterval.nonneg x, h⟩
        /-
          🎉 no goals
        -/
      /-
        case neg
        X : Type u
        inst✝ : TopologicalSpace X
        x₀ x₁ : X
        p : Path x₀ x₁
        x : ↑unitInterval
        h : Not (LE.le (↑x) (1 / 2))
        ⊢ Eq (p ⟨HMul.hMul (↑1) (HSub.hSub 2 (HMul.hMul 2 ↑x)), ⋯⟩) (p.symm.extend (HS …
      -/
    · rw [Path.symm, Path.extend, Set.IccExtend_of_mem]
        /-
          case neg
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          p : Path x₀ x₁
          x : ↑unitInterval
          h : Not (LE.le (↑x) (1 / 2))
          ⊢ Eq (p ⟨HMul.hMul (↑1) (HSub.hSub 2 (HMul.hMul 2 ↑x)), ⋯⟩) ({ toFun := Functi …
        -/
      · simp only [Set.Icc.coe_one, one_mul, coe_mk_mk, Function.comp_apply]
        /-
          case neg
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          p : Path x₀ x₁
          x : ↑unitInterval
          h : Not (LE.le (↑x) (1 / 2))
          ⊢ Eq (p ⟨HSub.hSub 2 (HMul.hMul 2 ↑x), ⋯⟩) (p (unitInterval.symm ⟨HSub.hSub (H …
        -/
        congr 1
        /-
          case neg.h.e_6.h
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          p : Path x₀ x₁
          x : ↑unitInterval
          h : Not (LE.le (↑x) (1 / 2))
          ⊢ Eq ⟨HSub.hSub 2 (HMul.hMul 2 ↑x), ⋯⟩ (unitInterval.symm ⟨HSub.hSub (HMul.hMu …
        -/
        ext
        /-
          case neg.h.e_6.h.a
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          p : Path x₀ x₁
          x : ↑unitInterval
          h : Not (LE.le (↑x) (1 / 2))
          ⊢ Eq ↑⟨HSub.hSub 2 (HMul.hMul 2 ↑x), ⋯⟩ ↑(unitInterval.symm ⟨HSub.hSub (HMul.h …
        -/
        norm_num [sub_sub_eq_add_sub]
        /-
          🎉 no goals
        -/
        /-
          case neg.hx
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          p : Path x₀ x₁
          x : ↑unitInterval
          h : Not (LE.le (↑x) (1 / 2))
          ⊢ Membership.mem (Set.Icc 0 1) (HSub.hSub (HMul.hMul 2 ↑x) 1)
        -/
      · rw [unitInterval.two_mul_sub_one_mem_iff]
        /-
          case neg.hx
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          p : Path x₀ x₁
          x : ↑unitInterval
          h : Not (LE.le (↑x) (1 / 2))
          ⊢ Membership.mem (Set.Icc (1 / 2) 1) ↑x
        -/
        exact ⟨(not_le.1 h).le, unitInterval.le_one x⟩
        /-
          🎉 no goals
        -/
  prop' t x hx := by
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      p : Path x₀ x₁
      t x : ↑unitInterval
      hx : Membership.mem (Insert.insert 0 (Singleton.singleton 1)) x
      ⊢ Eq ({ toFun := fun x => { toFun := fun x => p ⟨Path.Homotopy.reflTransSymmAu …
    -/
    simp only [Set.mem_singleton_iff, Set.mem_insert_iff] at hx
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      p : Path x₀ x₁
      t x : ↑unitInterval
      hx : Or (Eq x 0) (Eq x 1)
      ⊢ Eq ({ toFun := fun x => { toFun := fun x => p ⟨Path.Homotopy.reflTransSymmAu …
    -/
    simp only [ContinuousMap.coe_mk, coe_toContinuousMap, Path.refl_apply]
    cases hx with
    | inl hx
    | inr hx =>
      rw [hx]
      norm_num [reflTransSymmAux]


/-- For any path `p` from `x₀` to `x₁`, we have a homotopy from the constant path based at `x₁` to
  `p.symm.trans p`. -/
def reflSymmTrans (p : Path x₀ x₁) : Homotopy (Path.refl x₁) (p.symm.trans p) :=
  (reflTransSymm p.symm).cast rfl <| congr_arg _ (Path.symm_symm _)


/-- Auxiliary function for `trans_refl_reparam`. -/
def transReflReparamAux (t : I) : ℝ :=
  if (t : ℝ) ≤ 1 / 2 then 2 * t else 1


@[continuity]
theorem continuous_transReflReparamAux : Continuous transReflReparamAux := by
  refine continuous_if_le ?_ ?_ (Continuous.continuousOn ?_) (Continuous.continuousOn ?_) ?_ <;>
    [continuity; continuity; continuity; continuity; skip]
  /-
    case refine_5
    ⊢ ∀ (x : ↑unitInterval), Eq (↑x) (1 / 2) → Eq (HMul.hMul 2 ↑x) 1
  -/
  intro x hx
  /-
    case refine_5
    x : ↑unitInterval
    hx : Eq (↑x) (1 / 2)
    ⊢ Eq (HMul.hMul 2 ↑x) 1
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


theorem transReflReparamAux_mem_I (t : I) : transReflReparamAux t ∈ I := by
  /-
    t : ↑unitInterval
    ⊢ Membership.mem unitInterval (Path.Homotopy.transReflReparamAux t)
  -/
  unfold transReflReparamAux
  /-
    t : ↑unitInterval
    ⊢ Membership.mem unitInterval (ite (LE.le (↑t) (1 / 2)) (HMul.hMul 2 ↑t) 1)
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
  split_ifs <;> constructor <;> linarith [unitInterval.le_one t, unitInterval.nonneg t]
                                /-
                                  🎉 no goals
                                -/


theorem transReflReparamAux_zero : transReflReparamAux 0 = 0 := by
  /-
    ⊢ Eq (Path.Homotopy.transReflReparamAux 0) 0
  -/
  norm_num [transReflReparamAux]
  /-
    🎉 no goals
  -/


theorem transReflReparamAux_one : transReflReparamAux 1 = 1 := by
  /-
    ⊢ Eq (Path.Homotopy.transReflReparamAux 1) 1
  -/
  norm_num [transReflReparamAux]
  /-
    🎉 no goals
  -/


theorem trans_refl_reparam (p : Path x₀ x₁) :
    p.trans (Path.refl x₁) =
                                                                                    /-
                                                                                      X : Type u
                                                                                      inst✝ : TopologicalSpace X
                                                                                      x₀ x₁ : X
                                                                                      p : Path x₀ x₁
                                                                                      ⊢ Continuous fun t => ⟨Path.Homotopy.transReflReparamAux t, ⋯⟩
                                                                                    -/
      p.reparam (fun t => ⟨transReflReparamAux t, transReflReparamAux_mem_I t⟩) (by continuity)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
        (Subtype.ext transReflReparamAux_zero) (Subtype.ext transReflReparamAux_one) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x₀ x₁ : X
    p : Path x₀ x₁
    ⊢ Eq (p.trans (Path.refl x₁)) (p.reparam (fun t => ⟨Path.Homotopy.transReflRep …
  -/
  ext
  /-
    case a.h
    X : Type u
    inst✝ : TopologicalSpace X
    x₀ x₁ : X
    p : Path x₀ x₁
    x✝ : ↑unitInterval
    ⊢ Eq ((p.trans (Path.refl x₁)) x✝) ((p.reparam (fun t => ⟨Path.Homotopy.transR …
  -/
  unfold transReflReparamAux
  /-
    case a.h
    X : Type u
    inst✝ : TopologicalSpace X
    x₀ x₁ : X
    p : Path x₀ x₁
    x✝ : ↑unitInterval
    ⊢ Eq ((p.trans (Path.refl x₁)) x✝) ((p.reparam (fun t => ⟨ite (LE.le (↑t) (1 / …
  -/
  simp only [Path.trans_apply, not_le, coe_reparam, Function.comp_apply, one_div, Path.refl_apply]
  /-
    case a.h
    X : Type u
    inst✝ : TopologicalSpace X
    x₀ x₁ : X
    p : Path x₀ x₁
    x✝ : ↑unitInterval
    ⊢ Eq (dite (LE.le (↑x✝) (Inv.inv 2)) (fun h => p ⟨HMul.hMul 2 ↑x✝, ⋯⟩) fun h = …
  -/
  split_ifs
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      p : Path x₀ x₁
      x✝ : ↑unitInterval
      h✝¹ : LE.le (↑x✝) (Inv.inv 2)
      h✝ : LE.le (↑x✝) (Inv.inv 2)
      ⊢ Eq (p ⟨HMul.hMul 2 ↑x✝, ⋯⟩) (p ⟨HMul.hMul 2 ↑x✝, ⋯⟩)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      p : Path x₀ x₁
      x✝ : ↑unitInterval
      h✝¹ : LE.le (↑x✝) (Inv.inv 2)
      h✝ : Not (LE.le (↑x✝) (Inv.inv 2))
      ⊢ Eq (p ⟨HMul.hMul 2 ↑x✝, ⋯⟩) (p ⟨HMul.hMul 2 ↑x✝, ⋯⟩)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      p : Path x₀ x₁
      x✝ : ↑unitInterval
      h✝¹ : Not (LE.le (↑x✝) (Inv.inv 2))
      h✝ : LE.le (↑x✝) (Inv.inv 2)
      ⊢ Eq x₁ (p ⟨1, ⋯⟩)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      p : Path x₀ x₁
      x✝ : ↑unitInterval
      h✝¹ : Not (LE.le (↑x✝) (Inv.inv 2))
      h✝ : Not (LE.le (↑x✝) (Inv.inv 2))
      ⊢ Eq x₁ (p ⟨1, ⋯⟩)
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- For any path `p` from `x₀` to `x₁`, we have a homotopy from `p.trans (Path.refl x₁)` to `p`. -/
def transRefl (p : Path x₀ x₁) : Homotopy (p.trans (Path.refl x₁)) p :=
  ((Homotopy.reparam p (fun t => ⟨transReflReparamAux t, transReflReparamAux_mem_I t⟩)
              /-
                X : Type u
                inst✝ : TopologicalSpace X
                x₀ x₁ : X
                p : Path x₀ x₁
                ⊢ Continuous fun t => ⟨Path.Homotopy.transReflReparamAux t, ⋯⟩
              -/
          (by continuity) (Subtype.ext transReflReparamAux_zero)
              /-
                🎉 no goals
              -/
          (Subtype.ext transReflReparamAux_one)).cast
      rfl (trans_refl_reparam p).symm).symm


/-- For any path `p` from `x₀` to `x₁`, we have a homotopy from `(Path.refl x₀).trans p` to `p`. -/
def reflTrans (p : Path x₀ x₁) : Homotopy ((Path.refl x₀).trans p) p :=
                                    /-
                                      X : Type u
                                      inst✝ : TopologicalSpace X
                                      x₀ x₁ : X
                                      p : Path x₀ x₁
                                      ⊢ Eq (p.symm.trans (Path.refl x₀)).symm ((Path.refl x₀).trans p)
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  (transRefl p.symm).symm₂.cast (by simp) (by simp)
                                              /-
                                                🎉 no goals
                                              -/


/-- Auxiliary function for `trans_assoc_reparam`. -/
def transAssocReparamAux (t : I) : ℝ :=
  if (t : ℝ) ≤ 1 / 4 then 2 * t else if (t : ℝ) ≤ 1 / 2 then t + 1 / 4 else 1 / 2 * (t + 1)


@[continuity]
theorem continuous_transAssocReparamAux : Continuous transAssocReparamAux := by
  refine continuous_if_le ?_ ?_ (Continuous.continuousOn ?_)
    (continuous_if_le ?_ ?_
      (Continuous.continuousOn ?_) (Continuous.continuousOn ?_) ?_).continuousOn
      ?_ <;>
    [continuity; continuity; continuity; continuity; continuity; continuity; continuity; skip;
      skip] <;>
      /-
        case refine_8
        ⊢ ∀ (x : ↑unitInterval), Eq (↑x) (1 / 2) → Eq (HAdd.hAdd (↑x) (1 / 4)) (HMul.h …
      -/
      /-
        case refine_8
        x : ↑unitInterval
        hx : Eq (↑x) (1 / 2)
        ⊢ Eq (HAdd.hAdd (↑x) (1 / 4)) (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1))
      -/
      /-
        🎉 no goals
      -/
      /-
        case refine_9
        x : ↑unitInterval
        hx : Eq (↑x) (1 / 4)
        ⊢ Eq (HMul.hMul 2 ↑x) (ite (LE.le (↑x) (1 / 2)) (HAdd.hAdd (↑x) (1 / 4)) (HMul …
      -/
      norm_num [hx]
      /-
        🎉 no goals
      -/


theorem transAssocReparamAux_mem_I (t : I) : transAssocReparamAux t ∈ I := by
  /-
    t : ↑unitInterval
    ⊢ Membership.mem unitInterval (Path.Homotopy.transAssocReparamAux t)
  -/
  unfold transAssocReparamAux
  /-
    t : ↑unitInterval
    ⊢ Membership.mem unitInterval (ite (LE.le (↑t) (1 / 4)) (HMul.hMul 2 ↑t) (ite  …
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
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  split_ifs <;> constructor <;> linarith [unitInterval.le_one t, unitInterval.nonneg t]
                                /-
                                  🎉 no goals
                                -/


theorem transAssocReparamAux_zero : transAssocReparamAux 0 = 0 := by
  /-
    ⊢ Eq (Path.Homotopy.transAssocReparamAux 0) 0
  -/
  norm_num [transAssocReparamAux]
  /-
    🎉 no goals
  -/


theorem transAssocReparamAux_one : transAssocReparamAux 1 = 1 := by
  /-
    ⊢ Eq (Path.Homotopy.transAssocReparamAux 1) 1
  -/
  norm_num [transAssocReparamAux]
  /-
    🎉 no goals
  -/


theorem trans_assoc_reparam {x₀ x₁ x₂ x₃ : X} (p : Path x₀ x₁) (q : Path x₁ x₂) (r : Path x₂ x₃) :
    (p.trans q).trans r =
      (p.trans (q.trans r)).reparam
                                                                              /-
                                                                                X : Type u
                                                                                inst✝ : TopologicalSpace X
                                                                                x₀✝ x₁✝ x₀ x₁ x₂ x₃ : X
                                                                                p : Path x₀ x₁
                                                                                q : Path x₁ x₂
                                                                                r : Path x₂ x₃
                                                                                ⊢ Continuous fun t => ⟨Path.Homotopy.transAssocReparamAux t, ⋯⟩
                                                                              -/
        (fun t => ⟨transAssocReparamAux t, transAssocReparamAux_mem_I t⟩) (by continuity)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
        (Subtype.ext transAssocReparamAux_zero) (Subtype.ext transAssocReparamAux_one) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x₀ x₁ x₂ x₃ : X
    p : Path x₀ x₁
    q : Path x₁ x₂
    r : Path x₂ x₃
    ⊢ Eq ((p.trans q).trans r) ((p.trans (q.trans r)).reparam (fun t => ⟨Path.Homo …
  -/
  ext x
  simp only [transAssocReparamAux, Path.trans_apply, mul_inv_cancel_left₀, not_le,
    Function.comp_apply, Ne, not_false_iff, one_ne_zero, mul_ite, Subtype.coe_mk,
    Path.coe_reparam]
  -- TODO: why does split_ifs not reduce the ifs??????
  /-
    case a.h
    X : Type u
    inst✝ : TopologicalSpace X
    x₀ x₁ x₂ x₃ : X
    p : Path x₀ x₁
    q : Path x₁ x₂
    r : Path x₂ x₃
    x : ↑unitInterval
    ⊢ Eq (dite (LE.le (↑x) (1 / 2)) (fun h => dite (LE.le (HMul.hMul 2 ↑x) (1 / 2) …
  -/
  split_ifs with h₁ h₂ h₃ h₄ h₅
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : LE.le (HMul.hMul 2 ↑x) (1 / 2)
      h₃ : LE.le (↑x) (1 / 4)
      ⊢ Eq (p ⟨HMul.hMul 2 (HMul.hMul 2 ↑x), ⋯⟩) (p ⟨HMul.hMul 2 (HMul.hMul 2 ↑x), ⋯⟩)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : LE.le (HMul.hMul 2 ↑x) (1 / 2)
      h₃ : Not (LE.le (↑x) (1 / 4))
      h₄ : LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2)
      ⊢ Eq (p ⟨HMul.hMul 2 (HMul.hMul 2 ↑x), ⋯⟩) (p ⟨HMul.hMul 2 (HAdd.hAdd (↑x) (1  …
    -/
  · exfalso
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : LE.le (HMul.hMul 2 ↑x) (1 / 2)
      h₃ : Not (LE.le (↑x) (1 / 4))
      h₄ : LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2)
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : LE.le (HMul.hMul 2 ↑x) (1 / 2)
      h₃ : Not (LE.le (↑x) (1 / 4))
      h₄ : Not (LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2))
      h₅ : LE.le (HSub.hSub (HMul.hMul 2 (HAdd.hAdd (↑x) (1 / 4))) 1) (1 / 2)
      ⊢ Eq (p ⟨HMul.hMul 2 (HMul.hMul 2 ↑x), ⋯⟩) (q ⟨HMul.hMul 2 (HSub.hSub (HMul.hM …
    -/
  · exfalso
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : LE.le (HMul.hMul 2 ↑x) (1 / 2)
      h₃ : Not (LE.le (↑x) (1 / 4))
      h₄ : Not (LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2))
      h₅ : LE.le (HSub.hSub (HMul.hMul 2 (HAdd.hAdd (↑x) (1 / 4))) 1) (1 / 2)
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : LE.le (HMul.hMul 2 ↑x) (1 / 2)
      h₃ : Not (LE.le (↑x) (1 / 4))
      h₄ : Not (LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2))
      h₅ : Not (LE.le (HSub.hSub (HMul.hMul 2 (HAdd.hAdd (↑x) (1 / 4))) 1) (1 / 2))
      ⊢ Eq (p ⟨HMul.hMul 2 (HMul.hMul 2 ↑x), ⋯⟩) (r ⟨HSub.hSub (HMul.hMul 2 (HSub.hS …
    -/
  · exfalso
    /-
      case neg
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : LE.le (HMul.hMul 2 ↑x) (1 / 2)
      h₃ : Not (LE.le (↑x) (1 / 4))
      h₄ : Not (LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2))
      h₅ : Not (LE.le (HSub.hSub (HMul.hMul 2 (HAdd.hAdd (↑x) (1 / 4))) 1) (1 / 2))
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝¹ : LE.le (↑x) (1 / 4)
      h✝ : LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1) (1 / 2)
      ⊢ Eq (q ⟨HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1, ⋯⟩) (q ⟨HMul.hMul 2 (HSub …
    -/
  · exfalso
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝¹ : LE.le (↑x) (1 / 4)
      h✝ : LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1) (1 / 2)
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝¹ : LE.le (↑x) (1 / 4)
      h✝ : Not (LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1) (1 / 2))
      ⊢ Eq (q ⟨HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1, ⋯⟩) (r ⟨HSub.hSub (HMul.h …
    -/
  · exfalso
    /-
      case neg
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝¹ : LE.le (↑x) (1 / 4)
      h✝ : Not (LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1) (1 / 2))
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝¹ : Not (LE.le (↑x) (1 / 4))
      h✝ : LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2)
      ⊢ Eq (q ⟨HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1, ⋯⟩) (p ⟨HMul.hMul 2 (HAdd …
    -/
  · exfalso
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝¹ : Not (LE.le (↑x) (1 / 4))
      h✝ : LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2)
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝² : Not (LE.le (↑x) (1 / 4))
      h✝¹ : Not (LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2))
      h✝ : LE.le (HSub.hSub (HMul.hMul 2 (HAdd.hAdd (↑x) (1 / 4))) 1) (1 / 2)
      ⊢ Eq (q ⟨HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1, ⋯⟩) (q ⟨HMul.hMul 2 (HSub …
    -/
  · have h : 2 * (2 * (x : ℝ)) - 1 = 2 * (2 * (↑x + 1 / 4) - 1) := by linarith
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝² : Not (LE.le (↑x) (1 / 4))
      h✝¹ : Not (LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2))
      h✝ : LE.le (HSub.hSub (HMul.hMul 2 (HAdd.hAdd (↑x) (1 / 4))) 1) (1 / 2)
      h : Eq (HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1) (HMul.hMul 2 (HSub.hSub (H …
      ⊢ Eq (q ⟨HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1, ⋯⟩) (q ⟨HMul.hMul 2 (HSub …
    -/
    simp [h₂, h₁, h, dif_neg (show ¬False from id), dif_pos True.intro, if_false, if_true]
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝² : Not (LE.le (↑x) (1 / 4))
      h✝¹ : Not (LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2))
      h✝ : Not (LE.le (HSub.hSub (HMul.hMul 2 (HAdd.hAdd (↑x) (1 / 4))) 1) (1 / 2))
      ⊢ Eq (q ⟨HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1, ⋯⟩) (r ⟨HSub.hSub (HMul.h …
    -/
  · exfalso
    /-
      case neg
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : LE.le (↑x) (1 / 2)
      h₂ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝² : Not (LE.le (↑x) (1 / 4))
      h✝¹ : Not (LE.le (HAdd.hAdd (↑x) (1 / 4)) (1 / 2))
      h✝ : Not (LE.le (HSub.hSub (HMul.hMul 2 (HAdd.hAdd (↑x) (1 / 4))) 1) (1 / 2))
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝¹ : LE.le (↑x) (1 / 4)
      h✝ : LE.le (HMul.hMul 2 ↑x) (1 / 2)
      ⊢ Eq (r ⟨HSub.hSub (HMul.hMul 2 ↑x) 1, ⋯⟩) (p ⟨HMul.hMul 2 (HMul.hMul 2 ↑x), ⋯⟩)
    -/
  · exfalso
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝¹ : LE.le (↑x) (1 / 4)
      h✝ : LE.le (HMul.hMul 2 ↑x) (1 / 2)
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝² : LE.le (↑x) (1 / 4)
      h✝¹ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝ : LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1) (1 / 2)
      ⊢ Eq (r ⟨HSub.hSub (HMul.hMul 2 ↑x) 1, ⋯⟩) (q ⟨HMul.hMul 2 (HSub.hSub (HMul.hM …
    -/
  · exfalso
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝² : LE.le (↑x) (1 / 4)
      h✝¹ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝ : LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1) (1 / 2)
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝² : LE.le (↑x) (1 / 4)
      h✝¹ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝ : Not (LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1) (1 / 2))
      ⊢ Eq (r ⟨HSub.hSub (HMul.hMul 2 ↑x) 1, ⋯⟩) (r ⟨HSub.hSub (HMul.hMul 2 (HSub.hS …
    -/
  · exfalso
    /-
      case neg
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝² : LE.le (↑x) (1 / 4)
      h✝¹ : Not (LE.le (HMul.hMul 2 ↑x) (1 / 2))
      h✝ : Not (LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul 2 ↑x)) 1) (1 / 2))
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝¹ : Not (LE.le (↑x) (1 / 4))
      h✝ : LE.le (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1)) (1 / 2)
      ⊢ Eq (r ⟨HSub.hSub (HMul.hMul 2 ↑x) 1, ⋯⟩) (p ⟨HMul.hMul 2 (HMul.hMul (1 / 2)  …
    -/
  · exfalso
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝¹ : Not (LE.le (↑x) (1 / 4))
      h✝ : LE.le (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1)) (1 / 2)
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝² : Not (LE.le (↑x) (1 / 4))
      h✝¹ : Not (LE.le (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1)) (1 / 2))
      h✝ : LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1))) 1)  …
      ⊢ Eq (r ⟨HSub.hSub (HMul.hMul 2 ↑x) 1, ⋯⟩) (q ⟨HMul.hMul 2 (HSub.hSub (HMul.hM …
    -/
  · exfalso
    /-
      case pos
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝² : Not (LE.le (↑x) (1 / 4))
      h✝¹ : Not (LE.le (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1)) (1 / 2))
      h✝ : LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1))) 1)  …
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝² : Not (LE.le (↑x) (1 / 4))
      h✝¹ : Not (LE.le (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1)) (1 / 2))
      h✝ : Not (LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1)) …
      ⊢ Eq (r ⟨HSub.hSub (HMul.hMul 2 ↑x) 1, ⋯⟩) (r ⟨HSub.hSub (HMul.hMul 2 (HSub.hS …
    -/
  · congr
    /-
      case neg.h.e_6.h.e_val.e_a.e_a
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ x₂ x₃ : X
      p : Path x₀ x₁
      q : Path x₁ x₂
      r : Path x₂ x₃
      x : ↑unitInterval
      h₁ : Not (LE.le (↑x) (1 / 2))
      h✝² : Not (LE.le (↑x) (1 / 4))
      h✝¹ : Not (LE.le (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1)) (1 / 2))
      h✝ : Not (LE.le (HSub.hSub (HMul.hMul 2 (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1)) …
      ⊢ Eq (↑x) (HSub.hSub (HMul.hMul 2 (HMul.hMul (1 / 2) (HAdd.hAdd (↑x) 1))) 1)
    -/
    ring
    /-
      🎉 no goals
    -/


/-- For paths `p q r`, we have a homotopy from `(p.trans q).trans r` to `p.trans (q.trans r)`. -/
def transAssoc {x₀ x₁ x₂ x₃ : X} (p : Path x₀ x₁) (q : Path x₁ x₂) (r : Path x₂ x₃) :
    Homotopy ((p.trans q).trans r) (p.trans (q.trans r)) :=
  ((Homotopy.reparam (p.trans (q.trans r))
                                                                                /-
                                                                                  X : Type u
                                                                                  inst✝ : TopologicalSpace X
                                                                                  x₀✝ x₁✝ x₀ x₁ x₂ x₃ : X
                                                                                  p : Path x₀ x₁
                                                                                  q : Path x₁ x₂
                                                                                  r : Path x₂ x₃
                                                                                  ⊢ Continuous fun t => ⟨Path.Homotopy.transAssocReparamAux t, ⋯⟩
                                                                                -/
          (fun t => ⟨transAssocReparamAux t, transAssocReparamAux_mem_I t⟩) (by continuity)
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
          (Subtype.ext transAssocReparamAux_zero) (Subtype.ext transAssocReparamAux_one)).cast
      rfl (trans_assoc_reparam p q r).symm).symm


/-- The fundamental groupoid of a space `X` is defined to be a wrapper around `X`, and we
subsequently put a `CategoryTheory.Groupoid` structure on it. -/
@[ext]
structure FundamentalGroupoid (X : Type u) where
  /-- View a term of `FundamentalGroupoid X` as a term of `X`. -/
  as : X


/-- The equivalence between `X` and the underlying type of its fundamental groupoid.
  This is useful for transferring constructions (instances, etc.)
  from `X` to `πₓ X`. -/
@[simps]
def equiv (X : Type*) : FundamentalGroupoid X ≃ X where
  toFun x := x.as
  invFun x := .mk x
  left_inv _ := rfl
  right_inv _ := rfl


@[simp]
lemma isEmpty_iff (X : Type*) :
    IsEmpty (FundamentalGroupoid X) ↔ IsEmpty X :=
  equiv _ |>.isEmpty_congr


instance (X : Type*) [IsEmpty X] :
    IsEmpty (FundamentalGroupoid X) :=
  equiv _ |>.isEmpty


@[simp]
lemma nonempty_iff (X : Type*) :
    Nonempty (FundamentalGroupoid X) ↔ Nonempty X :=
  equiv _ |>.nonempty_congr


instance (X : Type*) [Nonempty X] :
    Nonempty (FundamentalGroupoid X) :=
  equiv _ |>.nonempty


@[simp]
lemma subsingleton_iff (X : Type*) :
    Subsingleton (FundamentalGroupoid X) ↔ Subsingleton X :=
  equiv _ |>.subsingleton_congr


instance (X : Type*) [Subsingleton X] :
    Subsingleton (FundamentalGroupoid X) :=
  equiv _ |>.subsingleton

-- TODO: It seems that `Equiv.nontrivial_congr` doesn't exist.
-- Once it is added, please add the corresponding lemma and instance.


instance {X : Type u} [Inhabited X] : Inhabited (FundamentalGroupoid X) :=
  ⟨⟨default⟩⟩


instance : CategoryTheory.Groupoid (FundamentalGroupoid X) where
  Hom x y := Path.Homotopic.Quotient x.as y.as
  id x := ⟦Path.refl x.as⟧
  comp {_ _ _} := Path.Homotopic.Quotient.comp
  id_comp {x _} f :=
    Quotient.inductionOn f fun a =>
      show ⟦(Path.refl x.as).trans a⟧ = ⟦a⟧ from Quotient.sound ⟨Path.Homotopy.reflTrans a⟩
  comp_id {_ y} f :=
    Quotient.inductionOn f fun a =>
      show ⟦a.trans (Path.refl y.as)⟧ = ⟦a⟧ from Quotient.sound ⟨Path.Homotopy.transRefl a⟩
  assoc {_ _ _ _} f g h :=
    Quotient.inductionOn₃ f g h fun p q r =>
      show ⟦(p.trans q).trans r⟧ = ⟦p.trans (q.trans r)⟧ from
        Quotient.sound ⟨Path.Homotopy.transAssoc p q r⟩
  inv {x y} p :=
    Quotient.lift (fun l : Path x.as y.as => ⟦l.symm⟧)
      (by
        /-
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          x y : FundamentalGroupoid X
          p : Quiver.Hom x y
          ⊢ ∀ (a b : Path x.as y.as), HasEquiv.Equiv a b → Eq ((fun l => Quotient.mk (Pa …
        -/
        rintro a b ⟨h⟩
        /-
          case intro
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          x y : FundamentalGroupoid X
          p : Quiver.Hom x y
          a b : Path x.as y.as
          h : a.Homotopy b
          ⊢ Eq ((fun l => Quotient.mk (Path.Homotopic.setoid y.as x.as) l.symm) a) ((fun …
        -/
        simp only
        /-
          case intro
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          x y : FundamentalGroupoid X
          p : Quiver.Hom x y
          a b : Path x.as y.as
          h : a.Homotopy b
          ⊢ Eq (Quotient.mk (Path.Homotopic.setoid y.as x.as) a.symm) (Quotient.mk (Path …
        -/
        rw [Quotient.eq]
        /-
          case intro
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          x y : FundamentalGroupoid X
          p : Quiver.Hom x y
          a b : Path x.as y.as
          h : a.Homotopy b
          ⊢ (Path.Homotopic.setoid y.as x.as) a.symm b.symm
        -/
        exact ⟨h.symm₂⟩)
        /-
          🎉 no goals
        -/
      p
  inv_comp {_ y} f :=
    Quotient.inductionOn f fun a =>
      show ⟦a.symm.trans a⟧ = ⟦Path.refl y.as⟧ from
        Quotient.sound ⟨(Path.Homotopy.reflSymmTrans a).symm⟩
  comp_inv {x _} f :=
    Quotient.inductionOn f fun a =>
      show ⟦a.trans a.symm⟧ = ⟦Path.refl x.as⟧ from
        Quotient.sound ⟨(Path.Homotopy.reflTransSymm a).symm⟩


theorem comp_eq (x y z : FundamentalGroupoid X) (p : x ⟶ y) (q : y ⟶ z) : p ≫ q = p.comp q := rfl


theorem id_eq_path_refl (x : FundamentalGroupoid X) : 𝟙 x = ⟦Path.refl x.as⟧ := rfl


/-- The functor sending a topological space `X` to its fundamental groupoid. -/
def fundamentalGroupoidFunctor : TopCat ⥤ CategoryTheory.Grpd where
  obj X := { α := FundamentalGroupoid X }
  map f :=
    { obj := fun x => ⟨f x.as⟩
                               /-
                                 X✝¹ : Type u
                                 inst✝ : TopologicalSpace X✝¹
                                 x₀ x₁ : X✝¹
                                 X✝ Y✝ : TopCat
                                 f : Quiver.Hom X✝ Y✝
                                 X Y : ↑((fun X => { α := FundamentalGroupoid ↑X, str := inferInstance }) X✝)
                                 p : Quiver.Hom X Y
                                 ⊢ Quiver.Hom ((fun x => { as := f x.as }) X) ((fun x => { as := f x.as }) Y)
                               -/
      map := fun {X Y} p => by exact Path.Homotopic.Quotient.mapFn p f
                               /-
                                 🎉 no goals
                               -/
      map_id := fun _ => rfl
      map_comp := fun {x y z} p q => by
        /-
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          X✝ Y✝ : TopCat
          f : Quiver.Hom X✝ Y✝
          x y z : ↑((fun X => { α := FundamentalGroupoid ↑X, str := inferInstance }) X✝)
          p : Quiver.Hom x y
          q : Quiver.Hom y z
          ⊢ Eq ({ obj := fun x => { as := f x.as }, map := fun {X Y} p => Path.Homotopic …
        -/
        refine Quotient.inductionOn₂ p q fun a b => ?_
        /-
          X : Type u
          inst✝ : TopologicalSpace X
          x₀ x₁ : X
          X✝ Y✝ : TopCat
          f : Quiver.Hom X✝ Y✝
          x y z : ↑((fun X => { α := FundamentalGroupoid ↑X, str := inferInstance }) X✝)
          p : Quiver.Hom x y
          q : Quiver.Hom y z
          a : Path x.as y.as
          b : Path y.as z.as
          ⊢ Eq ({ obj := fun x => { as := f x.as }, map := fun {X Y} p => Path.Homotopic …
        -/
        simp only [comp_eq, ← Path.Homotopic.map_lift, ← Path.Homotopic.comp_lift, Path.map_trans] }
        /-
          🎉 no goals
        -/
  map_id X := by
    /-
      X✝ : Type u
      inst✝ : TopologicalSpace X✝
      x₀ x₁ : X✝
      X : TopCat
      ⊢ Eq ({ obj := fun X => { α := FundamentalGroupoid ↑X, str := inferInstance }, …
    -/
    simp only
    /-
      X✝ : Type u
      inst✝ : TopologicalSpace X✝
      x₀ x₁ : X✝
      X : TopCat
      ⊢ Eq { obj := fun x => { as := (CategoryTheory.CategoryStruct.id X) x.as }, ma …
    -/
    change _ = (⟨_, _, _⟩ : FundamentalGroupoid X ⥤ FundamentalGroupoid X)
    /-
      X✝ : Type u
      inst✝ : TopologicalSpace X✝
      x₀ x₁ : X✝
      X : TopCat
      ⊢ Eq { obj := fun x => { as := (CategoryTheory.CategoryStruct.id X) x.as }, ma …
    -/
    congr
    /-
      case e_toPrefunctor.e_map
      X✝ : Type u
      inst✝ : TopologicalSpace X✝
      x₀ x₁ : X✝
      X : TopCat
      ⊢ Eq (fun {X_1 Y} p => Path.Homotopic.Quotient.mapFn p (CategoryTheory.Categor …
    -/
    ext x y p
    /-
      case e_toPrefunctor.e_map.h.h.h
      X✝ : Type u
      inst✝ : TopologicalSpace X✝
      x₀ x₁ : X✝
      X : TopCat
      x y : FundamentalGroupoid ↑X
      p : Quiver.Hom x y
      ⊢ Eq (Path.Homotopic.Quotient.mapFn p (CategoryTheory.CategoryStruct.id X)) p
    -/
    refine Quotient.inductionOn p fun q => ?_
    /-
      case e_toPrefunctor.e_map.h.h.h
      X✝ : Type u
      inst✝ : TopologicalSpace X✝
      x₀ x₁ : X✝
      X : TopCat
      x y : FundamentalGroupoid ↑X
      p : Quiver.Hom x y
      q : Path x.as y.as
      ⊢ Eq (Path.Homotopic.Quotient.mapFn (Quotient.mk (Path.Homotopic.setoid x.as y …
    -/
    rw [← Path.Homotopic.map_lift]
    /-
      case e_toPrefunctor.e_map.h.h.h
      X✝ : Type u
      inst✝ : TopologicalSpace X✝
      x₀ x₁ : X✝
      X : TopCat
      x y : FundamentalGroupoid ↑X
      p : Quiver.Hom x y
      q : Path x.as y.as
      ⊢ Eq (Quotient.mk (Path.Homotopic.setoid ((CategoryTheory.CategoryStruct.id X) …
    -/
    conv_rhs => rw [← q.map_id]
    /-
      case e_toPrefunctor.e_map.h.h.h
      X✝ : Type u
      inst✝ : TopologicalSpace X✝
      x₀ x₁ : X✝
      X : TopCat
      x y : FundamentalGroupoid ↑X
      p : Quiver.Hom x y
      q : Path x.as y.as
      ⊢ Eq (Quotient.mk (Path.Homotopic.setoid ((CategoryTheory.CategoryStruct.id X) …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      X✝ Y✝ Z✝ : TopCat
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => { α := FundamentalGroupoid ↑X, str := inferInstance }, …
    -/
    simp only
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      X✝ Y✝ Z✝ : TopCat
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq { obj := fun x => { as := (CategoryTheory.CategoryStruct.comp f g) x.as } …
    -/
    congr
    /-
      case e_toPrefunctor.e_map
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      X✝ Y✝ Z✝ : TopCat
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq (fun {X Y} p => Path.Homotopic.Quotient.mapFn p (CategoryTheory.CategoryS …
    -/
    ext x y p
    /-
      case e_toPrefunctor.e_map.h.h.h
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      X✝ Y✝ Z✝ : TopCat
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      x y : FundamentalGroupoid ↑X✝
      p : Quiver.Hom x y
      ⊢ Eq (Path.Homotopic.Quotient.mapFn p (CategoryTheory.CategoryStruct.comp f g) …
    -/
    refine Quotient.inductionOn p fun q => ?_
    /-
      case e_toPrefunctor.e_map.h.h.h
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      X✝ Y✝ Z✝ : TopCat
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      x y : FundamentalGroupoid ↑X✝
      p : Quiver.Hom x y
      q : Path x.as y.as
      ⊢ Eq (Path.Homotopic.Quotient.mapFn (Quotient.mk (Path.Homotopic.setoid x.as y …
    -/
    simp only [Quotient.map_mk, Path.map_map, Quotient.eq']
    /-
      case e_toPrefunctor.e_map.h.h.h
      X : Type u
      inst✝ : TopologicalSpace X
      x₀ x₁ : X
      X✝ Y✝ Z✝ : TopCat
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      x y : FundamentalGroupoid ↑X✝
      p : Quiver.Hom x y
      q : Path x.as y.as
      ⊢ Eq (Path.Homotopic.Quotient.mapFn (Quotient.mk (Path.Homotopic.setoid x.as y …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[inherit_doc] scoped notation "π" => FundamentalGroupoid.fundamentalGroupoidFunctor


/-- The fundamental groupoid of a topological space. -/
scoped notation "πₓ" => FundamentalGroupoid.fundamentalGroupoidFunctor.obj


/-- The functor between fundamental groupoids induced by a continuous map. -/
scoped notation "πₘ" => FundamentalGroupoid.fundamentalGroupoidFunctor.map


theorem map_eq {X Y : TopCat} {x₀ x₁ : X} (f : C(X, Y)) (p : Path.Homotopic.Quotient x₀ x₁) :
    (πₘ f).map p = p.mapFn f := rfl


/-- Help the typechecker by converting a point in a groupoid back to a point in
the underlying topological space. -/
abbrev toTop {X : TopCat} (x : πₓ X) : X := x.as


/-- Help the typechecker by converting a point in a topological space to a
point in the fundamental groupoid of that space. -/
abbrev fromTop {X : TopCat} (x : X) : πₓ X := ⟨x⟩


/-- Help the typechecker by converting an arrow in the fundamental groupoid of
a topological space back to a path in that space (i.e., `Path.Homotopic.Quotient`). -/
-- Porting note: Added `(X := X)` to the type.
abbrev toPath {X : TopCat} {x₀ x₁ : πₓ X} (p : x₀ ⟶ x₁) :
    Path.Homotopic.Quotient (X := X) x₀.as x₁.as :=
  p


/-- Help the typechecker by converting a path in a topological space to an arrow in the
fundamental groupoid of that space. -/
abbrev fromPath {X : TopCat} {x₀ x₁ : X} (p : Path.Homotopic.Quotient x₀ x₁) :
    FundamentalGroupoid.mk x₀ ⟶ FundamentalGroupoid.mk x₁ := p


