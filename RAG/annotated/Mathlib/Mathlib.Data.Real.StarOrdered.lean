/-- Although the instance `RCLike.toStarOrderedRing` exists, it is locked behind the
`ComplexOrder` scope because currently the order on `ℂ` is not enabled globally. But we
want `StarOrderedRing ℝ` to be available globally, so we include this instance separately.
In addition, providing this instance here makes it available earlier in the import
hierarchy; otherwise in order to access it we would need to import `Mathlib.Analysis.RCLike.Basic`.
-/
instance Real.instStarOrderedRing : StarOrderedRing ℝ :=
  StarOrderedRing.of_nonneg_iff' add_le_add_left fun r => by
    /-
      r : Real
      ⊢ Iff (LE.le 0 r) (Exists fun s => Eq r (HMul.hMul (Star.star s) s))
    -/
    refine ⟨fun hr => ⟨√r, (mul_self_sqrt hr).symm⟩, ?_⟩
    /-
      r : Real
      ⊢ (Exists fun s => Eq r (HMul.hMul (Star.star s) s)) → LE.le 0 r
    -/
    rintro ⟨s, rfl⟩
    /-
      case intro
      s : Real
      ⊢ LE.le 0 (HMul.hMul (Star.star s) s)
    -/
    exact mul_self_nonneg s
    /-
      🎉 no goals
    -/


instance NNReal.instStarOrderedRing : StarOrderedRing ℝ≥0 := by
  /-
    ⊢ StarOrderedRing NNReal
  -/
  refine .of_le_iff fun x y ↦ ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      x y : NNReal
      h : LE.le x y
      ⊢ Exists fun s => Eq y (HAdd.hAdd x (HMul.hMul (Star.star s) s))
    -/
  · obtain ⟨d, rfl⟩ := exists_add_of_le h
    /-
      case refine_1.intro
      x d : NNReal
      h : LE.le x (HAdd.hAdd x d)
      ⊢ Exists fun s => Eq (HAdd.hAdd x d) (HAdd.hAdd x (HMul.hMul (Star.star s) s))
    -/
    refine ⟨sqrt d, ?_⟩
    /-
      case refine_1.intro
      x d : NNReal
      h : LE.le x (HAdd.hAdd x d)
      ⊢ Eq (HAdd.hAdd x d) (HAdd.hAdd x (HMul.hMul (Star.star (NNReal.sqrt d)) (NNRe …
    -/
    simp only [star_trivial, mul_self_sqrt]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      x y : NNReal
      ⊢ (Exists fun s => Eq y (HAdd.hAdd x (HMul.hMul (Star.star s) s))) → LE.le x y
    -/
  · rintro ⟨p, -, rfl⟩
    /-
      case refine_2.intro.refl
      x p : NNReal
      ⊢ LE.le x (HAdd.hAdd x (HMul.hMul (Star.star p) p))
    -/
    exact le_self_add
    /-
      🎉 no goals
    -/

