theorem Int.natAbs_eq_iff_associated {a b : ℤ} : a.natAbs = b.natAbs ↔ Associated a b := by
  /-
    a b : Int
    ⊢ Iff (Eq a.natAbs b.natAbs) (Associated a b)
  -/
  refine Int.natAbs_eq_natAbs_iff.trans ?_
  /-
    a b : Int
    ⊢ Iff (Or (Eq a b) (Eq a (Neg.neg b))) (Associated a b)
  -/
  constructor
    /-
      case mp
      a b : Int
      ⊢ Or (Eq a b) (Eq a (Neg.neg b)) → Associated a b
    -/
  · rintro (rfl | rfl)
      /-
        case mp.inl
        a : Int
        ⊢ Associated a a
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        b : Int
        ⊢ Associated (Neg.neg b) b
      -/
    · exact ⟨-1, by simp⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      a b : Int
      ⊢ Associated a b → Or (Eq a b) (Eq a (Neg.neg b))
    -/
  · rintro ⟨u, rfl⟩
    /-
      case mpr.intro
      a : Int
      u : Units Int
      ⊢ Or (Eq a (HMul.hMul a ↑u)) (Eq a (Neg.neg (HMul.hMul a ↑u)))
    -/
    obtain rfl | rfl := Int.units_eq_one_or u
      /-
        case mpr.intro.inl
        a : Int
        ⊢ Or (Eq a (HMul.hMul a ↑1)) (Eq a (Neg.neg (HMul.hMul a ↑1)))
      -/
    · exact Or.inl (by simp)
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.inr
        a : Int
        ⊢ Or (Eq a (HMul.hMul a ↑(-1))) (Eq a (Neg.neg (HMul.hMul a ↑(-1))))
      -/
    · exact Or.inr (by simp)
      /-
        🎉 no goals
      -/

