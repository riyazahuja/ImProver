noncomputable instance : PosPart A where
  posPart := cfcₙ (·⁺ : ℝ → ℝ)


noncomputable instance : NegPart A where
  negPart := cfcₙ (·⁻ : ℝ → ℝ)


lemma posPart_def (a : A) : a⁺ = cfcₙ (·⁺ : ℝ → ℝ) a := rfl


lemma negPart_def (a : A) : a⁻ = cfcₙ (·⁻ : ℝ → ℝ) a := rfl


@[simp]
                                        /-
                                          A : Type u_1
                                          inst✝⁶ : NonUnitalRing A
                                          inst✝⁵ : Module Real A
                                          inst✝⁴ : SMulCommClass Real A A
                                          inst✝³ : IsScalarTower Real A A
                                          inst✝² : StarRing A
                                          inst✝¹ : TopologicalSpace A
                                          inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
                                          ⊢ Eq (PosPart.posPart 0) 0
                                        -/
lemma posPart_zero : (0 : A)⁺ = 0 := by simp [posPart_def]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
                                        /-
                                          A : Type u_1
                                          inst✝⁶ : NonUnitalRing A
                                          inst✝⁵ : Module Real A
                                          inst✝⁴ : SMulCommClass Real A A
                                          inst✝³ : IsScalarTower Real A A
                                          inst✝² : StarRing A
                                          inst✝¹ : TopologicalSpace A
                                          inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
                                          ⊢ Eq (NegPart.negPart 0) 0
                                        -/
lemma negPart_zero : (0 : A)⁻ = 0 := by simp [negPart_def]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
lemma posPart_mul_negPart (a : A) : a⁺ * a⁻ = 0 := by
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module Real A
    inst✝⁴ : SMulCommClass Real A A
    inst✝³ : IsScalarTower Real A A
    inst✝² : StarRing A
    inst✝¹ : TopologicalSpace A
    inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ⊢ Eq (HMul.hMul (PosPart.posPart a) (NegPart.negPart a)) 0
  -/
  rw [posPart_def, negPart_def]
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module Real A
    inst✝⁴ : SMulCommClass Real A A
    inst✝³ : IsScalarTower Real A A
    inst✝² : StarRing A
    inst✝¹ : TopologicalSpace A
    inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ⊢ Eq (HMul.hMul (cfcₙ (fun x => PosPart.posPart x) a) (cfcₙ (fun x => NegPart. …
  -/
  by_cases ha : IsSelfAdjoint a
    /-
      case pos
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : SMulCommClass Real A A
      inst✝³ : IsScalarTower Real A A
      inst✝² : StarRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha : IsSelfAdjoint a
      ⊢ Eq (HMul.hMul (cfcₙ (fun x => PosPart.posPart x) a) (cfcₙ (fun x => NegPart. …
    -/
  · rw [← cfcₙ_mul _ _, ← cfcₙ_zero ℝ a]
    /-
      case pos
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : SMulCommClass Real A A
      inst✝³ : IsScalarTower Real A A
      inst✝² : StarRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha : IsSelfAdjoint a
      ⊢ Eq (cfcₙ (fun x => HMul.hMul (PosPart.posPart x) (NegPart.negPart x)) a) (cf …
    -/
    refine cfcₙ_congr (fun x _ ↦ ?_)
    /-
      case pos
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : SMulCommClass Real A A
      inst✝³ : IsScalarTower Real A A
      inst✝² : StarRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha : IsSelfAdjoint a
      x : Real
      x✝ : Membership.mem (quasispectrum Real a) x
      ⊢ Eq (HMul.hMul (PosPart.posPart x) (NegPart.negPart x)) (0 x)
    -/
    simp only [_root_.posPart_def, _root_.negPart_def]
    /-
      case pos
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : SMulCommClass Real A A
      inst✝³ : IsScalarTower Real A A
      inst✝² : StarRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha : IsSelfAdjoint a
      x : Real
      x✝ : Membership.mem (quasispectrum Real a) x
      ⊢ Eq (HMul.hMul (Max.max x 0) (Max.max (Neg.neg x) 0)) (0 x)
    -/
    simpa using le_total x 0
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : SMulCommClass Real A A
      inst✝³ : IsScalarTower Real A A
      inst✝² : StarRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha : Not (IsSelfAdjoint a)
      ⊢ Eq (HMul.hMul (cfcₙ (fun x => PosPart.posPart x) a) (cfcₙ (fun x => NegPart. …
    -/
  · simp [cfcₙ_apply_of_not_predicate a ha]
    /-
      🎉 no goals
    -/


@[simp]
lemma negPart_mul_posPart (a : A) : a⁻ * a⁺ = 0 := by
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module Real A
    inst✝⁴ : SMulCommClass Real A A
    inst✝³ : IsScalarTower Real A A
    inst✝² : StarRing A
    inst✝¹ : TopologicalSpace A
    inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ⊢ Eq (HMul.hMul (NegPart.negPart a) (PosPart.posPart a)) 0
  -/
  rw [posPart_def, negPart_def]
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module Real A
    inst✝⁴ : SMulCommClass Real A A
    inst✝³ : IsScalarTower Real A A
    inst✝² : StarRing A
    inst✝¹ : TopologicalSpace A
    inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ⊢ Eq (HMul.hMul (cfcₙ (fun x => NegPart.negPart x) a) (cfcₙ (fun x => PosPart. …
  -/
  by_cases ha : IsSelfAdjoint a
    /-
      case pos
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : SMulCommClass Real A A
      inst✝³ : IsScalarTower Real A A
      inst✝² : StarRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha : IsSelfAdjoint a
      ⊢ Eq (HMul.hMul (cfcₙ (fun x => NegPart.negPart x) a) (cfcₙ (fun x => PosPart. …
    -/
  · rw [← cfcₙ_mul _ _, ← cfcₙ_zero ℝ a]
    /-
      case pos
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : SMulCommClass Real A A
      inst✝³ : IsScalarTower Real A A
      inst✝² : StarRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha : IsSelfAdjoint a
      ⊢ Eq (cfcₙ (fun x => HMul.hMul (NegPart.negPart x) (PosPart.posPart x)) a) (cf …
    -/
    refine cfcₙ_congr (fun x _ ↦ ?_)
    /-
      case pos
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : SMulCommClass Real A A
      inst✝³ : IsScalarTower Real A A
      inst✝² : StarRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha : IsSelfAdjoint a
      x : Real
      x✝ : Membership.mem (quasispectrum Real a) x
      ⊢ Eq (HMul.hMul (NegPart.negPart x) (PosPart.posPart x)) (0 x)
    -/
    simp only [_root_.posPart_def, _root_.negPart_def]
    /-
      case pos
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : SMulCommClass Real A A
      inst✝³ : IsScalarTower Real A A
      inst✝² : StarRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha : IsSelfAdjoint a
      x : Real
      x✝ : Membership.mem (quasispectrum Real a) x
      ⊢ Eq (HMul.hMul (Max.max (Neg.neg x) 0) (Max.max x 0)) (0 x)
    -/
    simpa using le_total 0 x
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : SMulCommClass Real A A
      inst✝³ : IsScalarTower Real A A
      inst✝² : StarRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha : Not (IsSelfAdjoint a)
      ⊢ Eq (HMul.hMul (cfcₙ (fun x => NegPart.negPart x) a) (cfcₙ (fun x => PosPart. …
    -/
  · simp [cfcₙ_apply_of_not_predicate a ha]
    /-
      🎉 no goals
    -/


lemma posPart_sub_negPart (a : A) (ha : IsSelfAdjoint a := by cfc_tac) : a⁺ - a⁻ = a := by
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module Real A
    inst✝⁴ : SMulCommClass Real A A
    inst✝³ : IsScalarTower Real A A
    inst✝² : StarRing A
    inst✝¹ : TopologicalSpace A
    inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Eq (HSub.hSub (PosPart.posPart a) (NegPart.negPart a)) a
  -/
  rw [posPart_def, negPart_def]
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module Real A
    inst✝⁴ : SMulCommClass Real A A
    inst✝³ : IsScalarTower Real A A
    inst✝² : StarRing A
    inst✝¹ : TopologicalSpace A
    inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Eq (HSub.hSub (cfcₙ (fun x => PosPart.posPart x) a) (cfcₙ (fun x => NegPart. …
  -/
  rw [← cfcₙ_sub _ _]
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module Real A
    inst✝⁴ : SMulCommClass Real A A
    inst✝³ : IsScalarTower Real A A
    inst✝² : StarRing A
    inst✝¹ : TopologicalSpace A
    inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Eq (cfcₙ (fun x => HSub.hSub (PosPart.posPart x) (NegPart.negPart x)) a) a
  -/
  conv_rhs => rw [← cfcₙ_id ℝ a]
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module Real A
    inst✝⁴ : SMulCommClass Real A A
    inst✝³ : IsScalarTower Real A A
    inst✝² : StarRing A
    inst✝¹ : TopologicalSpace A
    inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Eq (cfcₙ (fun x => HSub.hSub (PosPart.posPart x) (NegPart.negPart x)) a) (cf …
  -/
  congr! 2 with
  /-
    case h.e'_17.h
    A : Type u_1
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module Real A
    inst✝⁴ : SMulCommClass Real A A
    inst✝³ : IsScalarTower Real A A
    inst✝² : StarRing A
    inst✝¹ : TopologicalSpace A
    inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    x✝ : Real
    ⊢ Eq (HSub.hSub (PosPart.posPart x✝) (NegPart.negPart x✝)) (id x✝)
  -/
  exact _root_.posPart_sub_negPart _
  /-
    🎉 no goals
  -/


@[simp]
lemma posPart_neg (a : A) : (-a)⁺ = a⁻ := by
  /-
    A : Type u_1
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : Module Real A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : IsScalarTower Real A A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    a : A
    ⊢ Eq (PosPart.posPart (Neg.neg a)) (NegPart.negPart a)
  -/
  by_cases ha : IsSelfAdjoint a
    /-
      case pos
      A : Type u_1
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : Module Real A
      inst✝⁵ : SMulCommClass Real A A
      inst✝⁴ : IsScalarTower Real A A
      inst✝³ : StarRing A
      inst✝² : TopologicalSpace A
      inst✝¹ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      a : A
      ha : IsSelfAdjoint a
      ⊢ Eq (PosPart.posPart (Neg.neg a)) (NegPart.negPart a)
    -/
  · rw [posPart_def, negPart_def, ← cfcₙ_comp_neg _ _]
    /-
      case pos
      A : Type u_1
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : Module Real A
      inst✝⁵ : SMulCommClass Real A A
      inst✝⁴ : IsScalarTower Real A A
      inst✝³ : StarRing A
      inst✝² : TopologicalSpace A
      inst✝¹ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      a : A
      ha : IsSelfAdjoint a
      ⊢ Eq (cfcₙ (fun x => PosPart.posPart (Neg.neg x)) a) (cfcₙ (fun x => NegPart.n …
    -/
    congr! 2
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_1
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : Module Real A
      inst✝⁵ : SMulCommClass Real A A
      inst✝⁴ : IsScalarTower Real A A
      inst✝³ : StarRing A
      inst✝² : TopologicalSpace A
      inst✝¹ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      a : A
      ha : Not (IsSelfAdjoint a)
      ⊢ Eq (PosPart.posPart (Neg.neg a)) (NegPart.negPart a)
    -/
  · have ha' : ¬ IsSelfAdjoint (-a) := fun h ↦ ha (by simpa using h.neg)
    rw [posPart_def, negPart_def, cfcₙ_apply_of_not_predicate a ha,
      cfcₙ_apply_of_not_predicate _ ha']


@[simp]
lemma negPart_neg (a : A) : (-a)⁻ = a⁺ := by
  /-
    A : Type u_1
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : Module Real A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : IsScalarTower Real A A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    a : A
    ⊢ Eq (NegPart.negPart (Neg.neg a)) (PosPart.posPart a)
  -/
  rw [← eq_comm, ← sub_eq_zero, ← posPart_neg, neg_neg, sub_self]
  /-
    🎉 no goals
  -/


@[simp]
lemma posPart_smul {r : ℝ≥0} {a : A} : (r • a)⁺ = r • a⁺ := by
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝ : StarModule Real A
    r : NNReal
    a : A
    ⊢ Eq (PosPart.posPart (HSMul.hSMul r a)) (HSMul.hSMul r (PosPart.posPart a))
  -/
  by_cases ha : IsSelfAdjoint a
    /-
      case pos
      A : Type u_1
      inst✝⁸ : NonUnitalRing A
      inst✝⁷ : Module Real A
      inst✝⁶ : SMulCommClass Real A A
      inst✝⁵ : IsScalarTower Real A A
      inst✝⁴ : StarRing A
      inst✝³ : TopologicalSpace A
      inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      inst✝ : StarModule Real A
      r : NNReal
      a : A
      ha : IsSelfAdjoint a
      ⊢ Eq (PosPart.posPart (HSMul.hSMul r a)) (HSMul.hSMul r (PosPart.posPart a))
    -/
  · simp only [CFC.posPart_def, NNReal.smul_def]
    /-
      case pos
      A : Type u_1
      inst✝⁸ : NonUnitalRing A
      inst✝⁷ : Module Real A
      inst✝⁶ : SMulCommClass Real A A
      inst✝⁵ : IsScalarTower Real A A
      inst✝⁴ : StarRing A
      inst✝³ : TopologicalSpace A
      inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      inst✝ : StarModule Real A
      r : NNReal
      a : A
      ha : IsSelfAdjoint a
      ⊢ Eq (cfcₙ (fun x => PosPart.posPart x) (HSMul.hSMul (↑r) a)) (HSMul.hSMul (↑r …
    -/
    rw [← cfcₙ_comp_smul .., ← cfcₙ_smul ..]
    /-
      case pos
      A : Type u_1
      inst✝⁸ : NonUnitalRing A
      inst✝⁷ : Module Real A
      inst✝⁶ : SMulCommClass Real A A
      inst✝⁵ : IsScalarTower Real A A
      inst✝⁴ : StarRing A
      inst✝³ : TopologicalSpace A
      inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      inst✝ : StarModule Real A
      r : NNReal
      a : A
      ha : IsSelfAdjoint a
      ⊢ Eq (cfcₙ (fun x => PosPart.posPart (HSMul.hSMul (↑r) x)) a) (cfcₙ (fun x =>  …
    -/
    refine cfcₙ_congr fun x hx ↦ ?_
    /-
      case pos
      A : Type u_1
      inst✝⁸ : NonUnitalRing A
      inst✝⁷ : Module Real A
      inst✝⁶ : SMulCommClass Real A A
      inst✝⁵ : IsScalarTower Real A A
      inst✝⁴ : StarRing A
      inst✝³ : TopologicalSpace A
      inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      inst✝ : StarModule Real A
      r : NNReal
      a : A
      ha : IsSelfAdjoint a
      x : Real
      hx : Membership.mem (quasispectrum Real a) x
      ⊢ Eq (PosPart.posPart (HSMul.hSMul (↑r) x)) (HSMul.hSMul (↑r) (PosPart.posPart …
    -/
    simp [_root_.posPart_def, mul_max_of_nonneg]
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_1
      inst✝⁸ : NonUnitalRing A
      inst✝⁷ : Module Real A
      inst✝⁶ : SMulCommClass Real A A
      inst✝⁵ : IsScalarTower Real A A
      inst✝⁴ : StarRing A
      inst✝³ : TopologicalSpace A
      inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      inst✝ : StarModule Real A
      r : NNReal
      a : A
      ha : Not (IsSelfAdjoint a)
      ⊢ Eq (PosPart.posPart (HSMul.hSMul r a)) (HSMul.hSMul r (PosPart.posPart a))
    -/
  · obtain (rfl | hr) := eq_or_ne r 0
      /-
        case neg.inl
        A : Type u_1
        inst✝⁸ : NonUnitalRing A
        inst✝⁷ : Module Real A
        inst✝⁶ : SMulCommClass Real A A
        inst✝⁵ : IsScalarTower Real A A
        inst✝⁴ : StarRing A
        inst✝³ : TopologicalSpace A
        inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
        inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
        inst✝ : StarModule Real A
        a : A
        ha : Not (IsSelfAdjoint a)
        ⊢ Eq (PosPart.posPart (HSMul.hSMul 0 a)) (HSMul.hSMul 0 (PosPart.posPart a))
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        A : Type u_1
        inst✝⁸ : NonUnitalRing A
        inst✝⁷ : Module Real A
        inst✝⁶ : SMulCommClass Real A A
        inst✝⁵ : IsScalarTower Real A A
        inst✝⁴ : StarRing A
        inst✝³ : TopologicalSpace A
        inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
        inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
        inst✝ : StarModule Real A
        r : NNReal
        a : A
        ha : Not (IsSelfAdjoint a)
        hr : Ne r 0
        ⊢ Eq (PosPart.posPart (HSMul.hSMul r a)) (HSMul.hSMul r (PosPart.posPart a))
      -/
    · have := (not_iff_not.mpr <| (IsSelfAdjoint.all r).smul_iff hr.isUnit (x := a)) |>.mpr ha
      simp [CFC.posPart_def, cfcₙ_apply_of_not_predicate a ha,
        cfcₙ_apply_of_not_predicate _ this]


@[simp]
lemma negPart_smul {r : ℝ≥0} {a : A} : (r • a)⁻ = r • a⁻ := by
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝ : StarModule Real A
    r : NNReal
    a : A
    ⊢ Eq (NegPart.negPart (HSMul.hSMul r a)) (HSMul.hSMul r (NegPart.negPart a))
  -/
  simpa using posPart_smul (r := r) (a := -a)
  /-
    🎉 no goals
  -/


lemma posPart_smul_of_nonneg {r : ℝ} (hr : 0 ≤ r) {a : A} : (r • a)⁺ = r • a⁺ :=
  posPart_smul (r := ⟨r, hr⟩)


lemma posPart_smul_of_nonpos {r : ℝ} (hr : r ≤ 0) {a : A} : (r • a)⁺ = -r • a⁻ := by
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝ : StarModule Real A
    r : Real
    hr : LE.le r 0
    a : A
    ⊢ Eq (PosPart.posPart (HSMul.hSMul r a)) (HSMul.hSMul (Neg.neg r) (NegPart.neg …
  -/
  nth_rw 1 [← neg_neg r]
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝ : StarModule Real A
    r : Real
    hr : LE.le r 0
    a : A
    ⊢ Eq (PosPart.posPart (HSMul.hSMul (Neg.neg (Neg.neg r)) a)) (HSMul.hSMul (Neg …
  -/
  rw [neg_smul, ← smul_neg, posPart_smul_of_nonneg (neg_nonneg.mpr hr), posPart_neg]
  /-
    🎉 no goals
  -/


lemma negPart_smul_of_nonneg {r : ℝ} (hr : 0 ≤ r) {a : A} : (r • a)⁻ = r • a⁻ := by
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝ : StarModule Real A
    r : Real
    hr : LE.le 0 r
    a : A
    ⊢ Eq (NegPart.negPart (HSMul.hSMul r a)) (HSMul.hSMul r (NegPart.negPart a))
  -/
  conv_lhs => rw [← neg_neg r, neg_smul, negPart_neg, posPart_smul_of_nonpos (by simpa), neg_neg]
  /-
    🎉 no goals
  -/


lemma negPart_smul_of_nonpos {r : ℝ} (hr : r ≤ 0) {a : A} : (r • a)⁻ = -r • a⁺ := by
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝ : StarModule Real A
    r : Real
    hr : LE.le r 0
    a : A
    ⊢ Eq (NegPart.negPart (HSMul.hSMul r a)) (HSMul.hSMul (Neg.neg r) (PosPart.pos …
  -/
  conv_lhs => rw [← neg_neg r, neg_smul, negPart_neg, posPart_smul_of_nonneg (by simpa)]
  /-
    🎉 no goals
  -/


@[aesop norm apply (rule_sets := [CStarAlgebra])]
lemma posPart_nonneg (a : A) :
    0 ≤ a⁺ :=
                            /-
                              A : Type u_1
                              inst✝⁸ : NonUnitalRing A
                              inst✝⁷ : Module Real A
                              inst✝⁶ : SMulCommClass Real A A
                              inst✝⁵ : IsScalarTower Real A A
                              inst✝⁴ : StarRing A
                              inst✝³ : TopologicalSpace A
                              inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
                              inst✝¹ : PartialOrder A
                              inst✝ : StarOrderedRing A
                              a : A
                              x : Real
                              x✝ : Membership.mem (quasispectrum Real a) x
                              ⊢ LE.le 0 (PosPart.posPart x)
                            -/
  cfcₙ_nonneg (fun x _ ↦ by positivity)
                            /-
                              🎉 no goals
                            -/


@[aesop norm apply (rule_sets := [CStarAlgebra])]
lemma negPart_nonneg (a : A) :
    0 ≤ a⁻ :=
                            /-
                              A : Type u_1
                              inst✝⁸ : NonUnitalRing A
                              inst✝⁷ : Module Real A
                              inst✝⁶ : SMulCommClass Real A A
                              inst✝⁵ : IsScalarTower Real A A
                              inst✝⁴ : StarRing A
                              inst✝³ : TopologicalSpace A
                              inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
                              inst✝¹ : PartialOrder A
                              inst✝ : StarOrderedRing A
                              a : A
                              x : Real
                              x✝ : Membership.mem (quasispectrum Real a) x
                              ⊢ LE.le 0 (NegPart.negPart x)
                            -/
  cfcₙ_nonneg (fun x _ ↦ by positivity)
                            /-
                              🎉 no goals
                            -/


lemma posPart_eq_of_eq_sub_negPart {a b : A} (hab : a = b - a⁻) (hb : 0 ≤ b := by cfc_tac) :
    a⁺ = b := by
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a b : A
    hab : Eq a (HSub.hSub b (NegPart.negPart a))
    hb : autoParam (LE.le 0 b) _auto✝
    ⊢ Eq (PosPart.posPart a) b
  -/
  have ha := hab.symm ▸ hb.isSelfAdjoint.sub (negPart_nonneg a).isSelfAdjoint
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a b : A
    hab : Eq a (HSub.hSub b (NegPart.negPart a))
    hb : autoParam (LE.le 0 b) _auto✝
    ha : IsSelfAdjoint a
    ⊢ Eq (PosPart.posPart a) b
  -/
  nth_rw 1 [← posPart_sub_negPart a] at hab
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a b : A
    hab : Eq (HSub.hSub (PosPart.posPart a) (NegPart.negPart a)) (HSub.hSub b (Neg …
    hb : autoParam (LE.le 0 b) _auto✝
    ha : IsSelfAdjoint a
    ⊢ Eq (PosPart.posPart a) b
  -/
  simpa using hab
  /-
    🎉 no goals
  -/


lemma negPart_eq_of_eq_PosPart_sub {a c : A} (hac : a = a⁺ - c) (hc : 0 ≤ c := by cfc_tac) :
    a⁻ = c := by
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a c : A
    hac : Eq a (HSub.hSub (PosPart.posPart a) c)
    hc : autoParam (LE.le 0 c) _auto✝
    ⊢ Eq (NegPart.negPart a) c
  -/
  have ha := hac.symm ▸ (posPart_nonneg a).isSelfAdjoint.sub hc.isSelfAdjoint
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a c : A
    hac : Eq a (HSub.hSub (PosPart.posPart a) c)
    hc : autoParam (LE.le 0 c) _auto✝
    ha : IsSelfAdjoint a
    ⊢ Eq (NegPart.negPart a) c
  -/
  nth_rw 1 [← posPart_sub_negPart a] at hac
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a c : A
    hac : Eq (HSub.hSub (PosPart.posPart a) (NegPart.negPart a)) (HSub.hSub (PosPa …
    hc : autoParam (LE.le 0 c) _auto✝
    ha : IsSelfAdjoint a
    ⊢ Eq (NegPart.negPart a) c
  -/
  simpa using hac
  /-
    🎉 no goals
  -/


lemma le_posPart {a : A} (ha : IsSelfAdjoint a := by cfc_tac) : a ≤ a⁺ := by
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : SMulCommClass Real A A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ LE.le a (PosPart.posPart a)
  -/
  simpa [posPart_sub_negPart a] using sub_le_self a⁺ (negPart_nonneg a)
  /-
    🎉 no goals
  -/


lemma neg_negPart_le {a : A} (ha : IsSelfAdjoint a := by cfc_tac) : -a⁻ ≤ a := by
  simpa only [posPart_sub_negPart a, ← sub_eq_add_neg]
    using le_add_of_nonneg_left (a := -a⁻) (posPart_nonneg a)


lemma posPart_eq_self (a : A) : a⁺ = a ↔ 0 ≤ a := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ⊢ Iff (Eq (PosPart.posPart a) a) (LE.le 0 a)
  -/
  refine ⟨fun ha ↦ ha ▸ posPart_nonneg a, fun ha ↦ ?_⟩
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le 0 a
    ⊢ Eq (PosPart.posPart a) a
  -/
  conv_rhs => rw [← cfcₙ_id ℝ a]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le 0 a
    ⊢ Eq (PosPart.posPart a) (cfcₙ id a)
  -/
  rw [posPart_def]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le 0 a
    ⊢ Eq (cfcₙ (fun x => PosPart.posPart x) a) (cfcₙ id a)
  -/
  refine cfcₙ_congr (fun x hx ↦ ?_)
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le 0 a
    x : Real
    hx : Membership.mem (quasispectrum Real a) x
    ⊢ Eq (PosPart.posPart x) (id x)
  -/
  simpa [_root_.posPart_def] using quasispectrum_nonneg_of_nonneg a ha x hx
  /-
    🎉 no goals
  -/


@[deprecated posPart_eq_self (since := "2024-11-18")]
lemma eq_posPart_iff (a : A) : a = a⁺ ↔ 0 ≤ a := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ⊢ Iff (Eq a (PosPart.posPart a)) (LE.le 0 a)
  -/
  rw [eq_comm, posPart_eq_self]
  /-
    🎉 no goals
  -/


lemma negPart_eq_zero_iff (a : A) (ha : IsSelfAdjoint a := by cfc_tac) :
    a⁻ = 0 ↔ 0 ≤ a := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Iff (Eq (NegPart.negPart a) 0) (LE.le 0 a)
  -/
  rw [← posPart_eq_self, eq_comm (b := a)]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Iff (Eq (NegPart.negPart a) 0) (Eq a (PosPart.posPart a))
  -/
  nth_rw 2 [← posPart_sub_negPart a]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Iff (Eq (NegPart.negPart a) 0) (Eq (HSub.hSub (PosPart.posPart a) (NegPart.n …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma negPart_eq_neg (a : A) : a⁻ = -a ↔ a ≤ 0 := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ⊢ Iff (Eq (NegPart.negPart a) (Neg.neg a)) (LE.le a 0)
  -/
  rw [← neg_inj, neg_neg, eq_comm]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ⊢ Iff (Eq a (Neg.neg (NegPart.negPart a))) (LE.le a 0)
  -/
  refine ⟨fun ha ↦ by rw [ha, neg_nonpos]; exact negPart_nonneg a, fun ha ↦ ?_⟩
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le a 0
    ⊢ Eq a (Neg.neg (NegPart.negPart a))
  -/
  rw [← neg_nonneg] at ha
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le 0 (Neg.neg a)
    ⊢ Eq a (Neg.neg (NegPart.negPart a))
  -/
  rw [negPart_def, ← cfcₙ_neg]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le 0 (Neg.neg a)
    ⊢ Eq a (cfcₙ (fun x => Neg.neg (NegPart.negPart x)) a)
  -/
  have _ : IsSelfAdjoint a := neg_neg a ▸ (IsSelfAdjoint.neg <| .of_nonneg ha)
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le 0 (Neg.neg a)
    x✝ : IsSelfAdjoint a
    ⊢ Eq a (cfcₙ (fun x => Neg.neg (NegPart.negPart x)) a)
  -/
  conv_lhs => rw [← cfcₙ_id ℝ a]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le 0 (Neg.neg a)
    x✝ : IsSelfAdjoint a
    ⊢ Eq (cfcₙ id a) (cfcₙ (fun x => Neg.neg (NegPart.negPart x)) a)
  -/
  refine cfcₙ_congr fun x hx ↦ ?_
  rw [Unitization.quasispectrum_eq_spectrum_inr ℝ, ← neg_neg x, ← Set.mem_neg,
    spectrum.neg_eq, ← Unitization.inr_neg, ← Unitization.quasispectrum_eq_spectrum_inr ℝ] at hx
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le 0 (Neg.neg a)
    x✝ : IsSelfAdjoint a
    x : Real
    hx : Membership.mem (quasispectrum Real (Neg.neg a)) (Neg.neg x)
    ⊢ Eq (id x) (Neg.neg (NegPart.negPart x))
  -/
  rw [← neg_eq_iff_eq_neg, eq_comm]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le 0 (Neg.neg a)
    x✝ : IsSelfAdjoint a
    x : Real
    hx : Membership.mem (quasispectrum Real (Neg.neg a)) (Neg.neg x)
    ⊢ Eq (NegPart.negPart x) (Neg.neg (id x))
  -/
  simpa using quasispectrum_nonneg_of_nonneg _ ha _ hx
  /-
    🎉 no goals
  -/


@[deprecated negPart_eq_neg (since := "2024-11-18")]
lemma eq_negPart_iff (a : A) : a = -a⁻ ↔ a ≤ 0 := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ⊢ Iff (Eq a (Neg.neg (NegPart.negPart a))) (LE.le a 0)
  -/
  rw [← neg_inj, neg_neg, eq_comm, negPart_eq_neg]
  /-
    🎉 no goals
  -/


lemma posPart_eq_zero_iff (a : A) (ha : IsSelfAdjoint a := by cfc_tac) :
    a⁺ = 0 ↔ a ≤ 0 := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Iff (Eq (PosPart.posPart a) 0) (LE.le a 0)
  -/
  rw [← negPart_eq_neg, eq_comm (b := -a)]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Iff (Eq (PosPart.posPart a) 0) (Eq (Neg.neg a) (NegPart.negPart a))
  -/
  nth_rw 2 [← posPart_sub_negPart a]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Real A
    inst✝⁷ : SMulCommClass Real A A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Iff (Eq (PosPart.posPart a) 0) (Eq (Neg.neg (HSub.hSub (PosPart.posPart a) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


local notation "σₙ" => quasispectrum


open NonUnitalContinuousFunctionalCalculus in
/-- The positive and negative parts of a selfadjoint element `a` are unique. That is, if
`a = b - c` is the difference of nonnegative elements whose product is zero, then these are
precisely `a⁺` and `a⁻`. -/
lemma posPart_negPart_unique {a b c : A} (habc : a = b - c) (hbc : b * c = 0)
    (hb : 0 ≤ b := by cfc_tac) (hc : 0 ≤ c := by cfc_tac) :
    a⁺ = b ∧ a⁻ = c := by
  /- The key idea is to show that `cfcₙ f a = cfcₙ f b + cfcₙ f (-c)` for all real-valued `f`
  continuous on the union of the spectra of `a`, `b`, and `-c`. Then apply this to `f = (·⁺)`.
  The equality holds because both sides constitute star homomorphisms which agree on `f = id` since
  `a = b - c`. -/
  /- `a`, `b`, `-c` are selfadjoint. -/
  /-
    A : Type u_1
    inst✝¹² : NonUnitalRing A
    inst✝¹¹ : Module Real A
    inst✝¹⁰ : SMulCommClass Real A A
    inst✝⁹ : IsScalarTower Real A A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : NonnegSpectrumClass Real A
    inst✝² : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝¹ : TopologicalRing A
    inst✝ : T2Space A
    a b c : A
    habc : Eq a (HSub.hSub b c)
    hbc : Eq (HMul.hMul b c) 0
    hb : autoParam (LE.le 0 b) _auto✝
    hc : autoParam (LE.le 0 c) _auto✝
    ⊢ And (Eq (PosPart.posPart a) b) (Eq (NegPart.negPart a) c)
  -/
  have hb' : IsSelfAdjoint b := .of_nonneg hb
  /-
    A : Type u_1
    inst✝¹² : NonUnitalRing A
    inst✝¹¹ : Module Real A
    inst✝¹⁰ : SMulCommClass Real A A
    inst✝⁹ : IsScalarTower Real A A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : NonnegSpectrumClass Real A
    inst✝² : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝¹ : TopologicalRing A
    inst✝ : T2Space A
    a b c : A
    habc : Eq a (HSub.hSub b c)
    hbc : Eq (HMul.hMul b c) 0
    hb : autoParam (LE.le 0 b) _auto✝
    hc : autoParam (LE.le 0 c) _auto✝
    hb' : IsSelfAdjoint b
    ⊢ And (Eq (PosPart.posPart a) b) (Eq (NegPart.negPart a) c)
  -/
  have hc' : IsSelfAdjoint (-c) := .neg <| .of_nonneg hc
  /-
    A : Type u_1
    inst✝¹² : NonUnitalRing A
    inst✝¹¹ : Module Real A
    inst✝¹⁰ : SMulCommClass Real A A
    inst✝⁹ : IsScalarTower Real A A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : NonnegSpectrumClass Real A
    inst✝² : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝¹ : TopologicalRing A
    inst✝ : T2Space A
    a b c : A
    habc : Eq a (HSub.hSub b c)
    hbc : Eq (HMul.hMul b c) 0
    hb : autoParam (LE.le 0 b) _auto✝
    hc : autoParam (LE.le 0 c) _auto✝
    hb' : IsSelfAdjoint b
    hc' : IsSelfAdjoint (Neg.neg c)
    ⊢ And (Eq (PosPart.posPart a) b) (Eq (NegPart.negPart a) c)
  -/
  have ha : IsSelfAdjoint a := habc ▸ hb'.sub <| .of_nonneg hc
  /- It suffices to show `b = a⁺` since `a⁺ - a⁻ = a = b - c` -/
  /-
    A : Type u_1
    inst✝¹² : NonUnitalRing A
    inst✝¹¹ : Module Real A
    inst✝¹⁰ : SMulCommClass Real A A
    inst✝⁹ : IsScalarTower Real A A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : NonnegSpectrumClass Real A
    inst✝² : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝¹ : TopologicalRing A
    inst✝ : T2Space A
    a b c : A
    habc : Eq a (HSub.hSub b c)
    hbc : Eq (HMul.hMul b c) 0
    hb : autoParam (LE.le 0 b) _auto✝
    hc : autoParam (LE.le 0 c) _auto✝
    hb' : IsSelfAdjoint b
    hc' : IsSelfAdjoint (Neg.neg c)
    ha : IsSelfAdjoint a
    ⊢ And (Eq (PosPart.posPart a) b) (Eq (NegPart.negPart a) c)
  -/
  rw [and_iff_left_of_imp ?of_b_eq]
  case of_b_eq =>
    rintro rfl
    exact negPart_eq_of_eq_PosPart_sub habc hc
  /- `s := σₙ ℝ a ∪ σₙ ℝ b ∪ σₙ ℝ (-c)` is compact and each of these sets are subsets of `s`.
  Moreover, `0 ∈ s`. -/
  /-
    A : Type u_1
    inst✝¹² : NonUnitalRing A
    inst✝¹¹ : Module Real A
    inst✝¹⁰ : SMulCommClass Real A A
    inst✝⁹ : IsScalarTower Real A A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : NonnegSpectrumClass Real A
    inst✝² : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝¹ : TopologicalRing A
    inst✝ : T2Space A
    a b c : A
    habc : Eq a (HSub.hSub b c)
    hbc : Eq (HMul.hMul b c) 0
    hb : autoParam (LE.le 0 b) _auto✝
    hc : autoParam (LE.le 0 c) _auto✝
    hb' : IsSelfAdjoint b
    hc' : IsSelfAdjoint (Neg.neg c)
    ha : IsSelfAdjoint a
    ⊢ Eq (PosPart.posPart a) b
  -/
  let s := σₙ ℝ a ∪ σₙ ℝ b ∪ σₙ ℝ (-c)
  have hs : CompactSpace s := by
    refine isCompact_iff_compactSpace.mp <| (IsCompact.union ?_ ?_).union ?_
    all_goals exact isCompact_quasispectrum _
  obtain ⟨has, hbs, hcs⟩ : σₙ ℝ a ⊆ s ∧ σₙ ℝ b ⊆ s ∧ σₙ ℝ (-c) ⊆ s := by
    refine ⟨?_, ?_, ?_⟩; all_goals intro; aesop
  /-
    case intro.intro
    A : Type u_1
    inst✝¹² : NonUnitalRing A
    inst✝¹¹ : Module Real A
    inst✝¹⁰ : SMulCommClass Real A A
    inst✝⁹ : IsScalarTower Real A A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : NonnegSpectrumClass Real A
    inst✝² : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝¹ : TopologicalRing A
    inst✝ : T2Space A
    a b c : A
    habc : Eq a (HSub.hSub b c)
    hbc : Eq (HMul.hMul b c) 0
    hb : autoParam (LE.le 0 b) _auto✝
    hc : autoParam (LE.le 0 c) _auto✝
    hb' : IsSelfAdjoint b
    hc' : IsSelfAdjoint (Neg.neg c)
    ha : IsSelfAdjoint a
    s : Set Real := Union.union (Union.union (quasispectrum Real a) (quasispectrum …
    hs : CompactSpace ↑s
    has : HasSubset.Subset (quasispectrum Real a) s
    hbs : HasSubset.Subset (quasispectrum Real b) s
    hcs : HasSubset.Subset (quasispectrum Real (Neg.neg c)) s
    ⊢ Eq (PosPart.posPart a) b
  -/
  let zero : Zero s := ⟨0, by aesop⟩
  /-
    case intro.intro
    A : Type u_1
    inst✝¹² : NonUnitalRing A
    inst✝¹¹ : Module Real A
    inst✝¹⁰ : SMulCommClass Real A A
    inst✝⁹ : IsScalarTower Real A A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : NonnegSpectrumClass Real A
    inst✝² : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝¹ : TopologicalRing A
    inst✝ : T2Space A
    a b c : A
    habc : Eq a (HSub.hSub b c)
    hbc : Eq (HMul.hMul b c) 0
    hb : autoParam (LE.le 0 b) _auto✝
    hc : autoParam (LE.le 0 c) _auto✝
    hb' : IsSelfAdjoint b
    hc' : IsSelfAdjoint (Neg.neg c)
    ha : IsSelfAdjoint a
    s : Set Real := Union.union (Union.union (quasispectrum Real a) (quasispectrum …
    hs : CompactSpace ↑s
    has : HasSubset.Subset (quasispectrum Real a) s
    hbs : HasSubset.Subset (quasispectrum Real b) s
    hcs : HasSubset.Subset (quasispectrum Real (Neg.neg c)) s
    zero : Zero ↑s := { zero := ⟨0, ⋯⟩ }
    ⊢ Eq (PosPart.posPart a) b
  -/
  have s0 : (0 : s) = (0 : ℝ) := rfl
  /- The continuous functional calculi for functions `f g : C(s, ℝ)₀` applied to `b` and `(-c)`
  are orthogonal (i.e., the product is always zero). -/
  have mul₁ (f g : C(s, ℝ)₀) :
      (cfcₙHomSuperset hb' hbs f) * (cfcₙHomSuperset hc' hcs g) = 0 := by
    refine f.nonUnitalStarAlgHom_apply_mul_eq_zero s0 _ _ ?id ?star_id
      (cfcₙHomSuperset_continuous hb' hbs)
    case' star_id => rw [star_trivial]
    all_goals
      refine g.mul_nonUnitalStarAlgHom_apply_eq_zero s0 _ _ ?_ ?_
        (cfcₙHomSuperset_continuous hc' hcs)
      all_goals simp only [zero, star_trivial, cfcₙHomSuperset_id' hb' hbs,
        cfcₙHomSuperset_id' hc' hcs, mul_neg, hbc, neg_zero]
  have mul₂ (f g : C(s, ℝ)₀) : (cfcₙHomSuperset hc' hcs f) * (cfcₙHomSuperset hb' hbs g) = 0 := by
    simpa only [star_mul, star_zero, ← map_star, star_trivial] using congr(star $(mul₁ g f))
  /- `fun f ↦ cfcₙ f b + cfcₙ f (-c)` defines a star homomorphism `ψ : C(s, ℝ)₀ →⋆ₙₐ[ℝ] A` which
  agrees with the star homomorphism `cfcₙ · a : C(s, ℝ)₀ →⋆ₙₐ[ℝ] A` since
  `cfcₙ id a = a = b - c = cfcₙ id b + cfcₙ id (-c)`. -/
  let ψ : C(s, ℝ)₀ →⋆ₙₐ[ℝ] A :=
    { (cfcₙHomSuperset hb' hbs : C(s, ℝ)₀ →ₗ[ℝ] A) + (cfcₙHomSuperset hc' hcs : C(s, ℝ)₀ →ₗ[ℝ] A)
        with
      toFun := cfcₙHomSuperset hb' hbs + cfcₙHomSuperset hc' hcs
      map_zero' := by simp [-cfcₙHomSuperset_apply]
      map_mul' := fun f g ↦ by
        simp only [zero, Pi.add_apply, map_mul, mul_add, add_mul, mul₂, add_zero, mul₁,
          zero_add]
      map_star' := fun f ↦ by simp [← map_star] }
  have key : (cfcₙHomSuperset ha has) = ψ :=
    UniqueNonUnitalContinuousFunctionalCalculus.eq_of_continuous_of_map_id s rfl
    (cfcₙHomSuperset ha has) ψ (cfcₙHomSuperset_continuous ha has)
    ((cfcₙHomSuperset_continuous hb' hbs).add (cfcₙHomSuperset_continuous hc' hcs))
    (by simpa [zero, ψ, -cfcₙHomSuperset_apply, cfcₙHomSuperset_id, sub_eq_add_neg] using habc)
  /- Applying the equality of star homomorphisms to the function `(·⁺ : ℝ → ℝ)` we find that
  `b = cfcₙ id b + cfcₙ 0 (-c) = cfcₙ (·⁺) b - cfcₙ (·⁺) (-c) = cfcₙ (·⁺) a = a⁺`, where the
  second equality follows because these functions are equal on the spectra of `b` and `-c`,
  respectively, since `0 ≤ b` and `-c ≤ 0`. -/
  /-
    case intro.intro
    A : Type u_1
    inst✝¹² : NonUnitalRing A
    inst✝¹¹ : Module Real A
    inst✝¹⁰ : SMulCommClass Real A A
    inst✝⁹ : IsScalarTower Real A A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : NonnegSpectrumClass Real A
    inst✝² : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝¹ : TopologicalRing A
    inst✝ : T2Space A
    a b c : A
    habc : Eq a (HSub.hSub b c)
    hbc : Eq (HMul.hMul b c) 0
    hb : autoParam (LE.le 0 b) _auto✝
    hc : autoParam (LE.le 0 c) _auto✝
    hb' : IsSelfAdjoint b
    hc' : IsSelfAdjoint (Neg.neg c)
    ha : IsSelfAdjoint a
    s : Set Real := Union.union (Union.union (quasispectrum Real a) (quasispectrum …
    hs : CompactSpace ↑s
    has : HasSubset.Subset (quasispectrum Real a) s
    hbs : HasSubset.Subset (quasispectrum Real b) s
    hcs : HasSubset.Subset (quasispectrum Real (Neg.neg c)) s
    zero : Zero ↑s := { zero := ⟨0, ⋯⟩ }
    s0 : Eq (↑0) 0
    mul₁ : ∀ (f g : ContinuousMapZero (↑s) Real), Eq (HMul.hMul ((cfcₙHomSuperset  …
    mul₂ : ∀ (f g : ContinuousMapZero (↑s) Real), Eq (HMul.hMul ((cfcₙHomSuperset  …
    ψ : NonUnitalStarAlgHom Real (ContinuousMapZero (↑s) Real) A :=
      let __src := HAdd.hAdd ↑(cfcₙHomSuperset hb' hbs) ↑(cfcₙHomSuperset hc' hcs);
      { toFun := HAdd.hAdd ⇑(cfcₙHomSuperset hb' hbs) ⇑(cfcₙHomSuperset hc' hcs),  …
    key : Eq (cfcₙHomSuperset ha has) ψ
    ⊢ Eq (PosPart.posPart a) b
  -/
  let f : C(s, ℝ)₀ := ⟨⟨(·⁺), by fun_prop⟩, by simp [s0]⟩
  /-
    case intro.intro
    A : Type u_1
    inst✝¹² : NonUnitalRing A
    inst✝¹¹ : Module Real A
    inst✝¹⁰ : SMulCommClass Real A A
    inst✝⁹ : IsScalarTower Real A A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : NonnegSpectrumClass Real A
    inst✝² : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝¹ : TopologicalRing A
    inst✝ : T2Space A
    a b c : A
    habc : Eq a (HSub.hSub b c)
    hbc : Eq (HMul.hMul b c) 0
    hb : autoParam (LE.le 0 b) _auto✝
    hc : autoParam (LE.le 0 c) _auto✝
    hb' : IsSelfAdjoint b
    hc' : IsSelfAdjoint (Neg.neg c)
    ha : IsSelfAdjoint a
    s : Set Real := Union.union (Union.union (quasispectrum Real a) (quasispectrum …
    hs : CompactSpace ↑s
    has : HasSubset.Subset (quasispectrum Real a) s
    hbs : HasSubset.Subset (quasispectrum Real b) s
    hcs : HasSubset.Subset (quasispectrum Real (Neg.neg c)) s
    zero : Zero ↑s := { zero := ⟨0, ⋯⟩ }
    s0 : Eq (↑0) 0
    mul₁ : ∀ (f g : ContinuousMapZero (↑s) Real), Eq (HMul.hMul ((cfcₙHomSuperset  …
    mul₂ : ∀ (f g : ContinuousMapZero (↑s) Real), Eq (HMul.hMul ((cfcₙHomSuperset  …
    ψ : NonUnitalStarAlgHom Real (ContinuousMapZero (↑s) Real) A :=
      let __src := HAdd.hAdd ↑(cfcₙHomSuperset hb' hbs) ↑(cfcₙHomSuperset hc' hcs);
      { toFun := HAdd.hAdd ⇑(cfcₙHomSuperset hb' hbs) ⇑(cfcₙHomSuperset hc' hcs),  …
    key : Eq (cfcₙHomSuperset ha has) ψ
    f : ContinuousMapZero (↑s) Real := { toFun := fun x => PosPart.posPart ↑x, con …
    ⊢ Eq (PosPart.posPart a) b
  -/
  replace key := congr($key f)
  simp only [cfcₙHomSuperset_apply, NonUnitalStarAlgHom.coe_mk', NonUnitalAlgHom.coe_mk, ψ,
    Pi.add_apply, cfcₙHom_eq_cfcₙ_extend (·⁺)] at key
  /-
    case intro.intro
    A : Type u_1
    inst✝¹² : NonUnitalRing A
    inst✝¹¹ : Module Real A
    inst✝¹⁰ : SMulCommClass Real A A
    inst✝⁹ : IsScalarTower Real A A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : NonnegSpectrumClass Real A
    inst✝² : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝¹ : TopologicalRing A
    inst✝ : T2Space A
    a b c : A
    habc : Eq a (HSub.hSub b c)
    hbc : Eq (HMul.hMul b c) 0
    hb : autoParam (LE.le 0 b) _auto✝
    hc : autoParam (LE.le 0 c) _auto✝
    hb' : IsSelfAdjoint b
    hc' : IsSelfAdjoint (Neg.neg c)
    ha : IsSelfAdjoint a
    s : Set Real := Union.union (Union.union (quasispectrum Real a) (quasispectrum …
    hs : CompactSpace ↑s
    has : HasSubset.Subset (quasispectrum Real a) s
    hbs : HasSubset.Subset (quasispectrum Real b) s
    hcs : HasSubset.Subset (quasispectrum Real (Neg.neg c)) s
    zero : Zero ↑s := { zero := ⟨0, ⋯⟩ }
    s0 : Eq (↑0) 0
    mul₁ : ∀ (f g : ContinuousMapZero (↑s) Real), Eq (HMul.hMul ((cfcₙHomSuperset  …
    mul₂ : ∀ (f g : ContinuousMapZero (↑s) Real), Eq (HMul.hMul ((cfcₙHomSuperset  …
    ψ : NonUnitalStarAlgHom Real (ContinuousMapZero (↑s) Real) A :=
      let __src := HAdd.hAdd ↑(cfcₙHomSuperset hb' hbs) ↑(cfcₙHomSuperset hc' hcs);
      { toFun := HAdd.hAdd ⇑(cfcₙHomSuperset hb' hbs) ⇑(cfcₙHomSuperset hc' hcs),  …
    f : ContinuousMapZero (↑s) Real := { toFun := fun x => PosPart.posPart ↑x, con …
    key : Eq (cfcₙ (Function.extend Subtype.val ⇑(f.comp { toFun := Subtype.map id …
    ⊢ Eq (PosPart.posPart a) b
  -/
  symm
  calc
    b = cfcₙ (id : ℝ → ℝ) b + cfcₙ (0 : ℝ → ℝ) (-c) := by simp [cfcₙ_id ℝ b]
    _ = _ := by
      congr! 1
      all_goals
        refine cfcₙ_congr fun x hx ↦ Eq.symm ?_
        lift x to σₙ ℝ _ using hx
        simp only [zero, Subtype.val_injective.extend_apply, comp_apply, coe_mk,
          ContinuousMap.coe_mk, Subtype.map_coe, id_eq, _root_.posPart_eq_self, f, Pi.zero_apply,
          posPart_eq_zero]
      · exact quasispectrum_nonneg_of_nonneg b hb x.val x.property
      · obtain ⟨x, hx⟩ := x
        simp only [← neg_nonneg]
        rw [Unitization.quasispectrum_eq_spectrum_inr ℝ (-c), Unitization.inr_neg,
          ← spectrum.neg_eq, Set.mem_neg, ← Unitization.quasispectrum_eq_spectrum_inr ℝ c]
          at hx
        exact quasispectrum_nonneg_of_nonneg c hc _ hx
    _ = _ := key.symm
    _ = a⁺ := by
      refine cfcₙ_congr fun x hx ↦ ?_
      lift x to σₙ ℝ a using hx
      simp [zero, Subtype.val_injective.extend_apply, f]


@[simp]
lemma posPart_one : (1 : A)⁺ = 1 := by
  /-
    A : Type u_1
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra Real A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    ⊢ Eq (PosPart.posPart 1) 1
  -/
  rw [CFC.posPart_def, cfcₙ_eq_cfc]
  /-
    A : Type u_1
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra Real A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    ⊢ Eq (cfc (fun x => PosPart.posPart x) 1) 1
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma negPart_one : (1 : A)⁻ = 0 := by
  /-
    A : Type u_1
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra Real A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    ⊢ Eq (NegPart.negPart 1) 0
  -/
  rw [CFC.negPart_def, cfcₙ_eq_cfc]
  /-
    A : Type u_1
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra Real A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    ⊢ Eq (cfc (fun x => NegPart.negPart x) 1) 0
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma posPart_algebraMap (r : ℝ) : (algebraMap ℝ A r)⁺ = algebraMap ℝ A r⁺ := by
  /-
    A : Type u_1
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra Real A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    r : Real
    ⊢ Eq (PosPart.posPart ((algebraMap Real A) r)) ((algebraMap Real A) (PosPart.p …
  -/
  rw [CFC.posPart_def, cfcₙ_eq_cfc]
  /-
    A : Type u_1
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra Real A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    r : Real
    ⊢ Eq (cfc (fun x => PosPart.posPart x) ((algebraMap Real A) r)) ((algebraMap R …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma negPart_algebraMap (r : ℝ) : (algebraMap ℝ A r)⁻ = algebraMap ℝ A r⁻ := by
  /-
    A : Type u_1
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra Real A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    r : Real
    ⊢ Eq (NegPart.negPart ((algebraMap Real A) r)) ((algebraMap Real A) (NegPart.n …
  -/
  rw [CFC.negPart_def, cfcₙ_eq_cfc]
  /-
    A : Type u_1
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra Real A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    r : Real
    ⊢ Eq (cfc (fun x => NegPart.negPart x) ((algebraMap Real A) r)) ((algebraMap R …
  -/
  simp
  /-
    🎉 no goals
  -/


open NNReal in
@[simp]
lemma posPart_algebraMap_nnreal (r : ℝ≥0) : (algebraMap ℝ≥0 A r)⁺ = algebraMap ℝ≥0 A r := by
  /-
    A : Type u_1
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra Real A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    r : NNReal
    ⊢ Eq (PosPart.posPart ((algebraMap NNReal A) r)) ((algebraMap NNReal A) r)
  -/
  rw [CFC.posPart_def, cfcₙ_eq_cfc, IsScalarTower.algebraMap_apply ℝ≥0 ℝ A]
  /-
    A : Type u_1
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra Real A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    r : NNReal
    ⊢ Eq (cfc (fun x => PosPart.posPart x) ((algebraMap Real A) ((algebraMap NNRea …
  -/
  simp
  /-
    🎉 no goals
  -/


open NNReal in
@[simp]
lemma posPart_natCast (n : ℕ) : (n : A)⁺ = n := by
  /-
    A : Type u_1
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra Real A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    n : Nat
    ⊢ Eq (PosPart.posPart ↑n) ↑n
  -/
  rw [← map_natCast (algebraMap ℝ≥0 A), posPart_algebraMap_nnreal]
  /-
    🎉 no goals
  -/


lemma CStarAlgebra.linear_combination_nonneg (x : A) :
    ((ℜ x : A)⁺ - (ℜ x : A)⁻) + (I • (ℑ x : A)⁺ - I • (ℑ x : A)⁻) = x := by
  rw [CFC.posPart_sub_negPart _ (ℜ x).2, ← smul_sub, CFC.posPart_sub_negPart _ (ℑ x).2,
    realPart_add_I_smul_imaginaryPart x]


/-- A C⋆-algebra is spanned by its nonnegative elements. -/
lemma CStarAlgebra.span_nonneg : Submodule.span ℂ {a : A | 0 ≤ a} = ⊤ := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Complex A
    inst✝⁷ : SMulCommClass Complex A A
    inst✝⁶ : IsScalarTower Complex A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : StarModule Complex A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    ⊢ Eq (Submodule.span Complex (setOf fun a => LE.le 0 a)) Top.top
  -/
  refine eq_top_iff.mpr fun x _ => ?_
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Complex A
    inst✝⁷ : SMulCommClass Complex A A
    inst✝⁶ : IsScalarTower Complex A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : StarModule Complex A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x : A
    x✝ : Membership.mem Top.top x
    ⊢ Membership.mem (Submodule.span Complex (setOf fun a => LE.le 0 a)) x
  -/
  rw [← CStarAlgebra.linear_combination_nonneg x]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : Module Complex A
    inst✝⁷ : SMulCommClass Complex A A
    inst✝⁶ : IsScalarTower Complex A A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : StarModule Complex A
    inst✝² : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x : A
    x✝ : Membership.mem Top.top x
    ⊢ Membership.mem (Submodule.span Complex (setOf fun a => LE.le 0 a)) (HAdd.hAd …
  -/
  apply_rules [sub_mem, Submodule.smul_mem, add_mem]
  all_goals
    refine subset_span ?_
    first | apply CFC.negPart_nonneg | apply CFC.posPart_nonneg


