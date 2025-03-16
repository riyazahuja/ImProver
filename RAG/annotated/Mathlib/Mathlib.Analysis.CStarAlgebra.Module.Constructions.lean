/-- Reinterpret a C⋆-algebra `A` as a `CStarModule` over itself. -/
instance : CStarModule A A where
  inner x y := star x * y
  inner_add_right := mul_add ..
  inner_self_nonneg := star_mul_self_nonneg _
  inner_self := CStarRing.star_mul_self_eq_zero_iff _
  inner_op_smul_right := mul_assoc .. |>.symm
  inner_smul_right_complex := mul_smul_comm ..
                       /-
                         A : Type u_1
                         inst✝² : NonUnitalCStarAlgebra A
                         inst✝¹ : PartialOrder A
                         inst✝ : StarOrderedRing A
                         x y : A
                         ⊢ Eq (Star.star (Inner.inner x y)) (Inner.inner y x)
                       -/
  star_inner x y := by simp
                       /-
                         🎉 no goals
                       -/
  norm_eq_sqrt_norm_inner_self {x} := by
    /-
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      x : A
      ⊢ Eq (Norm.norm x) (Norm.norm (Inner.inner x x)).sqrt
    -/
    rw [← sq_eq_sq₀ (norm_nonneg _) (by positivity)]
    /-
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      x : A
      ⊢ Eq (HPow.hPow (Norm.norm x) 2) (HPow.hPow (Norm.norm (Inner.inner x x)).sqrt …
    -/
    simpa [sq] using Eq.symm <| CStarRing.norm_star_mul_self
    /-
      🎉 no goals
    -/


open scoped InnerProductSpace in
lemma inner_def (x y : A) : ⟪x, y⟫_A = star x * y := rfl


noncomputable instance : Norm (C⋆ᵐᵒᵈ (E × F)) where
  norm x := √‖⟪x.1, x.1⟫_A + ⟪x.2, x.2⟫_A‖


lemma prod_norm (x : C⋆ᵐᵒᵈ (E × F)) : ‖x‖ = √‖⟪x.1, x.1⟫_A + ⟪x.2, x.2⟫_A‖ := rfl


lemma prod_norm_sq (x : C⋆ᵐᵒᵈ (E × F)) : ‖x‖ ^ 2 = ‖⟪x.1, x.1⟫_A + ⟪x.2, x.2⟫_A‖ := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalCStarAlgebra A
    inst✝⁸ : PartialOrder A
    E : Type u_2
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : Module Complex E
    inst✝⁵ : SMul (MulOpposite A) E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : Module Complex F
    inst✝² : SMul (MulOpposite A) F
    inst✝¹ : CStarModule A E
    inst✝ : CStarModule A F
    x : WithCStarModule (Prod E F)
    ⊢ Eq (HPow.hPow (Norm.norm x) 2) (Norm.norm (HAdd.hAdd (Inner.inner x.1 x.1) ( …
  -/
  simp [prod_norm]
  /-
    🎉 no goals
  -/


lemma prod_norm_le_norm_add (x : C⋆ᵐᵒᵈ (E × F)) : ‖x‖ ≤ ‖x.1‖ + ‖x.2‖ := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalCStarAlgebra A
    inst✝⁸ : PartialOrder A
    E : Type u_2
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : Module Complex E
    inst✝⁵ : SMul (MulOpposite A) E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : Module Complex F
    inst✝² : SMul (MulOpposite A) F
    inst✝¹ : CStarModule A E
    inst✝ : CStarModule A F
    x : WithCStarModule (Prod E F)
    ⊢ LE.le (Norm.norm x) (HAdd.hAdd (Norm.norm x.1) (Norm.norm x.2))
  -/
  refine abs_le_of_sq_le_sq' ?_ (by positivity) |>.2
  calc ‖x‖ ^ 2 ≤ ‖⟪x.1, x.1⟫_A‖ + ‖⟪x.2, x.2⟫_A‖ := prod_norm_sq x ▸ norm_add_le _ _
    _ = ‖x.1‖ ^ 2 + 0 + ‖x.2‖ ^ 2 := by simp [norm_sq_eq]
    _ ≤ ‖x.1‖ ^ 2 + 2 * ‖x.1‖ * ‖x.2‖ + ‖x.2‖ ^ 2 := by gcongr; positivity
    _ = (‖x.1‖ + ‖x.2‖) ^ 2 := by ring


noncomputable instance : CStarModule A (C⋆ᵐᵒᵈ (E × F)) where
  inner x y := inner x.1 y.1 + inner x.2 y.2
                                /-
                                  A : Type u_1
                                  inst✝¹⁰ : NonUnitalCStarAlgebra A
                                  inst✝⁹ : PartialOrder A
                                  E : Type u_2
                                  F : Type u_3
                                  inst✝⁸ : NormedAddCommGroup E
                                  inst✝⁷ : Module Complex E
                                  inst✝⁶ : SMul (MulOpposite A) E
                                  inst✝⁵ : NormedAddCommGroup F
                                  inst✝⁴ : Module Complex F
                                  inst✝³ : SMul (MulOpposite A) F
                                  inst✝² : CStarModule A E
                                  inst✝¹ : CStarModule A F
                                  inst✝ : StarOrderedRing A
                                  x y z : WithCStarModule (Prod E F)
                                  ⊢ Eq (Inner.inner x (HAdd.hAdd y z)) (HAdd.hAdd (Inner.inner x y) (Inner.inner …
                                -/
  inner_add_right {x y z} := by simpa using add_add_add_comm ..
                                /-
                                  🎉 no goals
                                -/
  inner_self_nonneg := add_nonneg CStarModule.inner_self_nonneg CStarModule.inner_self_nonneg
  inner_self {x} := by
    /-
      A : Type u_1
      inst✝¹⁰ : NonUnitalCStarAlgebra A
      inst✝⁹ : PartialOrder A
      E : Type u_2
      F : Type u_3
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : Module Complex E
      inst✝⁶ : SMul (MulOpposite A) E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : Module Complex F
      inst✝³ : SMul (MulOpposite A) F
      inst✝² : CStarModule A E
      inst✝¹ : CStarModule A F
      inst✝ : StarOrderedRing A
      x : WithCStarModule (Prod E F)
      ⊢ Iff (Eq (Inner.inner x x) 0) (Eq x 0)
    -/
    refine ⟨fun h ↦ ?_, fun h ↦ by simp [h, CStarModule.inner_zero_left]⟩
    /-
      A : Type u_1
      inst✝¹⁰ : NonUnitalCStarAlgebra A
      inst✝⁹ : PartialOrder A
      E : Type u_2
      F : Type u_3
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : Module Complex E
      inst✝⁶ : SMul (MulOpposite A) E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : Module Complex F
      inst✝³ : SMul (MulOpposite A) F
      inst✝² : CStarModule A E
      inst✝¹ : CStarModule A F
      inst✝ : StarOrderedRing A
      x : WithCStarModule (Prod E F)
      h : Eq (Inner.inner x x) 0
      ⊢ Eq x 0
    -/
    apply equiv (E × F) |>.injective
    /-
      case a
      A : Type u_1
      inst✝¹⁰ : NonUnitalCStarAlgebra A
      inst✝⁹ : PartialOrder A
      E : Type u_2
      F : Type u_3
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : Module Complex E
      inst✝⁶ : SMul (MulOpposite A) E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : Module Complex F
      inst✝³ : SMul (MulOpposite A) F
      inst✝² : CStarModule A E
      inst✝¹ : CStarModule A F
      inst✝ : StarOrderedRing A
      x : WithCStarModule (Prod E F)
      h : Eq (Inner.inner x x) 0
      ⊢ Eq ((WithCStarModule.equiv (Prod E F)) x) ((WithCStarModule.equiv (Prod E F) …
    -/
    ext
      /-
        case a.fst
        A : Type u_1
        inst✝¹⁰ : NonUnitalCStarAlgebra A
        inst✝⁹ : PartialOrder A
        E : Type u_2
        F : Type u_3
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : Module Complex E
        inst✝⁶ : SMul (MulOpposite A) E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : Module Complex F
        inst✝³ : SMul (MulOpposite A) F
        inst✝² : CStarModule A E
        inst✝¹ : CStarModule A F
        inst✝ : StarOrderedRing A
        x : WithCStarModule (Prod E F)
        h : Eq (Inner.inner x x) 0
        ⊢ Eq ((WithCStarModule.equiv (Prod E F)) x).1 ((WithCStarModule.equiv (Prod E  …
      -/
    · refine inner_self.mp <| le_antisymm ?_ inner_self_nonneg
      /-
        case a.fst
        A : Type u_1
        inst✝¹⁰ : NonUnitalCStarAlgebra A
        inst✝⁹ : PartialOrder A
        E : Type u_2
        F : Type u_3
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : Module Complex E
        inst✝⁶ : SMul (MulOpposite A) E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : Module Complex F
        inst✝³ : SMul (MulOpposite A) F
        inst✝² : CStarModule A E
        inst✝¹ : CStarModule A F
        inst✝ : StarOrderedRing A
        x : WithCStarModule (Prod E F)
        h : Eq (Inner.inner x x) 0
        ⊢ LE.le (Inner.inner ((WithCStarModule.equiv (Prod E F)) x).1 ((WithCStarModul …
      -/
      exact le_add_of_nonneg_right CStarModule.inner_self_nonneg |>.trans_eq h
      /-
        🎉 no goals
      -/
      /-
        case a.snd
        A : Type u_1
        inst✝¹⁰ : NonUnitalCStarAlgebra A
        inst✝⁹ : PartialOrder A
        E : Type u_2
        F : Type u_3
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : Module Complex E
        inst✝⁶ : SMul (MulOpposite A) E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : Module Complex F
        inst✝³ : SMul (MulOpposite A) F
        inst✝² : CStarModule A E
        inst✝¹ : CStarModule A F
        inst✝ : StarOrderedRing A
        x : WithCStarModule (Prod E F)
        h : Eq (Inner.inner x x) 0
        ⊢ Eq ((WithCStarModule.equiv (Prod E F)) x).2 ((WithCStarModule.equiv (Prod E  …
      -/
    · refine inner_self.mp <| le_antisymm ?_ inner_self_nonneg
      /-
        case a.snd
        A : Type u_1
        inst✝¹⁰ : NonUnitalCStarAlgebra A
        inst✝⁹ : PartialOrder A
        E : Type u_2
        F : Type u_3
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : Module Complex E
        inst✝⁶ : SMul (MulOpposite A) E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : Module Complex F
        inst✝³ : SMul (MulOpposite A) F
        inst✝² : CStarModule A E
        inst✝¹ : CStarModule A F
        inst✝ : StarOrderedRing A
        x : WithCStarModule (Prod E F)
        h : Eq (Inner.inner x x) 0
        ⊢ LE.le (Inner.inner ((WithCStarModule.equiv (Prod E F)) x).2 ((WithCStarModul …
      -/
      exact le_add_of_nonneg_left CStarModule.inner_self_nonneg |>.trans_eq h
      /-
        🎉 no goals
      -/
                            /-
                              A : Type u_1
                              inst✝¹⁰ : NonUnitalCStarAlgebra A
                              inst✝⁹ : PartialOrder A
                              E : Type u_2
                              F : Type u_3
                              inst✝⁸ : NormedAddCommGroup E
                              inst✝⁷ : Module Complex E
                              inst✝⁶ : SMul (MulOpposite A) E
                              inst✝⁵ : NormedAddCommGroup F
                              inst✝⁴ : Module Complex F
                              inst✝³ : SMul (MulOpposite A) F
                              inst✝² : CStarModule A E
                              inst✝¹ : CStarModule A F
                              inst✝ : StarOrderedRing A
                              ⊢ ∀ {a : A} {x y : WithCStarModule (Prod E F)}, Eq (Inner.inner x (HSMul.hSMul …
                            -/
  inner_op_smul_right := by simp [add_mul]
                            /-
                              🎉 no goals
                            -/
                                 /-
                                   A : Type u_1
                                   inst✝¹⁰ : NonUnitalCStarAlgebra A
                                   inst✝⁹ : PartialOrder A
                                   E : Type u_2
                                   F : Type u_3
                                   inst✝⁸ : NormedAddCommGroup E
                                   inst✝⁷ : Module Complex E
                                   inst✝⁶ : SMul (MulOpposite A) E
                                   inst✝⁵ : NormedAddCommGroup F
                                   inst✝⁴ : Module Complex F
                                   inst✝³ : SMul (MulOpposite A) F
                                   inst✝² : CStarModule A E
                                   inst✝¹ : CStarModule A F
                                   inst✝ : StarOrderedRing A
                                   ⊢ ∀ {z : Complex} {x y : WithCStarModule (Prod E F)}, Eq (Inner.inner x (HSMul …
                                 -/
  inner_smul_right_complex := by simp [smul_add]
                                 /-
                                   🎉 no goals
                                 -/
                       /-
                         A : Type u_1
                         inst✝¹⁰ : NonUnitalCStarAlgebra A
                         inst✝⁹ : PartialOrder A
                         E : Type u_2
                         F : Type u_3
                         inst✝⁸ : NormedAddCommGroup E
                         inst✝⁷ : Module Complex E
                         inst✝⁶ : SMul (MulOpposite A) E
                         inst✝⁵ : NormedAddCommGroup F
                         inst✝⁴ : Module Complex F
                         inst✝³ : SMul (MulOpposite A) F
                         inst✝² : CStarModule A E
                         inst✝¹ : CStarModule A F
                         inst✝ : StarOrderedRing A
                         x y : WithCStarModule (Prod E F)
                         ⊢ Eq (Star.star (Inner.inner x y)) (Inner.inner y x)
                       -/
  star_inner x y := by simp
                       /-
                         🎉 no goals
                       -/
                                         /-
                                           A : Type u_1
                                           inst✝¹⁰ : NonUnitalCStarAlgebra A
                                           inst✝⁹ : PartialOrder A
                                           E : Type u_2
                                           F : Type u_3
                                           inst✝⁸ : NormedAddCommGroup E
                                           inst✝⁷ : Module Complex E
                                           inst✝⁶ : SMul (MulOpposite A) E
                                           inst✝⁵ : NormedAddCommGroup F
                                           inst✝⁴ : Module Complex F
                                           inst✝³ : SMul (MulOpposite A) F
                                           inst✝² : CStarModule A E
                                           inst✝¹ : CStarModule A F
                                           inst✝ : StarOrderedRing A
                                           x : WithCStarModule (Prod E F)
                                           ⊢ Eq (Norm.norm x) (Norm.norm (Inner.inner x x)).sqrt
                                         -/
  norm_eq_sqrt_norm_inner_self {x} := by with_reducible_and_instances rfl
                                         /-
                                           🎉 no goals
                                         -/


lemma prod_inner (x y : C⋆ᵐᵒᵈ (E × F)) : ⟪x, y⟫_A = ⟪x.1, y.1⟫_A + ⟪x.2, y.2⟫_A := rfl


lemma max_le_prod_norm (x : C⋆ᵐᵒᵈ (E × F)) : max ‖x.1‖ ‖x.2‖ ≤ ‖x‖ := by
  /-
    A : Type u_1
    inst✝¹⁰ : NonUnitalCStarAlgebra A
    inst✝⁹ : PartialOrder A
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : Module Complex E
    inst✝⁶ : SMul (MulOpposite A) E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : Module Complex F
    inst✝³ : SMul (MulOpposite A) F
    inst✝² : CStarModule A E
    inst✝¹ : CStarModule A F
    inst✝ : StarOrderedRing A
    x : WithCStarModule (Prod E F)
    ⊢ LE.le (Max.max (Norm.norm x.1) (Norm.norm x.2)) (Norm.norm x)
  -/
  rw [prod_norm]
  simp only [equiv_fst, norm_eq_sqrt_norm_inner_self (E := E),
    norm_eq_sqrt_norm_inner_self (E := F), equiv_snd, max_le_iff, norm_nonneg,
    Real.sqrt_le_sqrt_iff]
  /-
    A : Type u_1
    inst✝¹⁰ : NonUnitalCStarAlgebra A
    inst✝⁹ : PartialOrder A
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : Module Complex E
    inst✝⁶ : SMul (MulOpposite A) E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : Module Complex F
    inst✝³ : SMul (MulOpposite A) F
    inst✝² : CStarModule A E
    inst✝¹ : CStarModule A F
    inst✝ : StarOrderedRing A
    x : WithCStarModule (Prod E F)
    ⊢ And (LE.le (Norm.norm (Inner.inner x.1 x.1)) (Norm.norm (HAdd.hAdd (Inner.in …
  -/
  constructor
  all_goals
    apply CStarAlgebra.norm_le_norm_of_nonneg_of_le
    all_goals
      aesop (add safe apply CStarModule.inner_self_nonneg)


lemma norm_equiv_le_norm_prod (x : C⋆ᵐᵒᵈ (E × F)) : ‖equiv (E × F) x‖ ≤ ‖x‖ :=
  max_le_prod_norm x


private lemma antilipschitzWith_two_equiv_prod_aux : AntilipschitzWith 2 (equiv (E × F)) :=
  AddMonoidHomClass.antilipschitz_of_bound (linearEquiv ℂ (E × F)) fun x ↦ by
    /-
      A : Type u_1
      inst✝¹⁰ : NonUnitalCStarAlgebra A
      inst✝⁹ : PartialOrder A
      E : Type u_2
      F : Type u_3
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : Module Complex E
      inst✝⁶ : SMul (MulOpposite A) E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : Module Complex F
      inst✝³ : SMul (MulOpposite A) F
      inst✝² : CStarModule A E
      inst✝¹ : CStarModule A F
      inst✝ : StarOrderedRing A
      x : WithCStarModule (Prod E F)
      ⊢ LE.le (Norm.norm x) (HMul.hMul (↑2) (Norm.norm ((WithCStarModule.linearEquiv …
    -/
    apply prod_norm_le_norm_add x |>.trans
    /-
      A : Type u_1
      inst✝¹⁰ : NonUnitalCStarAlgebra A
      inst✝⁹ : PartialOrder A
      E : Type u_2
      F : Type u_3
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : Module Complex E
      inst✝⁶ : SMul (MulOpposite A) E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : Module Complex F
      inst✝³ : SMul (MulOpposite A) F
      inst✝² : CStarModule A E
      inst✝¹ : CStarModule A F
      inst✝ : StarOrderedRing A
      x : WithCStarModule (Prod E F)
      ⊢ LE.le (HAdd.hAdd (Norm.norm x.1) (Norm.norm x.2)) (HMul.hMul (↑2) (Norm.norm …
    -/
    simp only [NNReal.coe_ofNat, linearEquiv_apply, two_mul]
    /-
      A : Type u_1
      inst✝¹⁰ : NonUnitalCStarAlgebra A
      inst✝⁹ : PartialOrder A
      E : Type u_2
      F : Type u_3
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : Module Complex E
      inst✝⁶ : SMul (MulOpposite A) E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : Module Complex F
      inst✝³ : SMul (MulOpposite A) F
      inst✝² : CStarModule A E
      inst✝¹ : CStarModule A F
      inst✝ : StarOrderedRing A
      x : WithCStarModule (Prod E F)
      ⊢ LE.le (HAdd.hAdd (Norm.norm x.1) (Norm.norm x.2)) (HAdd.hAdd (Norm.norm ((Wi …
    -/
    gcongr
      /-
        case h₁
        A : Type u_1
        inst✝¹⁰ : NonUnitalCStarAlgebra A
        inst✝⁹ : PartialOrder A
        E : Type u_2
        F : Type u_3
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : Module Complex E
        inst✝⁶ : SMul (MulOpposite A) E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : Module Complex F
        inst✝³ : SMul (MulOpposite A) F
        inst✝² : CStarModule A E
        inst✝¹ : CStarModule A F
        inst✝ : StarOrderedRing A
        x : WithCStarModule (Prod E F)
        ⊢ LE.le (Norm.norm x.1) (Norm.norm ((WithCStarModule.equiv (Prod E F)) x))
      -/
    · exact norm_fst_le x
      /-
        🎉 no goals
      -/
      /-
        case h₂
        A : Type u_1
        inst✝¹⁰ : NonUnitalCStarAlgebra A
        inst✝⁹ : PartialOrder A
        E : Type u_2
        F : Type u_3
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : Module Complex E
        inst✝⁶ : SMul (MulOpposite A) E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : Module Complex F
        inst✝³ : SMul (MulOpposite A) F
        inst✝² : CStarModule A E
        inst✝¹ : CStarModule A F
        inst✝ : StarOrderedRing A
        x : WithCStarModule (Prod E F)
        ⊢ LE.le (Norm.norm x.2) (Norm.norm ((WithCStarModule.equiv (Prod E F)) x))
      -/
    · exact norm_snd_le x
      /-
        🎉 no goals
      -/


private lemma lipschitzWith_one_equiv_prod_aux : LipschitzWith 1 (equiv (E × F)) :=
  AddMonoidHomClass.lipschitz_of_bound_nnnorm (linearEquiv ℂ (E × F)) 1 <| by
    /-
      A : Type u_1
      inst✝¹⁰ : NonUnitalCStarAlgebra A
      inst✝⁹ : PartialOrder A
      E : Type u_2
      F : Type u_3
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : Module Complex E
      inst✝⁶ : SMul (MulOpposite A) E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : Module Complex F
      inst✝³ : SMul (MulOpposite A) F
      inst✝² : CStarModule A E
      inst✝¹ : CStarModule A F
      inst✝ : StarOrderedRing A
      ⊢ ∀ (x : WithCStarModule (Prod E F)), LE.le (NNNorm.nnnorm ((WithCStarModule.l …
    -/
    simpa using norm_equiv_le_norm_prod
    /-
      🎉 no goals
    -/


private lemma uniformity_prod_eq_aux :
    𝓤[(inferInstance : UniformSpace (E × F)).comap <| equiv _] = 𝓤 (C⋆ᵐᵒᵈ (E × F)) :=
  uniformity_eq_of_bilipschitz antilipschitzWith_two_equiv_prod_aux lipschitzWith_one_equiv_prod_aux


private lemma isBounded_prod_iff_aux (s : Set (C⋆ᵐᵒᵈ (E × F))) :
    @IsBounded _ (induced <| equiv (E × F)) s ↔ IsBounded s :=
  isBounded_iff_of_bilipschitz antilipschitzWith_two_equiv_prod_aux
    lipschitzWith_one_equiv_prod_aux s


noncomputable instance : NormedAddCommGroup (C⋆ᵐᵒᵈ (E × F)) :=
  .ofCoreReplaceAll normedSpaceCore uniformity_prod_eq_aux isBounded_prod_iff_aux


instance : NormedSpace ℂ (C⋆ᵐᵒᵈ (E × F)) := .ofCore normedSpaceCore


noncomputable instance : Norm (C⋆ᵐᵒᵈ (Π i, E i)) where
  norm x := √‖∑ i, ⟪x i, x i⟫_A‖


lemma pi_norm (x : C⋆ᵐᵒᵈ (Π i, E i)) : ‖x‖ = √‖∑ i, ⟪x i, x i⟫_A‖ := by
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalCStarAlgebra A
    inst✝⁵ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁴ : Fintype ι
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → Module Complex (E i)
    inst✝¹ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝ : (i : ι) → CStarModule A (E i)
    x : WithCStarModule ((i : ι) → E i)
    ⊢ Eq (Norm.norm x) (Norm.norm (Finset.univ.sum fun i => Inner.inner (x i) (x i …
  -/
  with_reducible_and_instances rfl
  /-
    🎉 no goals
  -/


lemma pi_norm_sq (x : C⋆ᵐᵒᵈ (Π i, E i)) : ‖x‖ ^ 2 = ‖∑ i, ⟪x i, x i⟫_A‖ := by
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalCStarAlgebra A
    inst✝⁵ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁴ : Fintype ι
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → Module Complex (E i)
    inst✝¹ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝ : (i : ι) → CStarModule A (E i)
    x : WithCStarModule ((i : ι) → E i)
    ⊢ Eq (HPow.hPow (Norm.norm x) 2) (Norm.norm (Finset.univ.sum fun i => Inner.in …
  -/
  simp [pi_norm]
  /-
    🎉 no goals
  -/


open Finset in
lemma pi_norm_le_sum_norm (x : C⋆ᵐᵒᵈ (Π i, E i)) : ‖x‖ ≤ ∑ i, ‖x i‖ := by
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalCStarAlgebra A
    inst✝⁵ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁴ : Fintype ι
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → Module Complex (E i)
    inst✝¹ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝ : (i : ι) → CStarModule A (E i)
    x : WithCStarModule ((i : ι) → E i)
    ⊢ LE.le (Norm.norm x) (Finset.univ.sum fun i => Norm.norm (x i))
  -/
  refine abs_le_of_sq_le_sq' ?_ (by positivity) |>.2
  calc ‖x‖ ^ 2 ≤ ∑ i, ‖⟪x i, x i⟫_A‖ := pi_norm_sq x ▸ norm_sum_le _ _
    _ = ∑ i, ‖x i‖ ^ 2 := by simp only [norm_sq_eq]
    _ ≤ (∑ i, ‖x i‖) ^ 2 := sum_sq_le_sq_sum_of_nonneg (fun _ _ ↦ norm_nonneg _)


open Finset in
noncomputable instance : CStarModule A (C⋆ᵐᵒᵈ (Π i, E i)) where
  inner x y := ∑ i, inner (x i) (y i)
                                /-
                                  A : Type u_1
                                  inst✝⁷ : NonUnitalCStarAlgebra A
                                  inst✝⁶ : PartialOrder A
                                  ι : Type u_2
                                  E : ι → Type u_3
                                  inst✝⁵ : Fintype ι
                                  inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
                                  inst✝³ : (i : ι) → Module Complex (E i)
                                  inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
                                  inst✝¹ : (i : ι) → CStarModule A (E i)
                                  inst✝ : StarOrderedRing A
                                  x y z : WithCStarModule ((i : ι) → E i)
                                  ⊢ Eq (Inner.inner x (HAdd.hAdd y z)) (HAdd.hAdd (Inner.inner x y) (Inner.inner …
                                -/
  inner_add_right {x y z} := by simp [inner_sum_right, sum_add_distrib]
                                /-
                                  🎉 no goals
                                -/
  inner_self_nonneg := sum_nonneg <| fun _ _ ↦ CStarModule.inner_self_nonneg
  inner_self {x} := by
    /-
      A : Type u_1
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      ι : Type u_2
      E : ι → Type u_3
      inst✝⁵ : Fintype ι
      inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
      inst✝³ : (i : ι) → Module Complex (E i)
      inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
      inst✝¹ : (i : ι) → CStarModule A (E i)
      inst✝ : StarOrderedRing A
      x : WithCStarModule ((i : ι) → E i)
      ⊢ Iff (Eq (Inner.inner x x) 0) (Eq x 0)
    -/
    refine ⟨fun h ↦ ?_, fun h ↦ by simp [h, CStarModule.inner_zero_left]⟩
    /-
      A : Type u_1
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      ι : Type u_2
      E : ι → Type u_3
      inst✝⁵ : Fintype ι
      inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
      inst✝³ : (i : ι) → Module Complex (E i)
      inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
      inst✝¹ : (i : ι) → CStarModule A (E i)
      inst✝ : StarOrderedRing A
      x : WithCStarModule ((i : ι) → E i)
      h : Eq (Inner.inner x x) 0
      ⊢ Eq x 0
    -/
    ext i
    /-
      case h
      A : Type u_1
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      ι : Type u_2
      E : ι → Type u_3
      inst✝⁵ : Fintype ι
      inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
      inst✝³ : (i : ι) → Module Complex (E i)
      inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
      inst✝¹ : (i : ι) → CStarModule A (E i)
      inst✝ : StarOrderedRing A
      x : WithCStarModule ((i : ι) → E i)
      h : Eq (Inner.inner x x) 0
      i : ι
      ⊢ Eq (x i) (0 i)
    -/
    refine inner_self.mp <| le_antisymm (le_of_le_of_eq ?_ h) inner_self_nonneg
    /-
      case h
      A : Type u_1
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      ι : Type u_2
      E : ι → Type u_3
      inst✝⁵ : Fintype ι
      inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
      inst✝³ : (i : ι) → Module Complex (E i)
      inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
      inst✝¹ : (i : ι) → CStarModule A (E i)
      inst✝ : StarOrderedRing A
      x : WithCStarModule ((i : ι) → E i)
      h : Eq (Inner.inner x x) 0
      i : ι
      ⊢ LE.le (Inner.inner (x i) (x i)) (Inner.inner x x)
    -/
    exact single_le_sum (fun i _ ↦ CStarModule.inner_self_nonneg (x := x i)) (mem_univ _)
    /-
      🎉 no goals
    -/
                            /-
                              A : Type u_1
                              inst✝⁷ : NonUnitalCStarAlgebra A
                              inst✝⁶ : PartialOrder A
                              ι : Type u_2
                              E : ι → Type u_3
                              inst✝⁵ : Fintype ι
                              inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
                              inst✝³ : (i : ι) → Module Complex (E i)
                              inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
                              inst✝¹ : (i : ι) → CStarModule A (E i)
                              inst✝ : StarOrderedRing A
                              ⊢ ∀ {a : A} {x y : WithCStarModule ((i : ι) → E i)}, Eq (Inner.inner x (HSMul. …
                            -/
  inner_op_smul_right := by simp [sum_mul]
                            /-
                              🎉 no goals
                            -/
                                 /-
                                   A : Type u_1
                                   inst✝⁷ : NonUnitalCStarAlgebra A
                                   inst✝⁶ : PartialOrder A
                                   ι : Type u_2
                                   E : ι → Type u_3
                                   inst✝⁵ : Fintype ι
                                   inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
                                   inst✝³ : (i : ι) → Module Complex (E i)
                                   inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
                                   inst✝¹ : (i : ι) → CStarModule A (E i)
                                   inst✝ : StarOrderedRing A
                                   ⊢ ∀ {z : Complex} {x y : WithCStarModule ((i : ι) → E i)}, Eq (Inner.inner x ( …
                                 -/
  inner_smul_right_complex := by simp [smul_sum]
                                 /-
                                   🎉 no goals
                                 -/
                       /-
                         A : Type u_1
                         inst✝⁷ : NonUnitalCStarAlgebra A
                         inst✝⁶ : PartialOrder A
                         ι : Type u_2
                         E : ι → Type u_3
                         inst✝⁵ : Fintype ι
                         inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
                         inst✝³ : (i : ι) → Module Complex (E i)
                         inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
                         inst✝¹ : (i : ι) → CStarModule A (E i)
                         inst✝ : StarOrderedRing A
                         x y : WithCStarModule ((i : ι) → E i)
                         ⊢ Eq (Star.star (Inner.inner x y)) (Inner.inner y x)
                       -/
  star_inner x y := by simp
                       /-
                         🎉 no goals
                       -/
                                         /-
                                           A : Type u_1
                                           inst✝⁷ : NonUnitalCStarAlgebra A
                                           inst✝⁶ : PartialOrder A
                                           ι : Type u_2
                                           E : ι → Type u_3
                                           inst✝⁵ : Fintype ι
                                           inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
                                           inst✝³ : (i : ι) → Module Complex (E i)
                                           inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
                                           inst✝¹ : (i : ι) → CStarModule A (E i)
                                           inst✝ : StarOrderedRing A
                                           x : WithCStarModule ((i : ι) → E i)
                                           ⊢ Eq (Norm.norm x) (Norm.norm (Inner.inner x x)).sqrt
                                         -/
  norm_eq_sqrt_norm_inner_self {x} := by with_reducible_and_instances rfl
                                         /-
                                           🎉 no goals
                                         -/


lemma pi_inner (x y : C⋆ᵐᵒᵈ (Π i, E i)) : ⟪x, y⟫_A = ∑ i, ⟪x i, y i⟫_A := rfl


@[simp]
lemma inner_single_left [DecidableEq ι] (x : C⋆ᵐᵒᵈ (Π i, E i)) {i : ι} (y : E i) :
    ⟪equiv _ |>.symm <| Pi.single i y, x⟫_A = ⟪y, x i⟫_A := by
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalCStarAlgebra A
    inst✝⁷ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁶ : Fintype ι
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → Module Complex (E i)
    inst✝³ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝² : (i : ι) → CStarModule A (E i)
    inst✝¹ : StarOrderedRing A
    inst✝ : DecidableEq ι
    x : WithCStarModule ((i : ι) → E i)
    i : ι
    y : E i
    ⊢ Eq (Inner.inner ((WithCStarModule.equiv ((j : ι) → E j)).symm (Pi.single i y …
  -/
  simp only [pi_inner, equiv_symm_pi_apply]
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalCStarAlgebra A
    inst✝⁷ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁶ : Fintype ι
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → Module Complex (E i)
    inst✝³ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝² : (i : ι) → CStarModule A (E i)
    inst✝¹ : StarOrderedRing A
    inst✝ : DecidableEq ι
    x : WithCStarModule ((i : ι) → E i)
    i : ι
    y : E i
    ⊢ Eq (Finset.univ.sum fun x_1 => Inner.inner (Pi.single i y x_1) (x x_1)) (Inn …
  -/
  rw [Finset.sum_eq_single i]
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalCStarAlgebra A
    inst✝⁷ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁶ : Fintype ι
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → Module Complex (E i)
    inst✝³ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝² : (i : ι) → CStarModule A (E i)
    inst✝¹ : StarOrderedRing A
    inst✝ : DecidableEq ι
    x : WithCStarModule ((i : ι) → E i)
    i : ι
    y : E i
    ⊢ Eq (Inner.inner (Pi.single i y i) (x i)) (Inner.inner y (x i))
  -/
  all_goals simp_all
  /-
    🎉 no goals
  -/


@[simp]
lemma inner_single_right [DecidableEq ι] (x : C⋆ᵐᵒᵈ (Π i, E i)) {i : ι} (y : E i) :
    ⟪x, equiv _ |>.symm <| Pi.single i y⟫_A = ⟪x i, y⟫_A := by
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalCStarAlgebra A
    inst✝⁷ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁶ : Fintype ι
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → Module Complex (E i)
    inst✝³ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝² : (i : ι) → CStarModule A (E i)
    inst✝¹ : StarOrderedRing A
    inst✝ : DecidableEq ι
    x : WithCStarModule ((i : ι) → E i)
    i : ι
    y : E i
    ⊢ Eq (Inner.inner x ((WithCStarModule.equiv ((i : ι) → E i)).symm (Pi.single i …
  -/
  simp only [pi_inner, equiv_symm_pi_apply]
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalCStarAlgebra A
    inst✝⁷ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁶ : Fintype ι
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → Module Complex (E i)
    inst✝³ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝² : (i : ι) → CStarModule A (E i)
    inst✝¹ : StarOrderedRing A
    inst✝ : DecidableEq ι
    x : WithCStarModule ((i : ι) → E i)
    i : ι
    y : E i
    ⊢ Eq (Finset.univ.sum fun x_1 => Inner.inner (x x_1) (Pi.single i y x_1)) (Inn …
  -/
  rw [Finset.sum_eq_single i]
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalCStarAlgebra A
    inst✝⁷ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁶ : Fintype ι
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → Module Complex (E i)
    inst✝³ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝² : (i : ι) → CStarModule A (E i)
    inst✝¹ : StarOrderedRing A
    inst✝ : DecidableEq ι
    x : WithCStarModule ((i : ι) → E i)
    i : ι
    y : E i
    ⊢ Eq (Inner.inner (x i) (Pi.single i y i)) (Inner.inner (x i) y)
  -/
  all_goals simp_all
  /-
    🎉 no goals
  -/


@[simp]
lemma norm_single [DecidableEq ι] (i : ι) (y : E i) :
    ‖equiv _ |>.symm <| Pi.single i y‖ = ‖y‖ := by
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalCStarAlgebra A
    inst✝⁷ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁶ : Fintype ι
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → Module Complex (E i)
    inst✝³ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝² : (i : ι) → CStarModule A (E i)
    inst✝¹ : StarOrderedRing A
    inst✝ : DecidableEq ι
    i : ι
    y : E i
    ⊢ Eq (Norm.norm ((WithCStarModule.equiv ((j : ι) → E j)).symm (Pi.single i y)) …
  -/
  let _ : NormedAddCommGroup (C⋆ᵐᵒᵈ (Π i, E i)) := normedAddCommGroup
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalCStarAlgebra A
    inst✝⁷ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁶ : Fintype ι
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → Module Complex (E i)
    inst✝³ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝² : (i : ι) → CStarModule A (E i)
    inst✝¹ : StarOrderedRing A
    inst✝ : DecidableEq ι
    i : ι
    y : E i
    x✝ : NormedAddCommGroup (WithCStarModule ((i : ι) → E i)) := CStarModule.norme …
    ⊢ Eq (Norm.norm ((WithCStarModule.equiv ((j : ι) → E j)).symm (Pi.single i y)) …
  -/
  rw [← sq_eq_sq₀ (by positivity) (by positivity)]
  /-
    A : Type u_1
    inst✝⁸ : NonUnitalCStarAlgebra A
    inst✝⁷ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁶ : Fintype ι
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → Module Complex (E i)
    inst✝³ : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝² : (i : ι) → CStarModule A (E i)
    inst✝¹ : StarOrderedRing A
    inst✝ : DecidableEq ι
    i : ι
    y : E i
    x✝ : NormedAddCommGroup (WithCStarModule ((i : ι) → E i)) := CStarModule.norme …
    ⊢ Eq (HPow.hPow (Norm.norm ((WithCStarModule.equiv ((j : ι) → E j)).symm (Pi.s …
  -/
  simp [norm_sq_eq]
  /-
    🎉 no goals
  -/


lemma norm_apply_le_norm (x : C⋆ᵐᵒᵈ (Π i, E i)) (i : ι) : ‖x i‖ ≤ ‖x‖ := by
  /-
    A : Type u_1
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : Fintype ι
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → Module Complex (E i)
    inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝¹ : (i : ι) → CStarModule A (E i)
    inst✝ : StarOrderedRing A
    x : WithCStarModule ((i : ι) → E i)
    i : ι
    ⊢ LE.le (Norm.norm (x i)) (Norm.norm x)
  -/
  let _ : NormedAddCommGroup (C⋆ᵐᵒᵈ (Π i, E i)) := normedAddCommGroup
  /-
    A : Type u_1
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : Fintype ι
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → Module Complex (E i)
    inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝¹ : (i : ι) → CStarModule A (E i)
    inst✝ : StarOrderedRing A
    x : WithCStarModule ((i : ι) → E i)
    i : ι
    x✝ : NormedAddCommGroup (WithCStarModule ((i : ι) → E i)) := CStarModule.norme …
    ⊢ LE.le (Norm.norm (x i)) (Norm.norm x)
  -/
  refine abs_le_of_sq_le_sq' ?_ (by positivity) |>.2
  /-
    A : Type u_1
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : Fintype ι
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → Module Complex (E i)
    inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝¹ : (i : ι) → CStarModule A (E i)
    inst✝ : StarOrderedRing A
    x : WithCStarModule ((i : ι) → E i)
    i : ι
    x✝ : NormedAddCommGroup (WithCStarModule ((i : ι) → E i)) := CStarModule.norme …
    ⊢ LE.le (HPow.hPow (Norm.norm (x i)) 2) (HPow.hPow (Norm.norm x) 2)
  -/
  rw [pi_norm_sq, norm_sq_eq]
  /-
    A : Type u_1
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : Fintype ι
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → Module Complex (E i)
    inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝¹ : (i : ι) → CStarModule A (E i)
    inst✝ : StarOrderedRing A
    x : WithCStarModule ((i : ι) → E i)
    i : ι
    x✝ : NormedAddCommGroup (WithCStarModule ((i : ι) → E i)) := CStarModule.norme …
    ⊢ LE.le (Norm.norm (Inner.inner (x i) (x i))) (Norm.norm (Finset.univ.sum fun  …
  -/
  refine CStarAlgebra.norm_le_norm_of_nonneg_of_le inner_self_nonneg ?_
  /-
    A : Type u_1
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : Fintype ι
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → Module Complex (E i)
    inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝¹ : (i : ι) → CStarModule A (E i)
    inst✝ : StarOrderedRing A
    x : WithCStarModule ((i : ι) → E i)
    i : ι
    x✝ : NormedAddCommGroup (WithCStarModule ((i : ι) → E i)) := CStarModule.norme …
    ⊢ LE.le (Inner.inner (x i) (x i)) (Finset.univ.sum fun i => Inner.inner (x i)  …
  -/
  exact Finset.single_le_sum (fun j _ ↦ inner_self_nonneg (x := x j)) (Finset.mem_univ i)
  /-
    🎉 no goals
  -/


open Finset in
lemma norm_equiv_le_norm_pi (x : C⋆ᵐᵒᵈ (Π i, E i)) : ‖equiv _ x‖ ≤ ‖x‖ := by
  /-
    A : Type u_1
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : Fintype ι
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → Module Complex (E i)
    inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝¹ : (i : ι) → CStarModule A (E i)
    inst✝ : StarOrderedRing A
    x : WithCStarModule ((i : ι) → E i)
    ⊢ LE.le (Norm.norm ((WithCStarModule.equiv ((i : ι) → E i)) x)) (Norm.norm x)
  -/
  let _ : NormedAddCommGroup (C⋆ᵐᵒᵈ (Π i, E i)) := normedAddCommGroup
  /-
    A : Type u_1
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : Fintype ι
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → Module Complex (E i)
    inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝¹ : (i : ι) → CStarModule A (E i)
    inst✝ : StarOrderedRing A
    x : WithCStarModule ((i : ι) → E i)
    x✝ : NormedAddCommGroup (WithCStarModule ((i : ι) → E i)) := CStarModule.norme …
    ⊢ LE.le (Norm.norm ((WithCStarModule.equiv ((i : ι) → E i)) x)) (Norm.norm x)
  -/
  rw [pi_norm_le_iff_of_nonneg (by positivity)]
  /-
    A : Type u_1
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : Fintype ι
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → Module Complex (E i)
    inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
    inst✝¹ : (i : ι) → CStarModule A (E i)
    inst✝ : StarOrderedRing A
    x : WithCStarModule ((i : ι) → E i)
    x✝ : NormedAddCommGroup (WithCStarModule ((i : ι) → E i)) := CStarModule.norme …
    ⊢ ∀ (i : ι), LE.le (Norm.norm ((WithCStarModule.equiv ((i : ι) → E i)) x i)) ( …
  -/
  simpa using norm_apply_le_norm x
  /-
    🎉 no goals
  -/


private lemma antilipschitzWith_card_equiv_pi_aux :
    AntilipschitzWith (Fintype.card ι) (equiv (Π i, E i)) :=
  AddMonoidHomClass.antilipschitz_of_bound (linearEquiv ℂ (Π i, E i)) fun x ↦ by
    /-
      A : Type u_1
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      ι : Type u_2
      E : ι → Type u_3
      inst✝⁵ : Fintype ι
      inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
      inst✝³ : (i : ι) → Module Complex (E i)
      inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
      inst✝¹ : (i : ι) → CStarModule A (E i)
      inst✝ : StarOrderedRing A
      x : WithCStarModule ((i : ι) → E i)
      ⊢ LE.le (Norm.norm x) (HMul.hMul (↑↑(Fintype.card ι)) (Norm.norm ((WithCStarMo …
    -/
    simp only [NNReal.coe_natCast, linearEquiv_apply]
    calc ‖x‖ ≤ ∑ i, ‖x i‖ := pi_norm_le_sum_norm x
      _ ≤ ∑ _, ‖⇑x‖ := Finset.sum_le_sum fun _ _ ↦ norm_le_pi_norm ..
      _ ≤ Fintype.card ι * ‖⇑x‖ := by simp


private lemma lipschitzWith_one_equiv_pi_aux : LipschitzWith 1 (equiv (Π i, E i)) :=
  AddMonoidHomClass.lipschitz_of_bound_nnnorm (linearEquiv ℂ (Π i, E i)) 1 <| by
    /-
      A : Type u_1
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      ι : Type u_2
      E : ι → Type u_3
      inst✝⁵ : Fintype ι
      inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
      inst✝³ : (i : ι) → Module Complex (E i)
      inst✝² : (i : ι) → SMul (MulOpposite A) (E i)
      inst✝¹ : (i : ι) → CStarModule A (E i)
      inst✝ : StarOrderedRing A
      ⊢ ∀ (x : WithCStarModule ((i : ι) → E i)), LE.le (NNNorm.nnnorm ((WithCStarMod …
    -/
    simpa using norm_equiv_le_norm_pi
    /-
      🎉 no goals
    -/


private lemma uniformity_pi_eq_aux :
    𝓤[(inferInstance : UniformSpace (Π i, E i)).comap <| equiv _] = 𝓤 (C⋆ᵐᵒᵈ (Π i, E i)) :=
  uniformity_eq_of_bilipschitz antilipschitzWith_card_equiv_pi_aux lipschitzWith_one_equiv_pi_aux


private lemma isBounded_pi_iff_aux (s : Set (C⋆ᵐᵒᵈ (Π i, E i))) :
    @IsBounded _ (induced <| equiv (Π i, E i)) s ↔ IsBounded s :=
  isBounded_iff_of_bilipschitz antilipschitzWith_card_equiv_pi_aux lipschitzWith_one_equiv_pi_aux s


noncomputable instance : NormedAddCommGroup (C⋆ᵐᵒᵈ (Π i, E i)) :=
  .ofCoreReplaceAll normedSpaceCore uniformity_pi_eq_aux isBounded_pi_iff_aux


instance : NormedSpace ℂ (C⋆ᵐᵒᵈ (Π i, E i)) := .ofCore normedSpaceCore


open scoped InnerProductSpace in
/-- Reinterpret an inner product space `E` over `ℂ` as a `CStarModule` over `ℂ`.

Note: this instance requires `SMul ℂᵐᵒᵖ E` and `IsCentralScalar ℂ E` instances to exist on `E`,
which is unlikely to occur in practice. However, in practice one could either add those instances
to the type `E` in question, or else supply them to this instance manually, which is reason behind
the naming of these two instance arguments. -/
instance instCStarModuleComplex : CStarModule ℂ E where
  inner x y := ⟪x, y⟫_ℂ
  inner_add_right := _root_.inner_add_right ..
  inner_self_nonneg {x} := by
    /-
      A : Type u_1
      inst✝³ : NonUnitalCStarAlgebra A
      inst✝² : PartialOrder A
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Complex E
      instSMulOp : SMul (MulOpposite Complex) E
      instCentral : IsCentralScalar Complex E
      x : E
      ⊢ LE.le 0 (Inner.inner x x)
    -/
    simp only
    /-
      A : Type u_1
      inst✝³ : NonUnitalCStarAlgebra A
      inst✝² : PartialOrder A
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Complex E
      instSMulOp : SMul (MulOpposite Complex) E
      instCentral : IsCentralScalar Complex E
      x : E
      ⊢ LE.le 0 (Inner.inner x x)
    -/
    rw [← inner_self_ofReal_re, RCLike.ofReal_nonneg]
    /-
      A : Type u_1
      inst✝³ : NonUnitalCStarAlgebra A
      inst✝² : PartialOrder A
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Complex E
      instSMulOp : SMul (MulOpposite Complex) E
      instCentral : IsCentralScalar Complex E
      x : E
      ⊢ LE.le 0 (RCLike.re (Inner.inner x x))
    -/
    exact inner_self_nonneg
    /-
      🎉 no goals
    -/
  inner_self := inner_self_eq_zero
                            /-
                              A : Type u_1
                              inst✝³ : NonUnitalCStarAlgebra A
                              inst✝² : PartialOrder A
                              E : Type u_2
                              inst✝¹ : NormedAddCommGroup E
                              inst✝ : InnerProductSpace Complex E
                              instSMulOp : SMul (MulOpposite Complex) E
                              instCentral : IsCentralScalar Complex E
                              ⊢ ∀ {a : Complex} {x y : E}, Eq (Inner.inner x (HSMul.hSMul (MulOpposite.op a) …
                            -/
  inner_op_smul_right := by simp [inner_smul_right, mul_comm]
                            /-
                              🎉 no goals
                            -/
  inner_smul_right_complex := inner_smul_right ..
  star_inner _ _ := inner_conj_symm ..
  norm_eq_sqrt_norm_inner_self {x} := by
    /-
      A : Type u_1
      inst✝³ : NonUnitalCStarAlgebra A
      inst✝² : PartialOrder A
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Complex E
      instSMulOp : SMul (MulOpposite Complex) E
      instCentral : IsCentralScalar Complex E
      x : E
      ⊢ Eq (Norm.norm x) (Norm.norm (Inner.inner x x)).sqrt
    -/
    simpa only [← inner_self_re_eq_norm] using norm_eq_sqrt_inner x
    /-
      🎉 no goals
    -/

-- Ensures that the two ways to obtain `CStarModule ℂ ℂ` are definitionally equal.

