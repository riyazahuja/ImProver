variable (𝕜) in
/-- The Lie bracket `[V, W] (x)` of two vector fields at a point, defined as
`DW(x) (V x) - DV(x) (W x)`. -/
def lieBracket (V W : E → E) (x : E) : E :=
  fderiv 𝕜 W x (V x) - fderiv 𝕜 V x (W x)


variable (𝕜) in
/-- The Lie bracket `[V, W] (x)` of two vector fields within a set at a point, defined as
`DW(x) (V x) - DV(x) (W x)` where the derivatives are taken inside `s`. -/
def lieBracketWithin (V W : E → E) (s : Set E) (x : E) : E :=
  fderivWithin 𝕜 W s x (V x) - fderivWithin 𝕜 V s x (W x)


lemma lieBracket_eq :
    lieBracket 𝕜 V W = fun x ↦ fderiv 𝕜 W x (V x) - fderiv 𝕜 V x (W x) := rfl


lemma lieBracketWithin_eq :
    lieBracketWithin 𝕜 V W s =
      fun x ↦ fderivWithin 𝕜 W s x (V x) - fderivWithin 𝕜 V s x (W x) := rfl


@[simp]
theorem lieBracketWithin_univ : lieBracketWithin 𝕜 V W univ = lieBracket 𝕜 V W := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V W Set.univ) (VectorField.lieBracket 𝕜 V …
  -/
  ext1 x
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    x : E
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V W Set.univ x) (VectorField.lieBracket 𝕜 …
  -/
  simp [lieBracketWithin, lieBracket]
  /-
    🎉 no goals
  -/


lemma lieBracketWithin_eq_zero_of_eq_zero (hV : V x = 0) (hW : W x = 0) :
    lieBracketWithin 𝕜 V W s x = 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s : Set E
    x : E
    hV : Eq (V x) 0
    hW : Eq (W x) 0
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V W s x) 0
  -/
  simp [lieBracketWithin, hV, hW]
  /-
    🎉 no goals
  -/


lemma lieBracket_eq_zero_of_eq_zero (hV : V x = 0) (hW : W x = 0) :
    lieBracket 𝕜 V W x = 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    x : E
    hV : Eq (V x) 0
    hW : Eq (W x) 0
    ⊢ Eq (VectorField.lieBracket 𝕜 V W x) 0
  -/
  simp [lieBracket, hV, hW]
  /-
    🎉 no goals
  -/


lemma lieBracketWithin_smul_left {c : 𝕜} (hV : DifferentiableWithinAt 𝕜 V s x)
    (hs : UniqueDiffWithinAt 𝕜 s x) :
    lieBracketWithin 𝕜 (c • V) W s x =
      c • lieBracketWithin 𝕜 V W s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s : Set E
    x : E
    c : 𝕜
    hV : DifferentiableWithinAt 𝕜 V s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 (HSMul.hSMul c V) W s x) (HSMul.hSMul c ( …
  -/
  simp only [lieBracketWithin, Pi.add_apply, map_add, Pi.smul_apply, map_smul, smul_sub]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s : Set E
    x : E
    c : 𝕜
    hV : DifferentiableWithinAt 𝕜 V s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (HSub.hSub (HSMul.hSMul c ((fderivWithin 𝕜 W s x) (V x))) ((fderivWithin  …
  -/
  rw [fderivWithin_const_smul' hs hV]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s : Set E
    x : E
    c : 𝕜
    hV : DifferentiableWithinAt 𝕜 V s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (HSub.hSub (HSMul.hSMul c ((fderivWithin 𝕜 W s x) (V x))) ((HSMul.hSMul c …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma lieBracket_smul_left {c : 𝕜} (hV : DifferentiableAt 𝕜 V x) :
    lieBracket 𝕜 (c • V) W x = c • lieBracket 𝕜 V W x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    x : E
    c : 𝕜
    hV : DifferentiableAt 𝕜 V x
    ⊢ Eq (VectorField.lieBracket 𝕜 (HSMul.hSMul c V) W x) (HSMul.hSMul c (VectorFi …
  -/
  simp only [← differentiableWithinAt_univ, ← lieBracketWithin_univ] at hV ⊢
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    x : E
    c : 𝕜
    hV : DifferentiableWithinAt 𝕜 V Set.univ x
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 (HSMul.hSMul c V) W Set.univ x) (HSMul.hS …
  -/
  exact lieBracketWithin_smul_left hV uniqueDiffWithinAt_univ
  /-
    🎉 no goals
  -/


lemma lieBracketWithin_smul_right {c : 𝕜} (hW : DifferentiableWithinAt 𝕜 W s x)
    (hs : UniqueDiffWithinAt 𝕜 s x) :
    lieBracketWithin 𝕜 V (c • W) s x =
      c • lieBracketWithin 𝕜 V W s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s : Set E
    x : E
    c : 𝕜
    hW : DifferentiableWithinAt 𝕜 W s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V (HSMul.hSMul c W) s x) (HSMul.hSMul c ( …
  -/
  simp only [lieBracketWithin, Pi.add_apply, map_add, Pi.smul_apply, map_smul, smul_sub]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s : Set E
    x : E
    c : 𝕜
    hW : DifferentiableWithinAt 𝕜 W s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (HSub.hSub ((fderivWithin 𝕜 (HSMul.hSMul c W) s x) (V x)) (HSMul.hSMul c  …
  -/
  rw [fderivWithin_const_smul' hs hW]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s : Set E
    x : E
    c : 𝕜
    hW : DifferentiableWithinAt 𝕜 W s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (HSub.hSub ((HSMul.hSMul c (fderivWithin 𝕜 W s x)) (V x)) (HSMul.hSMul c  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma lieBracket_smul_right {c : 𝕜} (hW : DifferentiableAt 𝕜 W x) :
    lieBracket 𝕜 V (c • W) x = c • lieBracket 𝕜 V W x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    x : E
    c : 𝕜
    hW : DifferentiableAt 𝕜 W x
    ⊢ Eq (VectorField.lieBracket 𝕜 V (HSMul.hSMul c W) x) (HSMul.hSMul c (VectorFi …
  -/
  simp only [← differentiableWithinAt_univ, ← lieBracketWithin_univ] at hW ⊢
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    x : E
    c : 𝕜
    hW : DifferentiableWithinAt 𝕜 W Set.univ x
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V (HSMul.hSMul c W) Set.univ x) (HSMul.hS …
  -/
  exact lieBracketWithin_smul_right hW uniqueDiffWithinAt_univ
  /-
    🎉 no goals
  -/


lemma lieBracketWithin_add_left (hV : DifferentiableWithinAt 𝕜 V s x)
    (hV₁ : DifferentiableWithinAt 𝕜 V₁ s x) (hs : UniqueDiffWithinAt 𝕜 s x) :
    lieBracketWithin 𝕜 (V + V₁) W s x =
      lieBracketWithin 𝕜 V W s x + lieBracketWithin 𝕜 V₁ W s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W V₁ : E → E
    s : Set E
    x : E
    hV : DifferentiableWithinAt 𝕜 V s x
    hV₁ : DifferentiableWithinAt 𝕜 V₁ s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 (HAdd.hAdd V V₁) W s x) (HAdd.hAdd (Vecto …
  -/
  simp only [lieBracketWithin, Pi.add_apply, map_add]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W V₁ : E → E
    s : Set E
    x : E
    hV : DifferentiableWithinAt 𝕜 V s x
    hV₁ : DifferentiableWithinAt 𝕜 V₁ s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (HSub.hSub (HAdd.hAdd ((fderivWithin 𝕜 W s x) (V x)) ((fderivWithin 𝕜 W s …
  -/
  rw [fderivWithin_add' hs hV hV₁, ContinuousLinearMap.add_apply]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W V₁ : E → E
    s : Set E
    x : E
    hV : DifferentiableWithinAt 𝕜 V s x
    hV₁ : DifferentiableWithinAt 𝕜 V₁ s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (HSub.hSub (HAdd.hAdd ((fderivWithin 𝕜 W s x) (V x)) ((fderivWithin 𝕜 W s …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


lemma lieBracket_add_left (hV : DifferentiableAt 𝕜 V x) (hV₁ : DifferentiableAt 𝕜 V₁ x) :
    lieBracket 𝕜 (V + V₁) W  x =
      lieBracket 𝕜 V W x + lieBracket 𝕜 V₁ W x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W V₁ : E → E
    x : E
    hV : DifferentiableAt 𝕜 V x
    hV₁ : DifferentiableAt 𝕜 V₁ x
    ⊢ Eq (VectorField.lieBracket 𝕜 (HAdd.hAdd V V₁) W x) (HAdd.hAdd (VectorField.l …
  -/
  simp only [lieBracket, Pi.add_apply, map_add]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W V₁ : E → E
    x : E
    hV : DifferentiableAt 𝕜 V x
    hV₁ : DifferentiableAt 𝕜 V₁ x
    ⊢ Eq (HSub.hSub (HAdd.hAdd ((fderiv 𝕜 W x) (V x)) ((fderiv 𝕜 W x) (V₁ x))) ((f …
  -/
  rw [fderiv_add' hV hV₁, ContinuousLinearMap.add_apply]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W V₁ : E → E
    x : E
    hV : DifferentiableAt 𝕜 V x
    hV₁ : DifferentiableAt 𝕜 V₁ x
    ⊢ Eq (HSub.hSub (HAdd.hAdd ((fderiv 𝕜 W x) (V x)) ((fderiv 𝕜 W x) (V₁ x))) (HA …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


lemma lieBracketWithin_add_right (hW : DifferentiableWithinAt 𝕜 W s x)
    (hW₁ : DifferentiableWithinAt 𝕜 W₁ s x) (hs :  UniqueDiffWithinAt 𝕜 s x) :
    lieBracketWithin 𝕜 V (W + W₁) s x =
      lieBracketWithin 𝕜 V W s x + lieBracketWithin 𝕜 V W₁ s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W W₁ : E → E
    s : Set E
    x : E
    hW : DifferentiableWithinAt 𝕜 W s x
    hW₁ : DifferentiableWithinAt 𝕜 W₁ s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V (HAdd.hAdd W W₁) s x) (HAdd.hAdd (Vecto …
  -/
  simp only [lieBracketWithin, Pi.add_apply, map_add]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W W₁ : E → E
    s : Set E
    x : E
    hW : DifferentiableWithinAt 𝕜 W s x
    hW₁ : DifferentiableWithinAt 𝕜 W₁ s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (HSub.hSub ((fderivWithin 𝕜 (HAdd.hAdd W W₁) s x) (V x)) (HAdd.hAdd ((fde …
  -/
  rw [fderivWithin_add' hs hW hW₁, ContinuousLinearMap.add_apply]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W W₁ : E → E
    s : Set E
    x : E
    hW : DifferentiableWithinAt 𝕜 W s x
    hW₁ : DifferentiableWithinAt 𝕜 W₁ s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (HSub.hSub (HAdd.hAdd ((fderivWithin 𝕜 W s x) (V x)) ((fderivWithin 𝕜 W₁  …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


lemma lieBracket_add_right (hW : DifferentiableAt 𝕜 W x) (hW₁ : DifferentiableAt 𝕜 W₁ x) :
    lieBracket 𝕜 V (W + W₁) x =
      lieBracket 𝕜 V W x + lieBracket 𝕜 V W₁ x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W W₁ : E → E
    x : E
    hW : DifferentiableAt 𝕜 W x
    hW₁ : DifferentiableAt 𝕜 W₁ x
    ⊢ Eq (VectorField.lieBracket 𝕜 V (HAdd.hAdd W W₁) x) (HAdd.hAdd (VectorField.l …
  -/
  simp only [lieBracket, Pi.add_apply, map_add]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W W₁ : E → E
    x : E
    hW : DifferentiableAt 𝕜 W x
    hW₁ : DifferentiableAt 𝕜 W₁ x
    ⊢ Eq (HSub.hSub ((fderiv 𝕜 (HAdd.hAdd W W₁) x) (V x)) (HAdd.hAdd ((fderiv 𝕜 V  …
  -/
  rw [fderiv_add' hW hW₁, ContinuousLinearMap.add_apply]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W W₁ : E → E
    x : E
    hW : DifferentiableAt 𝕜 W x
    hW₁ : DifferentiableAt 𝕜 W₁ x
    ⊢ Eq (HSub.hSub (HAdd.hAdd ((fderiv 𝕜 W x) (V x)) ((fderiv 𝕜 W₁ x) (V x))) (HA …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


lemma lieBracketWithin_swap : lieBracketWithin 𝕜 V W s = - lieBracketWithin 𝕜 W V s := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s : Set E
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V W s) (Neg.neg (VectorField.lieBracketWi …
  -/
  ext x; simp [lieBracketWithin]
         /-
           🎉 no goals
         -/


lemma lieBracket_swap : lieBracket 𝕜 V W x = - lieBracket 𝕜 W V x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    x : E
    ⊢ Eq (VectorField.lieBracket 𝕜 V W x) (Neg.neg (VectorField.lieBracket 𝕜 W V x))
  -/
  simp [lieBracket]
  /-
    🎉 no goals
  -/


@[simp] lemma lieBracketWithin_self : lieBracketWithin 𝕜 V V s = 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V : E → E
    s : Set E
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V V s) 0
  -/
  ext x; simp [lieBracketWithin]
         /-
           🎉 no goals
         -/


@[simp] lemma lieBracket_self : lieBracket 𝕜 V V = 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V : E → E
    ⊢ Eq (VectorField.lieBracket 𝕜 V V) 0
  -/
  ext x; simp [lieBracket]
         /-
           🎉 no goals
         -/


lemma _root_.ContDiffWithinAt.lieBracketWithin_vectorField
    {m n : WithTop ℕ∞} (hV : ContDiffWithinAt 𝕜 n V s x)
    (hW : ContDiffWithinAt 𝕜 n W s x) (hs : UniqueDiffOn 𝕜 s) (hmn : m + 1 ≤ n) (hx : x ∈ s) :
    ContDiffWithinAt 𝕜 m (lieBracketWithin 𝕜 V W s) s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s : Set E
    x : E
    m n : WithTop ENat
    hV : ContDiffWithinAt 𝕜 n V s x
    hW : ContDiffWithinAt 𝕜 n W s x
    hs : UniqueDiffOn 𝕜 s
    hmn : LE.le (HAdd.hAdd m 1) n
    hx : Membership.mem s x
    ⊢ ContDiffWithinAt 𝕜 m (VectorField.lieBracketWithin 𝕜 V W s) s x
  -/
  apply ContDiffWithinAt.sub
  · exact ContDiffWithinAt.clm_apply (hW.fderivWithin_right hs hmn hx)
      (hV.of_le (le_trans le_self_add hmn))
  · exact ContDiffWithinAt.clm_apply (hV.fderivWithin_right hs hmn hx)
      (hW.of_le (le_trans le_self_add hmn))


lemma _root_.ContDiffAt.lieBracket_vectorField {m n : WithTop ℕ∞} (hV : ContDiffAt 𝕜 n V x)
    (hW : ContDiffAt 𝕜 n W x) (hmn : m + 1 ≤ n) :
    ContDiffAt 𝕜 m (lieBracket 𝕜 V W) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    x : E
    m n : WithTop ENat
    hV : ContDiffAt 𝕜 n V x
    hW : ContDiffAt 𝕜 n W x
    hmn : LE.le (HAdd.hAdd m 1) n
    ⊢ ContDiffAt 𝕜 m (VectorField.lieBracket 𝕜 V W) x
  -/
  rw [← contDiffWithinAt_univ] at hV hW ⊢
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    x : E
    m n : WithTop ENat
    hV : ContDiffWithinAt 𝕜 n V Set.univ x
    hW : ContDiffWithinAt 𝕜 n W Set.univ x
    hmn : LE.le (HAdd.hAdd m 1) n
    ⊢ ContDiffWithinAt 𝕜 m (VectorField.lieBracket 𝕜 V W) Set.univ x
  -/
  simp_rw [← lieBracketWithin_univ]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    x : E
    m n : WithTop ENat
    hV : ContDiffWithinAt 𝕜 n V Set.univ x
    hW : ContDiffWithinAt 𝕜 n W Set.univ x
    hmn : LE.le (HAdd.hAdd m 1) n
    ⊢ ContDiffWithinAt 𝕜 m (VectorField.lieBracketWithin 𝕜 V W Set.univ) Set.univ x
  -/
  exact hV.lieBracketWithin_vectorField hW uniqueDiffOn_univ hmn (mem_univ _)
  /-
    🎉 no goals
  -/


lemma _root_.ContDiffOn.lieBracketWithin_vectorField {m n : WithTop ℕ∞} (hV : ContDiffOn 𝕜 n V s)
    (hW : ContDiffOn 𝕜 n W s) (hs : UniqueDiffOn 𝕜 s) (hmn : m + 1 ≤ n) :
    ContDiffOn 𝕜 m (lieBracketWithin 𝕜 V W s) s :=
  fun x hx ↦ (hV x hx).lieBracketWithin_vectorField (hW x hx) hs hmn hx


lemma _root_.ContDiff.lieBracket_vectorField {m n : WithTop ℕ∞} (hV : ContDiff 𝕜 n V)
    (hW : ContDiff 𝕜 n W) (hmn : m + 1 ≤ n) :
    ContDiff 𝕜 m (lieBracket 𝕜 V W) :=
  contDiff_iff_contDiffAt.2 (fun _ ↦ hV.contDiffAt.lieBracket_vectorField hW.contDiffAt hmn)


theorem lieBracketWithin_of_mem_nhdsWithin (st : t ∈ 𝓝[s] x) (hs : UniqueDiffWithinAt 𝕜 s x)
    (hV : DifferentiableWithinAt 𝕜 V t x) (hW : DifferentiableWithinAt 𝕜 W t x) :
    lieBracketWithin 𝕜 V W s x = lieBracketWithin 𝕜 V W t x := by
  simp [lieBracketWithin, fderivWithin_of_mem_nhdsWithin st hs hV,
    fderivWithin_of_mem_nhdsWithin st hs hW]


theorem lieBracketWithin_subset (st : s ⊆ t) (ht : UniqueDiffWithinAt 𝕜 s x)
    (hV : DifferentiableWithinAt 𝕜 V t x) (hW : DifferentiableWithinAt 𝕜 W t x) :
    lieBracketWithin 𝕜 V W s x = lieBracketWithin 𝕜 V W t x :=
  lieBracketWithin_of_mem_nhdsWithin (nhdsWithin_mono _ st self_mem_nhdsWithin) ht hV hW


theorem lieBracketWithin_inter (ht : t ∈ 𝓝 x) :
    lieBracketWithin 𝕜 V W (s ∩ t) x = lieBracketWithin 𝕜 V W s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s t : Set E
    x : E
    ht : Membership.mem (nhds x) t
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V W (Inter.inter s t) x) (VectorField.lie …
  -/
  simp [lieBracketWithin, fderivWithin_inter, ht]
  /-
    🎉 no goals
  -/


theorem lieBracketWithin_of_mem_nhds (h : s ∈ 𝓝 x) :
    lieBracketWithin 𝕜 V W s x = lieBracket 𝕜 V W x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s : Set E
    x : E
    h : Membership.mem (nhds x) s
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V W s x) (VectorField.lieBracket 𝕜 V W x)
  -/
  rw [← lieBracketWithin_univ, ← univ_inter s, lieBracketWithin_inter h]
  /-
    🎉 no goals
  -/


theorem lieBracketWithin_of_isOpen (hs : IsOpen s) (hx : x ∈ s) :
    lieBracketWithin 𝕜 V W s x = lieBracket 𝕜 V W x :=
  lieBracketWithin_of_mem_nhds (hs.mem_nhds hx)


theorem lieBracketWithin_eq_lieBracket (hs : UniqueDiffWithinAt 𝕜 s x)
    (hV : DifferentiableAt 𝕜 V x) (hW : DifferentiableAt 𝕜 W x) :
    lieBracketWithin 𝕜 V W s x = lieBracket 𝕜 V W x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s : Set E
    x : E
    hs : UniqueDiffWithinAt 𝕜 s x
    hV : DifferentiableAt 𝕜 V x
    hW : DifferentiableAt 𝕜 W x
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V W s x) (VectorField.lieBracket 𝕜 V W x)
  -/
  simp [lieBracketWithin, lieBracket, fderivWithin_eq_fderiv, hs, hV, hW]
  /-
    🎉 no goals
  -/


/-- Variant of `lieBracketWithin_congr_set` where one requires the sets to coincide only in
the complement of a point. -/
theorem lieBracketWithin_congr_set' (y : E) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    lieBracketWithin 𝕜 V W s x = lieBracketWithin 𝕜 V W t x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W : E → E
    s t : Set E
    x y : E
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V W s x) (VectorField.lieBracketWithin 𝕜  …
  -/
  simp [lieBracketWithin, fderivWithin_congr_set' _ h]
  /-
    🎉 no goals
  -/


theorem lieBracketWithin_congr_set (h : s =ᶠ[𝓝 x] t) :
    lieBracketWithin 𝕜 V W s x = lieBracketWithin 𝕜 V W t x :=
  lieBracketWithin_congr_set' x <| h.filter_mono inf_le_left


/-- Variant of `lieBracketWithin_eventually_congr_set` where one requires the sets to coincide only
in  the complement of a point. -/
theorem lieBracketWithin_eventually_congr_set' (y : E) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    lieBracketWithin 𝕜 V W s =ᶠ[𝓝 x] lieBracketWithin 𝕜 V W t :=
  (eventually_nhds_nhdsWithin.2 h).mono fun _ => lieBracketWithin_congr_set' y


theorem lieBracketWithin_eventually_congr_set (h : s =ᶠ[𝓝 x] t) :
    lieBracketWithin 𝕜 V W s =ᶠ[𝓝 x] lieBracketWithin 𝕜 V W t :=
  lieBracketWithin_eventually_congr_set' x <| h.filter_mono inf_le_left


theorem _root_.DifferentiableWithinAt.lieBracketWithin_congr_mono
    (hV : DifferentiableWithinAt 𝕜 V s x) (hVs : EqOn V₁ V t) (hVx : V₁ x = V x)
    (hW : DifferentiableWithinAt 𝕜 W s x) (hWs : EqOn W₁ W t) (hWx : W₁ x = W x)
    (hxt : UniqueDiffWithinAt 𝕜 t x) (h₁ : t ⊆ s) :
    lieBracketWithin 𝕜 V₁ W₁ t x = lieBracketWithin 𝕜 V W s x := by
  simp [lieBracketWithin, hV.fderivWithin_congr_mono, hW.fderivWithin_congr_mono, hVs, hVx,
    hWs, hWx, hxt, h₁]


theorem _root_.Filter.EventuallyEq.lieBracketWithin_vectorField_eq
    (hV : V₁ =ᶠ[𝓝[s] x] V) (hxV : V₁ x = V x) (hW : W₁ =ᶠ[𝓝[s] x] W) (hxW : W₁ x = W x) :
    lieBracketWithin 𝕜 V₁ W₁ s x = lieBracketWithin 𝕜 V W s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W V₁ W₁ : E → E
    s : Set E
    x : E
    hV : (nhdsWithin x s).EventuallyEq V₁ V
    hxV : Eq (V₁ x) (V x)
    hW : (nhdsWithin x s).EventuallyEq W₁ W
    hxW : Eq (W₁ x) (W x)
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V₁ W₁ s x) (VectorField.lieBracketWithin  …
  -/
  simp only [lieBracketWithin, hV.fderivWithin_eq hxV, hW.fderivWithin_eq hxW, hxV, hxW]
  /-
    🎉 no goals
  -/


theorem _root_.Filter.EventuallyEq.lieBracketWithin_vectorField_eq_of_mem
    (hV : V₁ =ᶠ[𝓝[s] x] V) (hW : W₁ =ᶠ[𝓝[s] x] W) (hx : x ∈ s) :
    lieBracketWithin 𝕜 V₁ W₁ s x = lieBracketWithin 𝕜 V W s x :=
  hV.lieBracketWithin_vectorField_eq (mem_of_mem_nhdsWithin hx hV :)
    hW (mem_of_mem_nhdsWithin hx hW :)


/-- If vector fields coincide on a neighborhood of a point within a set, then the Lie brackets
also coincide on a neighborhood of this point within this set. Version where one considers the Lie
bracket within a subset. -/
theorem _root_.Filter.EventuallyEq.lieBracketWithin_vectorField'
    (hV : V₁ =ᶠ[𝓝[s] x] V) (hW : W₁ =ᶠ[𝓝[s] x] W) (ht : t ⊆ s) :
    lieBracketWithin 𝕜 V₁ W₁ t =ᶠ[𝓝[s] x] lieBracketWithin 𝕜 V W t := by
  filter_upwards [hV.fderivWithin' ht (𝕜 := 𝕜), hW.fderivWithin' ht (𝕜 := 𝕜), hV, hW]
    with x hV' hW' hV hW
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W V₁ W₁ : E → E
    s t : Set E
    x✝ : E
    hV✝ : (nhdsWithin x✝ s).EventuallyEq V₁ V
    hW✝ : (nhdsWithin x✝ s).EventuallyEq W₁ W
    ht : HasSubset.Subset t s
    x : E
    hV' : Eq (fderivWithin 𝕜 V₁ t x) (fderivWithin 𝕜 V t x)
    hW' : Eq (fderivWithin 𝕜 W₁ t x) (fderivWithin 𝕜 W t x)
    hV : Eq (V₁ x) (V x)
    hW : Eq (W₁ x) (W x)
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 V₁ W₁ t x) (VectorField.lieBracketWithin  …
  -/
  simp [lieBracketWithin, hV', hW', hV, hW]
  /-
    🎉 no goals
  -/


protected theorem _root_.Filter.EventuallyEq.lieBracketWithin_vectorField
    (hV : V₁ =ᶠ[𝓝[s] x] V) (hW : W₁ =ᶠ[𝓝[s] x] W) :
    lieBracketWithin 𝕜 V₁ W₁ s =ᶠ[𝓝[s] x] lieBracketWithin 𝕜 V W s :=
  hV.lieBracketWithin_vectorField' hW Subset.rfl


protected theorem _root_.Filter.EventuallyEq.lieBracketWithin_vectorField_eq_of_insert
    (hV : V₁ =ᶠ[𝓝[insert x s] x] V) (hW : W₁ =ᶠ[𝓝[insert x s] x] W) :
    lieBracketWithin 𝕜 V₁ W₁ s x = lieBracketWithin 𝕜 V W s x := by
  apply mem_of_mem_nhdsWithin (mem_insert x s) (hV.lieBracketWithin_vectorField' hW
    (subset_insert x s))


theorem _root_.Filter.EventuallyEq.lieBracketWithin_vectorField_eq_nhds
    (hV : V₁ =ᶠ[𝓝 x] V) (hW : W₁ =ᶠ[𝓝 x] W) :
    lieBracketWithin 𝕜 V₁ W₁ s x = lieBracketWithin 𝕜 V W s x :=
  (hV.filter_mono nhdsWithin_le_nhds).lieBracketWithin_vectorField_eq hV.self_of_nhds
    (hW.filter_mono nhdsWithin_le_nhds) hW.self_of_nhds


theorem lieBracketWithin_congr
    (hV : EqOn V₁ V s) (hVx : V₁ x = V x) (hW : EqOn W₁ W s) (hWx : W₁ x = W x) :
    lieBracketWithin 𝕜 V₁ W₁ s x = lieBracketWithin 𝕜 V W s x :=
  (hV.eventuallyEq.filter_mono inf_le_right).lieBracketWithin_vectorField_eq hVx
    (hW.eventuallyEq.filter_mono inf_le_right) hWx


/-- Version of `lieBracketWithin_congr` in which one assumes that the point belongs to the
given set. -/
theorem lieBracketWithin_congr' (hV : EqOn V₁ V s) (hW : EqOn W₁ W s) (hx : x ∈ s) :
    lieBracketWithin 𝕜 V₁ W₁ s x = lieBracketWithin 𝕜 V W s x :=
  lieBracketWithin_congr hV (hV hx) hW (hW hx)


theorem _root_.Filter.EventuallyEq.lieBracket_vectorField_eq
    (hV : V₁ =ᶠ[𝓝 x] V) (hW : W₁ =ᶠ[𝓝 x] W) :
    lieBracket 𝕜 V₁ W₁ x = lieBracket 𝕜 V W x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W V₁ W₁ : E → E
    x : E
    hV : (nhds x).EventuallyEq V₁ V
    hW : (nhds x).EventuallyEq W₁ W
    ⊢ Eq (VectorField.lieBracket 𝕜 V₁ W₁ x) (VectorField.lieBracket 𝕜 V W x)
  -/
  rw [← lieBracketWithin_univ, ← lieBracketWithin_univ, hV.lieBracketWithin_vectorField_eq_nhds hW]
  /-
    🎉 no goals
  -/


protected theorem _root_.Filter.EventuallyEq.lieBracket_vectorField
    (hV : V₁ =ᶠ[𝓝 x] V) (hW : W₁ =ᶠ[𝓝 x] W) : lieBracket 𝕜 V₁ W₁ =ᶠ[𝓝 x] lieBracket 𝕜 V W := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W V₁ W₁ : E → E
    x : E
    hV : (nhds x).EventuallyEq V₁ V
    hW : (nhds x).EventuallyEq W₁ W
    ⊢ (nhds x).EventuallyEq (VectorField.lieBracket 𝕜 V₁ W₁) (VectorField.lieBrack …
  -/
  filter_upwards [hV.eventuallyEq_nhds, hW.eventuallyEq_nhds] with y hVy hWy
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V W V₁ W₁ : E → E
    x : E
    hV : (nhds x).EventuallyEq V₁ V
    hW : (nhds x).EventuallyEq W₁ W
    y : E
    hVy : (nhds y).EventuallyEq V₁ V
    hWy : (nhds y).EventuallyEq W₁ W
    ⊢ Eq (VectorField.lieBracket 𝕜 V₁ W₁ y) (VectorField.lieBracket 𝕜 V W y)
  -/
  exact hVy.lieBracket_vectorField_eq hWy
  /-
    🎉 no goals
  -/


/-- The Lie bracket of vector fields in vector spaces satisfies the Leibniz identity
`[U, [V, W]] = [[U, V], W] + [V, [U, W]]`. -/
lemma leibniz_identity_lieBracketWithin_of_isSymmSndFDerivWithinAt
    {U V W : E → E} {s : Set E} {x : E} (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s)
    (hU : ContDiffWithinAt 𝕜 2 U s x) (hV : ContDiffWithinAt 𝕜 2 V s x)
    (hW : ContDiffWithinAt 𝕜 2 W s x)
    (h'U : IsSymmSndFDerivWithinAt 𝕜 U s x) (h'V : IsSymmSndFDerivWithinAt 𝕜 V s x)
    (h'W : IsSymmSndFDerivWithinAt 𝕜 W s x) :
    lieBracketWithin 𝕜 U (lieBracketWithin 𝕜 V W s) s x =
      lieBracketWithin 𝕜 (lieBracketWithin 𝕜 U V s) W s x
      + lieBracketWithin 𝕜 V (lieBracketWithin 𝕜 U W s) s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    U V W : E → E
    s : Set E
    x : E
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hU : ContDiffWithinAt 𝕜 2 U s x
    hV : ContDiffWithinAt 𝕜 2 V s x
    hW : ContDiffWithinAt 𝕜 2 W s x
    h'U : IsSymmSndFDerivWithinAt 𝕜 U s x
    h'V : IsSymmSndFDerivWithinAt 𝕜 V s x
    h'W : IsSymmSndFDerivWithinAt 𝕜 W s x
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 U (VectorField.lieBracketWithin 𝕜 V W s)  …
  -/
  simp only [lieBracketWithin_eq, map_sub]
  have aux₁ {U V : E → E} (hU : ContDiffWithinAt 𝕜 2 U s x) (hV : ContDiffWithinAt 𝕜 2 V s x) :
      DifferentiableWithinAt 𝕜 (fun x ↦ (fderivWithin 𝕜 V s x) (U x)) s x :=
    have := hV.fderivWithin_right_apply (hU.of_le one_le_two) hs le_rfl hx
    this.differentiableWithinAt le_rfl
  have aux₂ {U V : E → E} (hU : ContDiffWithinAt 𝕜 2 U s x) (hV : ContDiffWithinAt 𝕜 2 V s x) :
      fderivWithin 𝕜 (fun y ↦ (fderivWithin 𝕜 U s y) (V y)) s x =
        (fderivWithin 𝕜 U s x).comp (fderivWithin 𝕜 V s x) +
        (fderivWithin 𝕜 (fderivWithin 𝕜 U s) s x).flip (V x) := by
    refine fderivWithin_clm_apply (hs x hx) ?_ (hV.differentiableWithinAt one_le_two)
    exact (hU.fderivWithin_right hs le_rfl hx).differentiableWithinAt le_rfl
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    U V W : E → E
    s : Set E
    x : E
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hU : ContDiffWithinAt 𝕜 2 U s x
    hV : ContDiffWithinAt 𝕜 2 V s x
    hW : ContDiffWithinAt 𝕜 2 W s x
    h'U : IsSymmSndFDerivWithinAt 𝕜 U s x
    h'V : IsSymmSndFDerivWithinAt 𝕜 V s x
    h'W : IsSymmSndFDerivWithinAt 𝕜 W s x
    aux₁ : ∀ {U V : E → E}, ContDiffWithinAt 𝕜 2 U s x → ContDiffWithinAt 𝕜 2 V s  …
    aux₂ : ∀ {U V : E → E}, ContDiffWithinAt 𝕜 2 U s x → ContDiffWithinAt 𝕜 2 V s  …
    ⊢ Eq (HSub.hSub ((fderivWithin 𝕜 (fun x => HSub.hSub ((fderivWithin 𝕜 W s x) ( …
  -/
  rw [fderivWithin_sub (hs x hx) (aux₁ hV hW) (aux₁ hW hV)]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    U V W : E → E
    s : Set E
    x : E
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hU : ContDiffWithinAt 𝕜 2 U s x
    hV : ContDiffWithinAt 𝕜 2 V s x
    hW : ContDiffWithinAt 𝕜 2 W s x
    h'U : IsSymmSndFDerivWithinAt 𝕜 U s x
    h'V : IsSymmSndFDerivWithinAt 𝕜 V s x
    h'W : IsSymmSndFDerivWithinAt 𝕜 W s x
    aux₁ : ∀ {U V : E → E}, ContDiffWithinAt 𝕜 2 U s x → ContDiffWithinAt 𝕜 2 V s  …
    aux₂ : ∀ {U V : E → E}, ContDiffWithinAt 𝕜 2 U s x → ContDiffWithinAt 𝕜 2 V s  …
    ⊢ Eq (HSub.hSub ((HSub.hSub (fderivWithin 𝕜 (fun x => (fderivWithin 𝕜 W s x) ( …
  -/
  rw [fderivWithin_sub (hs x hx) (aux₁ hU hV) (aux₁ hV hU)]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    U V W : E → E
    s : Set E
    x : E
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hU : ContDiffWithinAt 𝕜 2 U s x
    hV : ContDiffWithinAt 𝕜 2 V s x
    hW : ContDiffWithinAt 𝕜 2 W s x
    h'U : IsSymmSndFDerivWithinAt 𝕜 U s x
    h'V : IsSymmSndFDerivWithinAt 𝕜 V s x
    h'W : IsSymmSndFDerivWithinAt 𝕜 W s x
    aux₁ : ∀ {U V : E → E}, ContDiffWithinAt 𝕜 2 U s x → ContDiffWithinAt 𝕜 2 V s  …
    aux₂ : ∀ {U V : E → E}, ContDiffWithinAt 𝕜 2 U s x → ContDiffWithinAt 𝕜 2 V s  …
    ⊢ Eq (HSub.hSub ((HSub.hSub (fderivWithin 𝕜 (fun x => (fderivWithin 𝕜 W s x) ( …
  -/
  rw [fderivWithin_sub (hs x hx) (aux₁ hU hW) (aux₁ hW hU)]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    U V W : E → E
    s : Set E
    x : E
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hU : ContDiffWithinAt 𝕜 2 U s x
    hV : ContDiffWithinAt 𝕜 2 V s x
    hW : ContDiffWithinAt 𝕜 2 W s x
    h'U : IsSymmSndFDerivWithinAt 𝕜 U s x
    h'V : IsSymmSndFDerivWithinAt 𝕜 V s x
    h'W : IsSymmSndFDerivWithinAt 𝕜 W s x
    aux₁ : ∀ {U V : E → E}, ContDiffWithinAt 𝕜 2 U s x → ContDiffWithinAt 𝕜 2 V s  …
    aux₂ : ∀ {U V : E → E}, ContDiffWithinAt 𝕜 2 U s x → ContDiffWithinAt 𝕜 2 V s  …
    ⊢ Eq (HSub.hSub ((HSub.hSub (fderivWithin 𝕜 (fun x => (fderivWithin 𝕜 W s x) ( …
  -/
  rw [aux₂ hW hV, aux₂ hV hW, aux₂ hV hU, aux₂ hU hV, aux₂ hW hU, aux₂ hU hW]
  simp only [ContinuousLinearMap.coe_sub', Pi.sub_apply, ContinuousLinearMap.add_apply,
    ContinuousLinearMap.coe_comp', Function.comp_apply, ContinuousLinearMap.flip_apply, h'V.eq,
    h'U.eq, h'W.eq]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    U V W : E → E
    s : Set E
    x : E
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hU : ContDiffWithinAt 𝕜 2 U s x
    hV : ContDiffWithinAt 𝕜 2 V s x
    hW : ContDiffWithinAt 𝕜 2 W s x
    h'U : IsSymmSndFDerivWithinAt 𝕜 U s x
    h'V : IsSymmSndFDerivWithinAt 𝕜 V s x
    h'W : IsSymmSndFDerivWithinAt 𝕜 W s x
    aux₁ : ∀ {U V : E → E}, ContDiffWithinAt 𝕜 2 U s x → ContDiffWithinAt 𝕜 2 V s  …
    aux₂ : ∀ {U V : E → E}, ContDiffWithinAt 𝕜 2 U s x → ContDiffWithinAt 𝕜 2 V s  …
    ⊢ Eq (HSub.hSub (HSub.hSub (HAdd.hAdd ((fderivWithin 𝕜 W s x) ((fderivWithin 𝕜 …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


/-- The Lie bracket of vector fields in vector spaces satisfies the Leibniz identity
`[U, [V, W]] = [[U, V], W] + [V, [U, W]]`. -/
lemma leibniz_identity_lieBracketWithin (hn : minSmoothness 𝕜 2 ≤ n)
    {U V W : E → E} {s : Set E} {x : E}
    (hs : UniqueDiffOn 𝕜 s) (h'x : x ∈ closure (interior s)) (hx : x ∈ s)
    (hU : ContDiffWithinAt 𝕜 n U s x) (hV : ContDiffWithinAt 𝕜 n V s x)
    (hW : ContDiffWithinAt 𝕜 n W s x) :
    lieBracketWithin 𝕜 U (lieBracketWithin 𝕜 V W s) s x =
      lieBracketWithin 𝕜 (lieBracketWithin 𝕜 U V s) W s x
      + lieBracketWithin 𝕜 V (lieBracketWithin 𝕜 U W s) s x := by
  apply leibniz_identity_lieBracketWithin_of_isSymmSndFDerivWithinAt hs hx
    (hU.of_le (le_minSmoothness.trans hn)) (hV.of_le (le_minSmoothness.trans hn))
    (hW.of_le (le_minSmoothness.trans hn))
    /-
      case h'U
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      n : WithTop ENat
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      hn : LE.le (minSmoothness 𝕜 2) n
      U V W : E → E
      s : Set E
      x : E
      hs : UniqueDiffOn 𝕜 s
      h'x : Membership.mem (closure (interior s)) x
      hx : Membership.mem s x
      hU : ContDiffWithinAt 𝕜 n U s x
      hV : ContDiffWithinAt 𝕜 n V s x
      hW : ContDiffWithinAt 𝕜 n W s x
      ⊢ IsSymmSndFDerivWithinAt 𝕜 U s x
    -/
  · exact hU.isSymmSndFDerivWithinAt hn hs h'x hx
    /-
      🎉 no goals
    -/
    /-
      case h'V
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      n : WithTop ENat
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      hn : LE.le (minSmoothness 𝕜 2) n
      U V W : E → E
      s : Set E
      x : E
      hs : UniqueDiffOn 𝕜 s
      h'x : Membership.mem (closure (interior s)) x
      hx : Membership.mem s x
      hU : ContDiffWithinAt 𝕜 n U s x
      hV : ContDiffWithinAt 𝕜 n V s x
      hW : ContDiffWithinAt 𝕜 n W s x
      ⊢ IsSymmSndFDerivWithinAt 𝕜 V s x
    -/
  · exact hV.isSymmSndFDerivWithinAt hn hs h'x hx
    /-
      🎉 no goals
    -/
    /-
      case h'W
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      n : WithTop ENat
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      hn : LE.le (minSmoothness 𝕜 2) n
      U V W : E → E
      s : Set E
      x : E
      hs : UniqueDiffOn 𝕜 s
      h'x : Membership.mem (closure (interior s)) x
      hx : Membership.mem s x
      hU : ContDiffWithinAt 𝕜 n U s x
      hV : ContDiffWithinAt 𝕜 n V s x
      hW : ContDiffWithinAt 𝕜 n W s x
      ⊢ IsSymmSndFDerivWithinAt 𝕜 W s x
    -/
  · exact hW.isSymmSndFDerivWithinAt hn hs h'x hx
    /-
      🎉 no goals
    -/


/-- The Lie bracket of vector fields in vector spaces satisfies the Leibniz identity
`[U, [V, W]] = [[U, V], W] + [V, [U, W]]`. -/
lemma leibniz_identity_lieBracket (hn : minSmoothness 𝕜 2 ≤ n) {U V W : E → E} {x : E}
    (hU : ContDiffAt 𝕜 n U x) (hV : ContDiffAt 𝕜 n V x) (hW : ContDiffAt 𝕜 n W x) :
    lieBracket 𝕜 U (lieBracket 𝕜 V W) x =
      lieBracket 𝕜 (lieBracket 𝕜 U V) W x + lieBracket 𝕜 V (lieBracket 𝕜 U W) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    n : WithTop ENat
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    hn : LE.le (minSmoothness 𝕜 2) n
    U V W : E → E
    x : E
    hU : ContDiffAt 𝕜 n U x
    hV : ContDiffAt 𝕜 n V x
    hW : ContDiffAt 𝕜 n W x
    ⊢ Eq (VectorField.lieBracket 𝕜 U (VectorField.lieBracket 𝕜 V W) x) (HAdd.hAdd  …
  -/
  simp only [← lieBracketWithin_univ, ← contDiffWithinAt_univ] at hU hV hW ⊢
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    n : WithTop ENat
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    hn : LE.le (minSmoothness 𝕜 2) n
    U V W : E → E
    x : E
    hU : ContDiffWithinAt 𝕜 n U Set.univ x
    hV : ContDiffWithinAt 𝕜 n V Set.univ x
    hW : ContDiffWithinAt 𝕜 n W Set.univ x
    ⊢ Eq (VectorField.lieBracketWithin 𝕜 U (VectorField.lieBracketWithin 𝕜 V W Set …
  -/
  exact leibniz_identity_lieBracketWithin hn uniqueDiffOn_univ (by simp) (mem_univ _) hU hV hW
  /-
    🎉 no goals
  -/



variable (𝕜) in
/-- The pullback of a vector field under a function, defined
as `(f^* V) (x) = Df(x)^{-1} (V (f x))`. If `Df(x)` is not invertible, we use the junk value `0`.
-/
def pullback (f : E → F) (V : F → F) (x : E) : E := (fderiv 𝕜 f x).inverse (V (f x))


variable (𝕜) in
/-- The pullback within a set of a vector field under a function, defined
as `(f^* V) (x) = Df(x)^{-1} (V (f x))` where `Df(x)` is the derivative of `f` within `s`.
If `Df(x)` is not invertible, we use the junk value `0`.
-/
def pullbackWithin (f : E → F) (V : F → F) (s : Set E) (x : E) : E :=
  (fderivWithin 𝕜 f s x).inverse (V (f x))


lemma pullbackWithin_eq {f : E → F} {V : F → F} {s : Set E} :
    pullbackWithin 𝕜 f V s = fun x ↦ (fderivWithin 𝕜 f s x).inverse (V (f x)) := rfl


lemma pullback_eq_of_fderiv_eq
    {f : E → F} {M : E ≃L[𝕜] F} {x : E} (hf : M = fderiv 𝕜 f x) (V : F → F) :
    pullback 𝕜 f V x = M.symm (V (f x)) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    M : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    hf : Eq (↑M) (fderiv 𝕜 f x)
    V : F → F
    ⊢ Eq (VectorField.pullback 𝕜 f V x) (M.symm (V (f x)))
  -/
  simp [pullback, ← hf]
  /-
    🎉 no goals
  -/


lemma pullback_eq_of_not_isInvertible {f : E → F} {x : E}
    (h : ¬(fderiv 𝕜 f x).IsInvertible) (V : F → F) :
    pullback 𝕜 f V x = 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : Not (fderiv 𝕜 f x).IsInvertible
    V : F → F
    ⊢ Eq (VectorField.pullback 𝕜 f V x) 0
  -/
  simp [pullback, h]
  /-
    🎉 no goals
  -/


lemma pullbackWithin_eq_of_not_isInvertible {f : E → F} {x : E}
    (h : ¬(fderivWithin 𝕜 f s x).IsInvertible) (V : F → F) :
    pullbackWithin 𝕜 f V s x = 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    h : Not (fderivWithin 𝕜 f s x).IsInvertible
    V : F → F
    ⊢ Eq (VectorField.pullbackWithin 𝕜 f V s x) 0
  -/
  simp [pullbackWithin, h]
  /-
    🎉 no goals
  -/


lemma pullbackWithin_eq_of_fderivWithin_eq
    {f : E → F} {M : E ≃L[𝕜] F} {x : E} (hf : M = fderivWithin 𝕜 f s x) (V : F → F) :
    pullbackWithin 𝕜 f V s x = M.symm (V (f x)) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    M : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    hf : Eq (↑M) (fderivWithin 𝕜 f s x)
    V : F → F
    ⊢ Eq (VectorField.pullbackWithin 𝕜 f V s x) (M.symm (V (f x)))
  -/
  simp [pullbackWithin, ← hf]
  /-
    🎉 no goals
  -/


@[simp] lemma pullbackWithin_univ {f : E → F} {V : F → F} :
    pullbackWithin 𝕜 f V univ = pullback 𝕜 f V := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    V : F → F
    ⊢ Eq (VectorField.pullbackWithin 𝕜 f V Set.univ) (VectorField.pullback 𝕜 f V)
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    V : F → F
    x : E
    ⊢ Eq (VectorField.pullbackWithin 𝕜 f V Set.univ x) (VectorField.pullback 𝕜 f V …
  -/
  simp [pullbackWithin, pullback]
  /-
    🎉 no goals
  -/


lemma fderiv_pullback (f : E → F) (V : F → F) (x : E) (h'f : (fderiv 𝕜 f x).IsInvertible) :
    fderiv 𝕜 f x (pullback 𝕜 f V x) = V (f x) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    V : F → F
    x : E
    h'f : (fderiv 𝕜 f x).IsInvertible
    ⊢ Eq ((fderiv 𝕜 f x) (VectorField.pullback 𝕜 f V x)) (V (f x))
  -/
  rcases h'f with ⟨M, hM⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    V : F → F
    x : E
    M : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    hM : Eq (↑M) (fderiv 𝕜 f x)
    ⊢ Eq ((fderiv 𝕜 f x) (VectorField.pullback 𝕜 f V x)) (V (f x))
  -/
  simp [pullback_eq_of_fderiv_eq hM, ← hM]
  /-
    🎉 no goals
  -/


lemma fderivWithin_pullbackWithin {f : E → F} {V : F → F} {x : E}
    (h'f : (fderivWithin 𝕜 f s x).IsInvertible) :
    fderivWithin 𝕜 f s x (pullbackWithin 𝕜 f V s x) = V (f x) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    V : F → F
    x : E
    h'f : (fderivWithin 𝕜 f s x).IsInvertible
    ⊢ Eq ((fderivWithin 𝕜 f s x) (VectorField.pullbackWithin 𝕜 f V s x)) (V (f x))
  -/
  rcases h'f with ⟨M, hM⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    V : F → F
    x : E
    M : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    hM : Eq (↑M) (fderivWithin 𝕜 f s x)
    ⊢ Eq ((fderivWithin 𝕜 f s x) (VectorField.pullbackWithin 𝕜 f V s x)) (V (f x))
  -/
  simp [pullbackWithin_eq_of_fderivWithin_eq hM, ← hM]
  /-
    🎉 no goals
  -/


/-- If a `C^2` map has an invertible derivative within a set at a point, then nearby derivatives
can be written as continuous linear equivs, which depend in a `C^1` way on the point, as well as
their inverse, and moreover one can compute the derivative of the inverse. -/
lemma _root_.exists_continuousLinearEquiv_fderivWithin_symm_eq
    {f : E → F} {s : Set E} {x : E} (h'f : ContDiffWithinAt 𝕜 2 f s x)
    (hf : (fderivWithin 𝕜 f s x).IsInvertible) (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) :
    ∃ N : E → (E ≃L[𝕜] F), ContDiffWithinAt 𝕜 1 (fun y ↦ (N y : E →L[𝕜] F)) s x
    ∧ ContDiffWithinAt 𝕜 1 (fun y ↦ ((N y).symm : F →L[𝕜] E)) s x
    ∧ (∀ᶠ y in 𝓝[s] x, N y = fderivWithin 𝕜 f s y)
    ∧ ∀ v, fderivWithin 𝕜 (fun y ↦ ((N y).symm : F →L[𝕜] E)) s x v
      = - (N x).symm  ∘L ((fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x v)) ∘L (N x).symm := by
  classical
  rcases hf with ⟨M, hM⟩
  let U := {y | ∃ (N : E ≃L[𝕜] F), N = fderivWithin 𝕜 f s y}
  have hU : U ∈ 𝓝[s] x := by
    have I : range ((↑) : (E ≃L[𝕜] F) → E →L[𝕜] F) ∈ 𝓝 (fderivWithin 𝕜 f s x) := by
      rw [← hM]
      exact M.nhds
    have : ContinuousWithinAt (fderivWithin 𝕜 f s) s x :=
      (h'f.fderivWithin_right (m := 1) hs le_rfl hx).continuousWithinAt
    exact this I
  let N : E → (E ≃L[𝕜] F) := fun x ↦ if h : x ∈ U then h.choose else M
  have eN : (fun y ↦ (N y : E →L[𝕜] F)) =ᶠ[𝓝[s] x] fun y ↦ fderivWithin 𝕜 f s y := by
    filter_upwards [hU] with y hy
    simpa only [hy, ↓reduceDIte, N] using Exists.choose_spec hy
  have e'N : N x = fderivWithin 𝕜 f s x := by apply mem_of_mem_nhdsWithin hx eN
  have hN : ContDiffWithinAt 𝕜 1 (fun y ↦ (N y : E →L[𝕜] F)) s x := by
    have : ContDiffWithinAt 𝕜 1 (fun y ↦ fderivWithin 𝕜 f s y) s x :=
      h'f.fderivWithin_right (m := 1) hs le_rfl hx
    apply this.congr_of_eventuallyEq eN e'N
  have hN' : ContDiffWithinAt 𝕜 1 (fun y ↦ ((N y).symm : F →L[𝕜] E)) s x := by
    have : ContDiffWithinAt 𝕜 1 (ContinuousLinearMap.inverse ∘ (fun y ↦ (N y : E →L[𝕜] F))) s x :=
      (contDiffAt_map_inverse (N x)).comp_contDiffWithinAt x hN
    convert this with y
    simp only [Function.comp_apply, ContinuousLinearMap.inverse_equiv]
  refine ⟨N, hN, hN', eN, fun v ↦ ?_⟩
  have A' y : ContinuousLinearMap.compL 𝕜 F E F (N y : E →L[𝕜] F) ((N y).symm : F →L[𝕜] E)
      = ContinuousLinearMap.id 𝕜 F := by ext; simp
  have : fderivWithin 𝕜 (fun y ↦ ContinuousLinearMap.compL 𝕜 F E F (N y : E →L[𝕜] F)
      ((N y).symm : F →L[𝕜] E)) s x v = 0 := by
    simp [A', fderivWithin_const_apply, hs x hx]
  have I : (N x : E →L[𝕜] F) ∘L (fderivWithin 𝕜 (fun y ↦ ((N y).symm : F →L[𝕜] E)) s x v) =
      - (fderivWithin 𝕜 (fun y ↦ (N y : E →L[𝕜] F)) s x v) ∘L ((N x).symm : F →L[𝕜] E) := by
    rw [ContinuousLinearMap.fderivWithin_of_bilinear _ (hN.differentiableWithinAt le_rfl)
      (hN'.differentiableWithinAt le_rfl) (hs x hx)] at this
    simpa [eq_neg_iff_add_eq_zero] using this
  have B (M : F →L[𝕜] E) : M = ((N x).symm : F →L[𝕜] E) ∘L ((N x) ∘L M) := by
    ext; simp
  rw [B (fderivWithin 𝕜 (fun y ↦ ((N y).symm : F →L[𝕜] E)) s x v), I]
  simp only [ContinuousLinearMap.comp_neg, neg_inj, eN.fderivWithin_eq e'N]


lemma DifferentiableWithinAt.pullbackWithin {f : E → F} {V : F → F} {s : Set E} {t : Set F} {x : E}
    (hV : DifferentiableWithinAt 𝕜 V t (f x))
    (hf : ContDiffWithinAt 𝕜 2 f s x) (hf' : (fderivWithin 𝕜 f s x).IsInvertible)
    (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) (hst : MapsTo f s t) :
    DifferentiableWithinAt 𝕜 (pullbackWithin 𝕜 f V s) s x := by
  rcases exists_continuousLinearEquiv_fderivWithin_symm_eq hf hf' hs hx
    with ⟨M, -, M_symm_smooth, hM, -⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace E
    f : E → F
    V : F → F
    s : Set E
    t : Set F
    x : E
    hV : DifferentiableWithinAt 𝕜 V t (f x)
    hf : ContDiffWithinAt 𝕜 2 f s x
    hf' : (fderivWithin 𝕜 f s x).IsInvertible
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hst : Set.MapsTo f s t
    M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
    M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
    hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
    ⊢ DifferentiableWithinAt 𝕜 (VectorField.pullbackWithin 𝕜 f V s) s x
  -/
  simp only [pullbackWithin_eq]
  have : DifferentiableWithinAt 𝕜 (fun y ↦ ((M y).symm : F →L[𝕜] E) (V (f y))) s x := by
    apply DifferentiableWithinAt.clm_apply
    · exact M_symm_smooth.differentiableWithinAt le_rfl
    · exact hV.comp _ (hf.differentiableWithinAt one_le_two) hst
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace E
    f : E → F
    V : F → F
    s : Set E
    t : Set F
    x : E
    hV : DifferentiableWithinAt 𝕜 V t (f x)
    hf : ContDiffWithinAt 𝕜 2 f s x
    hf' : (fderivWithin 𝕜 f s x).IsInvertible
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hst : Set.MapsTo f s t
    M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
    M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
    hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
    this : DifferentiableWithinAt 𝕜 (fun y => ↑(M y).symm (V (f y))) s x
    ⊢ DifferentiableWithinAt 𝕜 (fun x => (fderivWithin 𝕜 f s x).inverse (V (f x))) …
  -/
  apply this.congr_of_eventuallyEq
    /-
      case intro.intro.intro.intro.h₁
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace E
      f : E → F
      V : F → F
      s : Set E
      t : Set F
      x : E
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hf : ContDiffWithinAt 𝕜 2 f s x
      hf' : (fderivWithin 𝕜 f s x).IsInvertible
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      hst : Set.MapsTo f s t
      M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
      M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
      hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
      this : DifferentiableWithinAt 𝕜 (fun y => ↑(M y).symm (V (f y))) s x
      ⊢ (nhdsWithin x s).EventuallyEq (fun x => (fderivWithin 𝕜 f s x).inverse (V (f …
    -/
  · filter_upwards [hM] with y hy using by simp [← hy]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.hx
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace E
      f : E → F
      V : F → F
      s : Set E
      t : Set F
      x : E
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hf : ContDiffWithinAt 𝕜 2 f s x
      hf' : (fderivWithin 𝕜 f s x).IsInvertible
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      hst : Set.MapsTo f s t
      M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
      M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
      hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
      this : DifferentiableWithinAt 𝕜 (fun y => ↑(M y).symm (V (f y))) s x
      ⊢ Eq ((fderivWithin 𝕜 f s x).inverse (V (f x))) (↑(M x).symm (V (f x)))
    -/
  · have hMx : M x = fderivWithin 𝕜 f s x := by apply mem_of_mem_nhdsWithin hx hM
    /-
      case intro.intro.intro.intro.hx
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace E
      f : E → F
      V : F → F
      s : Set E
      t : Set F
      x : E
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hf : ContDiffWithinAt 𝕜 2 f s x
      hf' : (fderivWithin 𝕜 f s x).IsInvertible
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      hst : Set.MapsTo f s t
      M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
      M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
      hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
      this : DifferentiableWithinAt 𝕜 (fun y => ↑(M y).symm (V (f y))) s x
      hMx : Eq (↑(M x)) (fderivWithin 𝕜 f s x)
      ⊢ Eq ((fderivWithin 𝕜 f s x).inverse (V (f x))) (↑(M x).symm (V (f x)))
    -/
    simp [← hMx]
    /-
      🎉 no goals
    -/


/-- If a `C^2` map has an invertible derivative at a point, then nearby derivatives can be written
as continuous linear equivs, which depend in a `C^1` way on the point, as well as their inverse, and
moreover one can compute the derivative of the inverse. -/
lemma _root_.exists_continuousLinearEquiv_fderiv_symm_eq
    {f : E → F} {x : E} (h'f : ContDiffAt 𝕜 2 f x) (hf : (fderiv 𝕜 f x).IsInvertible) :
    ∃ N : E → (E ≃L[𝕜] F), ContDiffAt 𝕜 1 (fun y ↦ (N y : E →L[𝕜] F)) x
    ∧ ContDiffAt 𝕜 1 (fun y ↦ ((N y).symm : F →L[𝕜] E)) x
    ∧ (∀ᶠ y in 𝓝 x, N y = fderiv 𝕜 f y)
    ∧ ∀ v, fderiv 𝕜 (fun y ↦ ((N y).symm : F →L[𝕜] E)) x v
      = - (N x).symm  ∘L ((fderiv 𝕜 (fderiv 𝕜 f) x v)) ∘L (N x).symm := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace E
    f : E → F
    x : E
    h'f : ContDiffAt 𝕜 2 f x
    hf : (fderiv 𝕜 f x).IsInvertible
    ⊢ Exists fun N => And (ContDiffAt 𝕜 1 (fun y => ↑(N y)) x) (And (ContDiffAt 𝕜  …
  -/
  simp only [← fderivWithin_univ, ← contDiffWithinAt_univ, ← nhdsWithin_univ] at hf h'f ⊢
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace E
    f : E → F
    x : E
    hf : (fderivWithin 𝕜 f Set.univ x).IsInvertible
    h'f : ContDiffWithinAt 𝕜 2 f Set.univ x
    ⊢ Exists fun N => And (ContDiffWithinAt 𝕜 1 (fun y => ↑(N y)) Set.univ x) (And …
  -/
  exact exists_continuousLinearEquiv_fderivWithin_symm_eq h'f hf uniqueDiffOn_univ (mem_univ _)
  /-
    🎉 no goals
  -/


/-- The Lie bracket commutes with taking pullbacks. This requires the function to have symmetric
second derivative. Version in a complete space. One could also give a version avoiding
completeness but requiring that `f` is a local diffeo. -/
lemma pullbackWithin_lieBracketWithin_of_isSymmSndFDerivWithinAt
    {f : E → F} {V W : F → F} {x : E} {t : Set F}
    (hf : IsSymmSndFDerivWithinAt 𝕜 f s x) (h'f : ContDiffWithinAt 𝕜 2 f s x)
    (hV : DifferentiableWithinAt 𝕜 V t (f x)) (hW : DifferentiableWithinAt 𝕜 W t (f x))
    (hu : UniqueDiffOn 𝕜 s) (hx : x ∈ s) (hst : MapsTo f s t) :
    pullbackWithin 𝕜 f (lieBracketWithin 𝕜 V W t) s x
      = lieBracketWithin 𝕜 (pullbackWithin 𝕜 f V s) (pullbackWithin 𝕜 f W s) s x := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    s : Set E
    inst✝ : CompleteSpace E
    f : E → F
    V W : F → F
    x : E
    t : Set F
    hf : IsSymmSndFDerivWithinAt 𝕜 f s x
    h'f : ContDiffWithinAt 𝕜 2 f s x
    hV : DifferentiableWithinAt 𝕜 V t (f x)
    hW : DifferentiableWithinAt 𝕜 W t (f x)
    hu : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hst : Set.MapsTo f s t
    ⊢ Eq (VectorField.pullbackWithin 𝕜 f (VectorField.lieBracketWithin 𝕜 V W t) s  …
  -/
  by_cases h : (fderivWithin 𝕜 f s x).IsInvertible; swap
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      s : Set E
      inst✝ : CompleteSpace E
      f : E → F
      V W : F → F
      x : E
      t : Set F
      hf : IsSymmSndFDerivWithinAt 𝕜 f s x
      h'f : ContDiffWithinAt 𝕜 2 f s x
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hW : DifferentiableWithinAt 𝕜 W t (f x)
      hu : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      hst : Set.MapsTo f s t
      h : Not (fderivWithin 𝕜 f s x).IsInvertible
      ⊢ Eq (VectorField.pullbackWithin 𝕜 f (VectorField.lieBracketWithin 𝕜 V W t) s  …
    -/
  · simp [pullbackWithin_eq_of_not_isInvertible h, lieBracketWithin_eq]
    /-
      🎉 no goals
    -/
  rcases exists_continuousLinearEquiv_fderivWithin_symm_eq h'f h hu hx
    with ⟨M, -, M_symm_smooth, hM, M_diff⟩
  /-
    case pos.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    s : Set E
    inst✝ : CompleteSpace E
    f : E → F
    V W : F → F
    x : E
    t : Set F
    hf : IsSymmSndFDerivWithinAt 𝕜 f s x
    h'f : ContDiffWithinAt 𝕜 2 f s x
    hV : DifferentiableWithinAt 𝕜 V t (f x)
    hW : DifferentiableWithinAt 𝕜 W t (f x)
    hu : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hst : Set.MapsTo f s t
    h : (fderivWithin 𝕜 f s x).IsInvertible
    M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
    M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
    hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
    M_diff : ∀ (v : E), Eq ((fderivWithin 𝕜 (fun y => ↑(M y).symm) s x) v) (Neg.ne …
    ⊢ Eq (VectorField.pullbackWithin 𝕜 f (VectorField.lieBracketWithin 𝕜 V W t) s  …
  -/
  have hMx : M x = fderivWithin 𝕜 f s x := (mem_of_mem_nhdsWithin hx hM :)
  have AV : fderivWithin 𝕜 (pullbackWithin 𝕜 f V s) s x =
      fderivWithin 𝕜 (fun y ↦ ((M y).symm : F →L[𝕜] E) (V (f y))) s x := by
    apply Filter.EventuallyEq.fderivWithin_eq_of_mem _ hx
    filter_upwards [hM] with y hy using pullbackWithin_eq_of_fderivWithin_eq hy _
  have AW : fderivWithin 𝕜 (pullbackWithin 𝕜 f W s) s x =
      fderivWithin 𝕜 (fun y ↦ ((M y).symm : F →L[𝕜] E) (W (f y))) s x := by
    apply Filter.EventuallyEq.fderivWithin_eq_of_mem _ hx
    filter_upwards [hM] with y hy using pullbackWithin_eq_of_fderivWithin_eq hy _
  /-
    case pos.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    s : Set E
    inst✝ : CompleteSpace E
    f : E → F
    V W : F → F
    x : E
    t : Set F
    hf : IsSymmSndFDerivWithinAt 𝕜 f s x
    h'f : ContDiffWithinAt 𝕜 2 f s x
    hV : DifferentiableWithinAt 𝕜 V t (f x)
    hW : DifferentiableWithinAt 𝕜 W t (f x)
    hu : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hst : Set.MapsTo f s t
    h : (fderivWithin 𝕜 f s x).IsInvertible
    M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
    M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
    hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
    M_diff : ∀ (v : E), Eq ((fderivWithin 𝕜 (fun y => ↑(M y).symm) s x) v) (Neg.ne …
    hMx : Eq (↑(M x)) (fderivWithin 𝕜 f s x)
    AV : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f V s) s x) (fderivWithi …
    AW : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f W s) s x) (fderivWithi …
    ⊢ Eq (VectorField.pullbackWithin 𝕜 f (VectorField.lieBracketWithin 𝕜 V W t) s  …
  -/
  have Af : DifferentiableWithinAt 𝕜 f s x := h'f.differentiableWithinAt one_le_two
  /-
    case pos.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    s : Set E
    inst✝ : CompleteSpace E
    f : E → F
    V W : F → F
    x : E
    t : Set F
    hf : IsSymmSndFDerivWithinAt 𝕜 f s x
    h'f : ContDiffWithinAt 𝕜 2 f s x
    hV : DifferentiableWithinAt 𝕜 V t (f x)
    hW : DifferentiableWithinAt 𝕜 W t (f x)
    hu : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hst : Set.MapsTo f s t
    h : (fderivWithin 𝕜 f s x).IsInvertible
    M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
    M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
    hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
    M_diff : ∀ (v : E), Eq ((fderivWithin 𝕜 (fun y => ↑(M y).symm) s x) v) (Neg.ne …
    hMx : Eq (↑(M x)) (fderivWithin 𝕜 f s x)
    AV : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f V s) s x) (fderivWithi …
    AW : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f W s) s x) (fderivWithi …
    Af : DifferentiableWithinAt 𝕜 f s x
    ⊢ Eq (VectorField.pullbackWithin 𝕜 f (VectorField.lieBracketWithin 𝕜 V W t) s  …
  -/
  simp only [lieBracketWithin_eq, pullbackWithin_eq_of_fderivWithin_eq hMx, map_sub, AV, AW]
  /-
    case pos.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    s : Set E
    inst✝ : CompleteSpace E
    f : E → F
    V W : F → F
    x : E
    t : Set F
    hf : IsSymmSndFDerivWithinAt 𝕜 f s x
    h'f : ContDiffWithinAt 𝕜 2 f s x
    hV : DifferentiableWithinAt 𝕜 V t (f x)
    hW : DifferentiableWithinAt 𝕜 W t (f x)
    hu : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hst : Set.MapsTo f s t
    h : (fderivWithin 𝕜 f s x).IsInvertible
    M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
    M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
    hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
    M_diff : ∀ (v : E), Eq ((fderivWithin 𝕜 (fun y => ↑(M y).symm) s x) v) (Neg.ne …
    hMx : Eq (↑(M x)) (fderivWithin 𝕜 f s x)
    AV : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f V s) s x) (fderivWithi …
    AW : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f W s) s x) (fderivWithi …
    Af : DifferentiableWithinAt 𝕜 f s x
    ⊢ Eq (HSub.hSub ((M x).symm ((fderivWithin 𝕜 W t (f x)) (V (f x)))) ((M x).sym …
  -/
  rw [fderivWithin_clm_apply, fderivWithin_clm_apply]
  · simp [fderivWithin_comp' x hW Af hst (hu x hx), ← hMx,
      fderivWithin_comp' x hV Af hst (hu x hx), M_diff, hf.eq]
    /-
      case pos.intro.intro.intro.intro.hxs
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      s : Set E
      inst✝ : CompleteSpace E
      f : E → F
      V W : F → F
      x : E
      t : Set F
      hf : IsSymmSndFDerivWithinAt 𝕜 f s x
      h'f : ContDiffWithinAt 𝕜 2 f s x
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hW : DifferentiableWithinAt 𝕜 W t (f x)
      hu : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      hst : Set.MapsTo f s t
      h : (fderivWithin 𝕜 f s x).IsInvertible
      M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
      M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
      hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
      M_diff : ∀ (v : E), Eq ((fderivWithin 𝕜 (fun y => ↑(M y).symm) s x) v) (Neg.ne …
      hMx : Eq (↑(M x)) (fderivWithin 𝕜 f s x)
      AV : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f V s) s x) (fderivWithi …
      AW : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f W s) s x) (fderivWithi …
      Af : DifferentiableWithinAt 𝕜 f s x
      ⊢ UniqueDiffWithinAt 𝕜 s x
    -/
  · exact hu x hx
    /-
      🎉 no goals
    -/
    /-
      case pos.intro.intro.intro.intro.hc
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      s : Set E
      inst✝ : CompleteSpace E
      f : E → F
      V W : F → F
      x : E
      t : Set F
      hf : IsSymmSndFDerivWithinAt 𝕜 f s x
      h'f : ContDiffWithinAt 𝕜 2 f s x
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hW : DifferentiableWithinAt 𝕜 W t (f x)
      hu : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      hst : Set.MapsTo f s t
      h : (fderivWithin 𝕜 f s x).IsInvertible
      M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
      M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
      hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
      M_diff : ∀ (v : E), Eq ((fderivWithin 𝕜 (fun y => ↑(M y).symm) s x) v) (Neg.ne …
      hMx : Eq (↑(M x)) (fderivWithin 𝕜 f s x)
      AV : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f V s) s x) (fderivWithi …
      AW : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f W s) s x) (fderivWithi …
      Af : DifferentiableWithinAt 𝕜 f s x
      ⊢ DifferentiableWithinAt 𝕜 (fun y => ↑(M y).symm) s x
    -/
  · exact M_symm_smooth.differentiableWithinAt le_rfl
    /-
      🎉 no goals
    -/
    /-
      case pos.intro.intro.intro.intro.hu
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      s : Set E
      inst✝ : CompleteSpace E
      f : E → F
      V W : F → F
      x : E
      t : Set F
      hf : IsSymmSndFDerivWithinAt 𝕜 f s x
      h'f : ContDiffWithinAt 𝕜 2 f s x
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hW : DifferentiableWithinAt 𝕜 W t (f x)
      hu : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      hst : Set.MapsTo f s t
      h : (fderivWithin 𝕜 f s x).IsInvertible
      M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
      M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
      hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
      M_diff : ∀ (v : E), Eq ((fderivWithin 𝕜 (fun y => ↑(M y).symm) s x) v) (Neg.ne …
      hMx : Eq (↑(M x)) (fderivWithin 𝕜 f s x)
      AV : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f V s) s x) (fderivWithi …
      AW : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f W s) s x) (fderivWithi …
      Af : DifferentiableWithinAt 𝕜 f s x
      ⊢ DifferentiableWithinAt 𝕜 (fun y => V (f y)) s x
    -/
  · exact hV.comp x Af hst
    /-
      🎉 no goals
    -/
    /-
      case pos.intro.intro.intro.intro.hxs
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      s : Set E
      inst✝ : CompleteSpace E
      f : E → F
      V W : F → F
      x : E
      t : Set F
      hf : IsSymmSndFDerivWithinAt 𝕜 f s x
      h'f : ContDiffWithinAt 𝕜 2 f s x
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hW : DifferentiableWithinAt 𝕜 W t (f x)
      hu : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      hst : Set.MapsTo f s t
      h : (fderivWithin 𝕜 f s x).IsInvertible
      M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
      M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
      hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
      M_diff : ∀ (v : E), Eq ((fderivWithin 𝕜 (fun y => ↑(M y).symm) s x) v) (Neg.ne …
      hMx : Eq (↑(M x)) (fderivWithin 𝕜 f s x)
      AV : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f V s) s x) (fderivWithi …
      AW : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f W s) s x) (fderivWithi …
      Af : DifferentiableWithinAt 𝕜 f s x
      ⊢ UniqueDiffWithinAt 𝕜 s x
    -/
  · exact hu x hx
    /-
      🎉 no goals
    -/
    /-
      case pos.intro.intro.intro.intro.hc
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      s : Set E
      inst✝ : CompleteSpace E
      f : E → F
      V W : F → F
      x : E
      t : Set F
      hf : IsSymmSndFDerivWithinAt 𝕜 f s x
      h'f : ContDiffWithinAt 𝕜 2 f s x
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hW : DifferentiableWithinAt 𝕜 W t (f x)
      hu : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      hst : Set.MapsTo f s t
      h : (fderivWithin 𝕜 f s x).IsInvertible
      M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
      M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
      hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
      M_diff : ∀ (v : E), Eq ((fderivWithin 𝕜 (fun y => ↑(M y).symm) s x) v) (Neg.ne …
      hMx : Eq (↑(M x)) (fderivWithin 𝕜 f s x)
      AV : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f V s) s x) (fderivWithi …
      AW : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f W s) s x) (fderivWithi …
      Af : DifferentiableWithinAt 𝕜 f s x
      ⊢ DifferentiableWithinAt 𝕜 (fun y => ↑(M y).symm) s x
    -/
  · exact M_symm_smooth.differentiableWithinAt le_rfl
    /-
      🎉 no goals
    -/
    /-
      case pos.intro.intro.intro.intro.hu
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      s : Set E
      inst✝ : CompleteSpace E
      f : E → F
      V W : F → F
      x : E
      t : Set F
      hf : IsSymmSndFDerivWithinAt 𝕜 f s x
      h'f : ContDiffWithinAt 𝕜 2 f s x
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hW : DifferentiableWithinAt 𝕜 W t (f x)
      hu : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      hst : Set.MapsTo f s t
      h : (fderivWithin 𝕜 f s x).IsInvertible
      M : E → ContinuousLinearEquiv (RingHom.id 𝕜) E F
      M_symm_smooth : ContDiffWithinAt 𝕜 1 (fun y => ↑(M y).symm) s x
      hM : Filter.Eventually (fun y => Eq (↑(M y)) (fderivWithin 𝕜 f s y)) (nhdsWith …
      M_diff : ∀ (v : E), Eq ((fderivWithin 𝕜 (fun y => ↑(M y).symm) s x) v) (Neg.ne …
      hMx : Eq (↑(M x)) (fderivWithin 𝕜 f s x)
      AV : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f V s) s x) (fderivWithi …
      AW : Eq (fderivWithin 𝕜 (VectorField.pullbackWithin 𝕜 f W s) s x) (fderivWithi …
      Af : DifferentiableWithinAt 𝕜 f s x
      ⊢ DifferentiableWithinAt 𝕜 (fun y => W (f y)) s x
    -/
  · exact hW.comp x Af hst
    /-
      🎉 no goals
    -/


/-- The Lie bracket commutes with taking pullbacks. This requires the function to have symmetric
second derivative. Version in a complete space. One could also give a version avoiding
completeness but requiring that `f` is a local diffeo. Variant where unique differentiability and
the invariance property are only required in a smaller set `u`. -/
lemma pullbackWithin_lieBracketWithin_of_isSymmSndFDerivWithinAt_of_eventuallyEq
    {f : E → F} {V W : F → F} {x : E} {t : Set F} {u : Set E}
    (hf : IsSymmSndFDerivWithinAt 𝕜 f s x) (h'f : ContDiffWithinAt 𝕜 2 f s x)
    (hV : DifferentiableWithinAt 𝕜 V t (f x)) (hW : DifferentiableWithinAt 𝕜 W t (f x))
    (hu : UniqueDiffOn 𝕜 u) (hx : x ∈ u) (hst : MapsTo f u t) (hus : u =ᶠ[𝓝 x] s) :
    pullbackWithin 𝕜 f (lieBracketWithin 𝕜 V W t) s x
      = lieBracketWithin 𝕜 (pullbackWithin 𝕜 f V s) (pullbackWithin 𝕜 f W s) s x := calc
  pullbackWithin 𝕜 f (lieBracketWithin 𝕜 V W t) s x
  _ = pullbackWithin 𝕜 f (lieBracketWithin 𝕜 V W t) u x := by
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      s : Set E
      inst✝ : CompleteSpace E
      f : E → F
      V W : F → F
      x : E
      t : Set F
      u : Set E
      hf : IsSymmSndFDerivWithinAt 𝕜 f s x
      h'f : ContDiffWithinAt 𝕜 2 f s x
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hW : DifferentiableWithinAt 𝕜 W t (f x)
      hu : UniqueDiffOn 𝕜 u
      hx : Membership.mem u x
      hst : Set.MapsTo f u t
      hus : (nhds x).EventuallyEq u s
      ⊢ Eq (VectorField.pullbackWithin 𝕜 f (VectorField.lieBracketWithin 𝕜 V W t) s  …
    -/
    simp only [pullbackWithin]
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      s : Set E
      inst✝ : CompleteSpace E
      f : E → F
      V W : F → F
      x : E
      t : Set F
      u : Set E
      hf : IsSymmSndFDerivWithinAt 𝕜 f s x
      h'f : ContDiffWithinAt 𝕜 2 f s x
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hW : DifferentiableWithinAt 𝕜 W t (f x)
      hu : UniqueDiffOn 𝕜 u
      hx : Membership.mem u x
      hst : Set.MapsTo f u t
      hus : (nhds x).EventuallyEq u s
      ⊢ Eq ((fderivWithin 𝕜 f s x).inverse (VectorField.lieBracketWithin 𝕜 V W t (f  …
    -/
    congr 2
    /-
      case e_a.e_a
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      s : Set E
      inst✝ : CompleteSpace E
      f : E → F
      V W : F → F
      x : E
      t : Set F
      u : Set E
      hf : IsSymmSndFDerivWithinAt 𝕜 f s x
      h'f : ContDiffWithinAt 𝕜 2 f s x
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hW : DifferentiableWithinAt 𝕜 W t (f x)
      hu : UniqueDiffOn 𝕜 u
      hx : Membership.mem u x
      hst : Set.MapsTo f u t
      hus : (nhds x).EventuallyEq u s
      ⊢ Eq (fderivWithin 𝕜 f s x) (fderivWithin 𝕜 f u x)
    -/
    exact fderivWithin_congr_set hus.symm
    /-
      🎉 no goals
    -/
  _ = lieBracketWithin 𝕜 (pullbackWithin 𝕜 f V u) (pullbackWithin 𝕜 f W u) u x :=
    pullbackWithin_lieBracketWithin_of_isSymmSndFDerivWithinAt
      (hf.congr_set hus.symm) (h'f.congr_set hus.symm) hV hW hu hx hst
  _ = lieBracketWithin 𝕜 (pullbackWithin 𝕜 f V s) (pullbackWithin 𝕜 f W s) u x := by
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      s : Set E
      inst✝ : CompleteSpace E
      f : E → F
      V W : F → F
      x : E
      t : Set F
      u : Set E
      hf : IsSymmSndFDerivWithinAt 𝕜 f s x
      h'f : ContDiffWithinAt 𝕜 2 f s x
      hV : DifferentiableWithinAt 𝕜 V t (f x)
      hW : DifferentiableWithinAt 𝕜 W t (f x)
      hu : UniqueDiffOn 𝕜 u
      hx : Membership.mem u x
      hst : Set.MapsTo f u t
      hus : (nhds x).EventuallyEq u s
      ⊢ Eq (VectorField.lieBracketWithin 𝕜 (VectorField.pullbackWithin 𝕜 f V u) (Vec …
    -/
    apply Filter.EventuallyEq.lieBracketWithin_vectorField_eq_of_mem _ _ hx
      /-
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        s : Set E
        inst✝ : CompleteSpace E
        f : E → F
        V W : F → F
        x : E
        t : Set F
        u : Set E
        hf : IsSymmSndFDerivWithinAt 𝕜 f s x
        h'f : ContDiffWithinAt 𝕜 2 f s x
        hV : DifferentiableWithinAt 𝕜 V t (f x)
        hW : DifferentiableWithinAt 𝕜 W t (f x)
        hu : UniqueDiffOn 𝕜 u
        hx : Membership.mem u x
        hst : Set.MapsTo f u t
        hus : (nhds x).EventuallyEq u s
        ⊢ (nhdsWithin x u).EventuallyEq (VectorField.pullbackWithin 𝕜 f V u) (VectorFi …
      -/
    · apply nhdsWithin_le_nhds
      /-
        case a
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        s : Set E
        inst✝ : CompleteSpace E
        f : E → F
        V W : F → F
        x : E
        t : Set F
        u : Set E
        hf : IsSymmSndFDerivWithinAt 𝕜 f s x
        h'f : ContDiffWithinAt 𝕜 2 f s x
        hV : DifferentiableWithinAt 𝕜 V t (f x)
        hW : DifferentiableWithinAt 𝕜 W t (f x)
        hu : UniqueDiffOn 𝕜 u
        hx : Membership.mem u x
        hst : Set.MapsTo f u t
        hus : (nhds x).EventuallyEq u s
        ⊢ Membership.mem (nhds x) (setOf fun x => (fun x => Eq (VectorField.pullbackWi …
      -/
      filter_upwards [fderivWithin_eventually_congr_set (𝕜 := 𝕜) (f := f) hus] with y hy
      /-
        case h
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        s : Set E
        inst✝ : CompleteSpace E
        f : E → F
        V W : F → F
        x : E
        t : Set F
        u : Set E
        hf : IsSymmSndFDerivWithinAt 𝕜 f s x
        h'f : ContDiffWithinAt 𝕜 2 f s x
        hV : DifferentiableWithinAt 𝕜 V t (f x)
        hW : DifferentiableWithinAt 𝕜 W t (f x)
        hu : UniqueDiffOn 𝕜 u
        hx : Membership.mem u x
        hst : Set.MapsTo f u t
        hus : (nhds x).EventuallyEq u s
        y : E
        hy : Eq (fderivWithin 𝕜 f u y) (fderivWithin 𝕜 f s y)
        ⊢ Eq (VectorField.pullbackWithin 𝕜 f V u y) (VectorField.pullbackWithin 𝕜 f V  …
      -/
      simp [pullbackWithin, hy]
      /-
        🎉 no goals
      -/
      /-
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        s : Set E
        inst✝ : CompleteSpace E
        f : E → F
        V W : F → F
        x : E
        t : Set F
        u : Set E
        hf : IsSymmSndFDerivWithinAt 𝕜 f s x
        h'f : ContDiffWithinAt 𝕜 2 f s x
        hV : DifferentiableWithinAt 𝕜 V t (f x)
        hW : DifferentiableWithinAt 𝕜 W t (f x)
        hu : UniqueDiffOn 𝕜 u
        hx : Membership.mem u x
        hst : Set.MapsTo f u t
        hus : (nhds x).EventuallyEq u s
        ⊢ (nhdsWithin x u).EventuallyEq (VectorField.pullbackWithin 𝕜 f W u) (VectorFi …
      -/
    · apply nhdsWithin_le_nhds
      /-
        case a
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        s : Set E
        inst✝ : CompleteSpace E
        f : E → F
        V W : F → F
        x : E
        t : Set F
        u : Set E
        hf : IsSymmSndFDerivWithinAt 𝕜 f s x
        h'f : ContDiffWithinAt 𝕜 2 f s x
        hV : DifferentiableWithinAt 𝕜 V t (f x)
        hW : DifferentiableWithinAt 𝕜 W t (f x)
        hu : UniqueDiffOn 𝕜 u
        hx : Membership.mem u x
        hst : Set.MapsTo f u t
        hus : (nhds x).EventuallyEq u s
        ⊢ Membership.mem (nhds x) (setOf fun x => (fun x => Eq (VectorField.pullbackWi …
      -/
      filter_upwards [fderivWithin_eventually_congr_set (𝕜 := 𝕜) (f := f) hus] with y hy
      /-
        case h
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        s : Set E
        inst✝ : CompleteSpace E
        f : E → F
        V W : F → F
        x : E
        t : Set F
        u : Set E
        hf : IsSymmSndFDerivWithinAt 𝕜 f s x
        h'f : ContDiffWithinAt 𝕜 2 f s x
        hV : DifferentiableWithinAt 𝕜 V t (f x)
        hW : DifferentiableWithinAt 𝕜 W t (f x)
        hu : UniqueDiffOn 𝕜 u
        hx : Membership.mem u x
        hst : Set.MapsTo f u t
        hus : (nhds x).EventuallyEq u s
        y : E
        hy : Eq (fderivWithin 𝕜 f u y) (fderivWithin 𝕜 f s y)
        ⊢ Eq (VectorField.pullbackWithin 𝕜 f W u y) (VectorField.pullbackWithin 𝕜 f W  …
      -/
      simp [pullbackWithin, hy]
      /-
        🎉 no goals
      -/
  _ = lieBracketWithin 𝕜 (pullbackWithin 𝕜 f V s) (pullbackWithin 𝕜 f W s) s x :=
    lieBracketWithin_congr_set hus


/-- The Lie bracket commutes with taking pullbacks. This requires the function to have symmetric
second derivative. Version in a complete space. One could also give a version avoiding
completeness but requiring that `f` is a local diffeo. -/
lemma pullback_lieBracket_of_isSymmSndFDerivAt {f : E → F} {V W : F → F} {x : E}
    (hf : IsSymmSndFDerivAt 𝕜 f x) (h'f : ContDiffAt 𝕜 2 f x)
    (hV : DifferentiableAt 𝕜 V (f x)) (hW : DifferentiableAt 𝕜 W (f x)) :
    pullback 𝕜 f (lieBracket 𝕜 V W) x = lieBracket 𝕜 (pullback 𝕜 f V) (pullback 𝕜 f W) x := by
  simp only [← lieBracketWithin_univ, ← pullbackWithin_univ, ← isSymmSndFDerivWithinAt_univ,
    ← differentiableWithinAt_univ] at hf h'f hV hW ⊢
  exact pullbackWithin_lieBracketWithin_of_isSymmSndFDerivWithinAt hf h'f hV hW uniqueDiffOn_univ
    (mem_univ _) (mapsTo_univ _ _)


/-- The Lie bracket commutes with taking pullbacks. This requires the function to have symmetric
second derivative. Version in a complete space. One could also give a version avoiding
completeness but requiring that `f` is a local diffeo. -/
lemma pullbackWithin_lieBracketWithin
    {f : E → F} {V W : F → F} {x : E} {t : Set F} (hn : minSmoothness 𝕜 2 ≤ n)
    (h'f : ContDiffWithinAt 𝕜 n f s x)
    (hV : DifferentiableWithinAt 𝕜 V t (f x)) (hW : DifferentiableWithinAt 𝕜 W t (f x))
    (hu : UniqueDiffOn 𝕜 s) (hx : x ∈ s) (h'x : x ∈ closure (interior s)) (hst : MapsTo f s t) :
    pullbackWithin 𝕜 f (lieBracketWithin 𝕜 V W t) s x
      = lieBracketWithin 𝕜 (pullbackWithin 𝕜 f V s) (pullbackWithin 𝕜 f W s) s x :=
  pullbackWithin_lieBracketWithin_of_isSymmSndFDerivWithinAt
  (h'f.isSymmSndFDerivWithinAt hn hu h'x hx) (h'f.of_le (le_minSmoothness.trans hn)) hV hW hu hx hst


/-- The Lie bracket commutes with taking pullbacks. One could also give a version avoiding
completeness but requiring that `f` is a local diffeo. -/
lemma pullback_lieBracket (hn : minSmoothness 𝕜 2 ≤ n)
    {f : E → F} {V W : F → F} {x : E} (h'f : ContDiffAt 𝕜 n f x)
    (hV : DifferentiableAt 𝕜 V (f x)) (hW : DifferentiableAt 𝕜 W (f x)) :
    pullback 𝕜 f (lieBracket 𝕜 V W) x = lieBracket 𝕜 (pullback 𝕜 f V) (pullback 𝕜 f W) x :=
  pullback_lieBracket_of_isSymmSndFDerivAt (h'f.isSymmSndFDerivAt hn)
    (h'f.of_le (le_minSmoothness.trans hn)) hV hW


