/-- Given `(k, a) : Unitization 𝕜 A`, the second coordinate of `Unitization.splitMul (k, a)` is
the natural representation of `Unitization 𝕜 A` on `A` given by multiplication on the left in
`A →L[𝕜] A`; note that this is not just `NonUnitalAlgHom.Lmul` for a few reasons: (a) that would
either be `A` acting on `A`, or (b) `Unitization 𝕜 A` acting on `Unitization 𝕜 A`, and (c) that's a
`NonUnitalAlgHom` but here we need an `AlgHom`. In addition, the first coordinate of
`Unitization.splitMul (k, a)` should just be `k`. See `Unitization.splitMul_apply` also. -/
def splitMul : Unitization 𝕜 A →ₐ[𝕜] 𝕜 × (A →L[𝕜] A) :=
  (lift 0).prod (lift <| NonUnitalAlgHom.Lmul 𝕜 A)


@[simp]
theorem splitMul_apply (x : Unitization 𝕜 A) :
    splitMul 𝕜 A x = (x.fst, algebraMap 𝕜 (A →L[𝕜] A) x.fst + mul 𝕜 A x.snd) :=
                                      /-
                                        𝕜 : Type u_1
                                        A : Type u_2
                                        inst✝⁴ : NontriviallyNormedField 𝕜
                                        inst✝³ : NonUnitalNormedRing A
                                        inst✝² : NormedSpace 𝕜 A
                                        inst✝¹ : IsScalarTower 𝕜 A A
                                        inst✝ : SMulCommClass 𝕜 A A
                                        x : Unitization 𝕜 A
                                        ⊢ Eq { fst := HAdd.hAdd x.fst 0, snd := (Unitization.lift (NonUnitalAlgHom.Lmu …
                                      -/
  show (x.fst + 0, _) = (x.fst, _) by rw [add_zero]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- this lemma establishes that if `ContinuousLinearMap.mul 𝕜 A` is injective, then so is
`Unitization.splitMul 𝕜 A`. When `A` is a `RegularNormedAlgebra`, then
`ContinuousLinearMap.mul 𝕜 A` is an isometry, and is therefore automatically injective. -/
theorem splitMul_injective_of_clm_mul_injective
    (h : Function.Injective (mul 𝕜 A)) :
    Function.Injective (splitMul 𝕜 A) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NonUnitalNormedRing A
    inst✝² : NormedSpace 𝕜 A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : SMulCommClass 𝕜 A A
    h : Function.Injective ⇑(ContinuousLinearMap.mul 𝕜 A)
    ⊢ Function.Injective ⇑(Unitization.splitMul 𝕜 A)
  -/
  rw [injective_iff_map_eq_zero]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NonUnitalNormedRing A
    inst✝² : NormedSpace 𝕜 A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : SMulCommClass 𝕜 A A
    h : Function.Injective ⇑(ContinuousLinearMap.mul 𝕜 A)
    ⊢ ∀ (a : Unitization 𝕜 A), Eq ((Unitization.splitMul 𝕜 A) a) 0 → Eq a 0
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NonUnitalNormedRing A
    inst✝² : NormedSpace 𝕜 A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : SMulCommClass 𝕜 A A
    h : Function.Injective ⇑(ContinuousLinearMap.mul 𝕜 A)
    x : Unitization 𝕜 A
    hx : Eq ((Unitization.splitMul 𝕜 A) x) 0
    ⊢ Eq x 0
  -/
  induction x
  /-
    case inl_add_inr
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NonUnitalNormedRing A
    inst✝² : NormedSpace 𝕜 A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : SMulCommClass 𝕜 A A
    h : Function.Injective ⇑(ContinuousLinearMap.mul 𝕜 A)
    r✝ : 𝕜
    a✝ : A
    hx : Eq ((Unitization.splitMul 𝕜 A) (HAdd.hAdd (Unitization.inl r✝) ↑a✝)) 0
    ⊢ Eq (HAdd.hAdd (Unitization.inl r✝) ↑a✝) 0
  -/
  rw [map_add] at hx
  simp only [splitMul_apply, fst_inl, snd_inl, map_zero, add_zero, fst_inr, snd_inr,
    zero_add, Prod.mk_add_mk, Prod.mk_eq_zero] at hx
  /-
    case inl_add_inr
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NonUnitalNormedRing A
    inst✝² : NormedSpace 𝕜 A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : SMulCommClass 𝕜 A A
    h : Function.Injective ⇑(ContinuousLinearMap.mul 𝕜 A)
    r✝ : 𝕜
    a✝ : A
    hx : And (Eq r✝ 0) (Eq (HAdd.hAdd ((algebraMap 𝕜 (ContinuousLinearMap (RingHom …
    ⊢ Eq (HAdd.hAdd (Unitization.inl r✝) ↑a✝) 0
  -/
  obtain ⟨rfl, hx⟩ := hx
  /-
    case inl_add_inr.intro
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NonUnitalNormedRing A
    inst✝² : NormedSpace 𝕜 A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : SMulCommClass 𝕜 A A
    h : Function.Injective ⇑(ContinuousLinearMap.mul 𝕜 A)
    a✝ : A
    hx : Eq (HAdd.hAdd ((algebraMap 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) A A)) 0) …
    ⊢ Eq (HAdd.hAdd (Unitization.inl 0) ↑a✝) 0
  -/
  simp only [map_zero, zero_add, inl_zero] at hx ⊢
  /-
    case inl_add_inr.intro
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NonUnitalNormedRing A
    inst✝² : NormedSpace 𝕜 A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : SMulCommClass 𝕜 A A
    h : Function.Injective ⇑(ContinuousLinearMap.mul 𝕜 A)
    a✝ : A
    hx : Eq ((ContinuousLinearMap.mul 𝕜 A) a✝) 0
    ⊢ Eq (↑a✝) 0
  -/
  rw [← map_zero (mul 𝕜 A)] at hx
  /-
    case inl_add_inr.intro
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NonUnitalNormedRing A
    inst✝² : NormedSpace 𝕜 A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : SMulCommClass 𝕜 A A
    h : Function.Injective ⇑(ContinuousLinearMap.mul 𝕜 A)
    a✝ : A
    hx : Eq ((ContinuousLinearMap.mul 𝕜 A) a✝) ((ContinuousLinearMap.mul 𝕜 A) 0)
    ⊢ Eq (↑a✝) 0
  -/
  rw [h hx, inr_zero]
  /-
    🎉 no goals
  -/


/-- In a `RegularNormedAlgebra`, the map `Unitization.splitMul 𝕜 A` is injective.
We will use this to pull back the norm from `𝕜 × (A →L[𝕜] A)` to `Unitization 𝕜 A`. -/
theorem splitMul_injective : Function.Injective (splitMul 𝕜 A) :=
  splitMul_injective_of_clm_mul_injective (isometry_mul 𝕜 A).injective


/-- Pull back the normed ring structure from `𝕜 × (A →L[𝕜] A)` to `Unitization 𝕜 A` using the
algebra homomorphism `Unitization.splitMul 𝕜 A`. This does not give us the desired topology,
uniformity or bornology on `Unitization 𝕜 A` (which we want to agree with `Prod`), so we only use
it as a local instance to build the real one. -/
noncomputable abbrev normedRingAux : NormedRing (Unitization 𝕜 A) :=
  NormedRing.induced (Unitization 𝕜 A) (𝕜 × (A →L[𝕜] A)) (splitMul 𝕜 A) (splitMul_injective 𝕜 A)


/-- Pull back the normed algebra structure from `𝕜 × (A →L[𝕜] A)` to `Unitization 𝕜 A` using the
algebra homomorphism `Unitization.splitMul 𝕜 A`. This uses the wrong `NormedRing` instance (i.e.,
`Unitization.normedRingAux`), so we only use it as a local instance to build the real one. -/
noncomputable abbrev normedAlgebraAux : NormedAlgebra 𝕜 (Unitization 𝕜 A) :=
  NormedAlgebra.induced 𝕜 (Unitization 𝕜 A) (𝕜 × (A →L[𝕜] A)) (splitMul 𝕜 A)


theorem norm_def (x : Unitization 𝕜 A) : ‖x‖ = ‖splitMul 𝕜 A x‖ :=
  rfl


theorem nnnorm_def (x : Unitization 𝕜 A) : ‖x‖₊ = ‖splitMul 𝕜 A x‖₊ :=
  rfl


/-- This is often the more useful lemma to rewrite the norm as opposed to `Unitization.norm_def`. -/
theorem norm_eq_sup (x : Unitization 𝕜 A) :
    ‖x‖ = ‖x.fst‖ ⊔ ‖algebraMap 𝕜 (A →L[𝕜] A) x.fst + mul 𝕜 A x.snd‖ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : RegularNormedAlgebra 𝕜 A
    x : Unitization 𝕜 A
    ⊢ Eq (Norm.norm x) (Max.max (Norm.norm x.fst) (Norm.norm (HAdd.hAdd ((algebraM …
  -/
  rw [norm_def, splitMul_apply, Prod.norm_def]
  /-
    🎉 no goals
  -/


/-- This is often the more useful lemma to rewrite the norm as opposed to
`Unitization.nnnorm_def`. -/
theorem nnnorm_eq_sup (x : Unitization 𝕜 A) :
    ‖x‖₊ = ‖x.fst‖₊ ⊔ ‖algebraMap 𝕜 (A →L[𝕜] A) x.fst + mul 𝕜 A x.snd‖₊ :=
  NNReal.eq <| norm_eq_sup x


theorem lipschitzWith_addEquiv :
    LipschitzWith 2 (Unitization.addEquiv 𝕜 A) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : RegularNormedAlgebra 𝕜 A
    ⊢ LipschitzWith 2 ⇑(Unitization.addEquiv 𝕜 A)
  -/
  rw [← Real.toNNReal_ofNat]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : RegularNormedAlgebra 𝕜 A
    ⊢ LipschitzWith 2.toNNReal ⇑(Unitization.addEquiv 𝕜 A)
  -/
  refine AddMonoidHomClass.lipschitz_of_bound (Unitization.addEquiv 𝕜 A) 2 fun x => ?_
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : RegularNormedAlgebra 𝕜 A
    x : Unitization 𝕜 A
    ⊢ LE.le (Norm.norm ((Unitization.addEquiv 𝕜 A) x)) (HMul.hMul 2 (Norm.norm x))
  -/
  rw [norm_eq_sup, Prod.norm_def]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : RegularNormedAlgebra 𝕜 A
    x : Unitization 𝕜 A
    ⊢ LE.le (Max.max (Norm.norm ((Unitization.addEquiv 𝕜 A) x).1) (Norm.norm ((Uni …
  -/
  refine max_le ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NonUnitalNormedRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : RegularNormedAlgebra 𝕜 A
      x : Unitization 𝕜 A
      ⊢ LE.le (Norm.norm ((Unitization.addEquiv 𝕜 A) x).1) (HMul.hMul 2 (Max.max (No …
    -/
  · rw [mul_max_of_nonneg _ _ (zero_le_two : (0 : ℝ) ≤ 2)]
    /-
      case refine_1
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NonUnitalNormedRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : RegularNormedAlgebra 𝕜 A
      x : Unitization 𝕜 A
      ⊢ LE.le (Norm.norm ((Unitization.addEquiv 𝕜 A) x).1) (Max.max (HMul.hMul 2 (No …
    -/
    exact le_max_of_le_left ((le_add_of_nonneg_left (norm_nonneg _)).trans_eq (two_mul _).symm)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NonUnitalNormedRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : RegularNormedAlgebra 𝕜 A
      x : Unitization 𝕜 A
      ⊢ LE.le (Norm.norm ((Unitization.addEquiv 𝕜 A) x).2) (HMul.hMul 2 (Max.max (No …
    -/
  · nontriviality A
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NonUnitalNormedRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : RegularNormedAlgebra 𝕜 A
      x : Unitization 𝕜 A
      a✝ : Nontrivial A
      ⊢ LE.le (Norm.norm ((Unitization.addEquiv 𝕜 A) x).2) (HMul.hMul 2 (Max.max (No …
    -/
    rw [two_mul]
    calc
      ‖x.snd‖ = ‖mul 𝕜 A x.snd‖ :=
        .symm <| (isometry_mul 𝕜 A).norm_map_of_map_zero (map_zero _) _
      _ ≤ ‖algebraMap 𝕜 _ x.fst + mul 𝕜 A x.snd‖ + ‖x.fst‖ := by
        simpa only [add_comm _ (mul 𝕜 A x.snd), norm_algebraMap'] using
          norm_le_add_norm_add (mul 𝕜 A x.snd) (algebraMap 𝕜 _ x.fst)
      _ ≤ _ := add_le_add le_sup_right le_sup_left


theorem antilipschitzWith_addEquiv :
    AntilipschitzWith 2 (addEquiv 𝕜 A) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : RegularNormedAlgebra 𝕜 A
    ⊢ AntilipschitzWith 2 ⇑(Unitization.addEquiv 𝕜 A)
  -/
  refine AddMonoidHomClass.antilipschitz_of_bound (addEquiv 𝕜 A) fun x => ?_
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : RegularNormedAlgebra 𝕜 A
    x : Unitization 𝕜 A
    ⊢ LE.le (Norm.norm x) (HMul.hMul (↑2) (Norm.norm ((Unitization.addEquiv 𝕜 A) x …
  -/
  rw [norm_eq_sup, Prod.norm_def, NNReal.coe_two]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : RegularNormedAlgebra 𝕜 A
    x : Unitization 𝕜 A
    ⊢ LE.le (Max.max (Norm.norm x.fst) (Norm.norm (HAdd.hAdd ((algebraMap 𝕜 (Conti …
  -/
  refine max_le ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NonUnitalNormedRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : RegularNormedAlgebra 𝕜 A
      x : Unitization 𝕜 A
      ⊢ LE.le (Norm.norm x.fst) (HMul.hMul 2 (Max.max (Norm.norm ((Unitization.addEq …
    -/
  · rw [mul_max_of_nonneg _ _ (zero_le_two : (0 : ℝ) ≤ 2)]
    /-
      case refine_1
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NonUnitalNormedRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : RegularNormedAlgebra 𝕜 A
      x : Unitization 𝕜 A
      ⊢ LE.le (Norm.norm x.fst) (Max.max (HMul.hMul 2 (Norm.norm ((Unitization.addEq …
    -/
    exact le_max_of_le_left ((le_add_of_nonneg_left (norm_nonneg _)).trans_eq (two_mul _).symm)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NonUnitalNormedRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : RegularNormedAlgebra 𝕜 A
      x : Unitization 𝕜 A
      ⊢ LE.le (Norm.norm (HAdd.hAdd ((algebraMap 𝕜 (ContinuousLinearMap (RingHom.id  …
    -/
  · nontriviality A
    calc
      ‖algebraMap 𝕜 _ x.fst + mul 𝕜 A x.snd‖ ≤ ‖algebraMap 𝕜 _ x.fst‖ + ‖mul 𝕜 A x.snd‖ :=
        norm_add_le _ _
      _ = ‖x.fst‖ + ‖x.snd‖ := by
        rw [norm_algebraMap', (AddMonoidHomClass.isometry_iff_norm (mul 𝕜 A)).mp (isometry_mul 𝕜 A)]
      _ ≤ _ := (add_le_add (le_max_left _ _) (le_max_right _ _)).trans_eq (two_mul _).symm


theorem uniformity_eq_aux :
    𝓤[instUniformSpaceProd.comap <| addEquiv 𝕜 A] = 𝓤 (Unitization 𝕜 A) := by
  have key : IsUniformInducing (addEquiv 𝕜 A) :=
    antilipschitzWith_addEquiv.isUniformInducing lipschitzWith_addEquiv.uniformContinuous
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : RegularNormedAlgebra 𝕜 A
    key : IsUniformInducing ⇑(Unitization.addEquiv 𝕜 A)
    ⊢ Eq (uniformity (Unitization 𝕜 A)) (uniformity (Unitization 𝕜 A))
  -/
  rw [← key.comap_uniformity]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : RegularNormedAlgebra 𝕜 A
    key : IsUniformInducing ⇑(Unitization.addEquiv 𝕜 A)
    ⊢ Eq (uniformity (Unitization 𝕜 A)) (Filter.comap (fun x => { fst := (Unitizat …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem cobounded_eq_aux :
    @cobounded _ (Bornology.induced <| addEquiv 𝕜 A) = cobounded (Unitization 𝕜 A) :=
  le_antisymm lipschitzWith_addEquiv.comap_cobounded_le
    antilipschitzWith_addEquiv.tendsto_cobounded.le_comap


/-- The uniformity on `Unitization 𝕜 A` is inherited from `𝕜 × A`. -/
instance instUniformSpace : UniformSpace (Unitization 𝕜 A) :=
  instUniformSpaceProd.comap (addEquiv 𝕜 A)


/-- The natural equivalence between `Unitization 𝕜 A` and `𝕜 × A` as a uniform equivalence. -/
def uniformEquivProd : (Unitization 𝕜 A) ≃ᵤ (𝕜 × A) :=
  Equiv.toUniformEquivOfIsUniformInducing (addEquiv 𝕜 A) ⟨rfl⟩


/-- The bornology on `Unitization 𝕜 A` is inherited from `𝕜 × A`. -/
instance instBornology : Bornology (Unitization 𝕜 A) :=
  Bornology.induced <| addEquiv 𝕜 A


theorem isUniformEmbedding_addEquiv {𝕜} [NontriviallyNormedField 𝕜] :
    IsUniformEmbedding (addEquiv 𝕜 A) where
  comap_uniformity := rfl
  injective := (addEquiv 𝕜 A).injective


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_addEquiv := isUniformEmbedding_addEquiv


/-- `Unitization 𝕜 A` is complete whenever `𝕜` and `A` are also. -/
instance instCompleteSpace [CompleteSpace 𝕜] [CompleteSpace A] :
    CompleteSpace (Unitization 𝕜 A) :=
  uniformEquivProd.completeSpace_iff.2 .prod


/-- Pull back the metric structure from `𝕜 × (A →L[𝕜] A)` to `Unitization 𝕜 A` using the
algebra homomorphism `Unitization.splitMul 𝕜 A`, but replace the bornology and the uniformity so
that they coincide with `𝕜 × A`. -/
noncomputable instance instMetricSpace : MetricSpace (Unitization 𝕜 A) :=
  (normedRingAux.toMetricSpace.replaceUniformity uniformity_eq_aux).replaceBornology
    fun s => Filter.ext_iff.1 cobounded_eq_aux (sᶜ)


/-- Pull back the normed ring structure from `𝕜 × (A →L[𝕜] A)` to `Unitization 𝕜 A` using the
algebra homomorphism `Unitization.splitMul 𝕜 A`. -/
noncomputable instance instNormedRing : NormedRing (Unitization 𝕜 A) where
  dist_eq := normedRingAux.dist_eq
  norm_mul := normedRingAux.norm_mul
  norm := normedRingAux.norm


/-- Pull back the normed algebra structure from `𝕜 × (A →L[𝕜] A)` to `Unitization 𝕜 A` using the
algebra homomorphism `Unitization.splitMul 𝕜 A`. -/
instance instNormedAlgebra : NormedAlgebra 𝕜 (Unitization 𝕜 A) where
  norm_smul_le k x := by
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NonUnitalNormedRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : RegularNormedAlgebra 𝕜 A
      k : 𝕜
      x : Unitization 𝕜 A
      ⊢ LE.le (Norm.norm (HSMul.hSMul k x)) (HMul.hMul (Norm.norm k) (Norm.norm x))
    -/
    rw [norm_def, map_smul]
    -- Note: this used to be `rw [norm_smul, ← norm_def]` before https://github.com/leanprover-community/mathlib4/pull/8386
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NonUnitalNormedRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : RegularNormedAlgebra 𝕜 A
      k : 𝕜
      x : Unitization 𝕜 A
      ⊢ LE.le (Norm.norm (HSMul.hSMul k ((Unitization.splitMul 𝕜 A) x))) (HMul.hMul  …
    -/
    exact (norm_smul k (splitMul 𝕜 A x)).le
    /-
      🎉 no goals
    -/


instance instNormOneClass : NormOneClass (Unitization 𝕜 A) where
  norm_one := by simpa only [norm_eq_sup, fst_one, norm_one, snd_one, map_one, map_zero,
      add_zero, sup_eq_left] using opNorm_le_bound _ zero_le_one fun x => by simp


lemma norm_inr (a : A) : ‖(a : Unitization 𝕜 A)‖ = ‖a‖ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : RegularNormedAlgebra 𝕜 A
    a : A
    ⊢ Eq (Norm.norm ↑a) (Norm.norm a)
  -/
  simp [norm_eq_sup]
  /-
    🎉 no goals
  -/


lemma nnnorm_inr (a : A) : ‖(a : Unitization 𝕜 A)‖₊ = ‖a‖₊ :=
  NNReal.eq <| norm_inr a


lemma isometry_inr : Isometry ((↑) : A → Unitization 𝕜 A) :=
  AddMonoidHomClass.isometry_of_norm (inrNonUnitalAlgHom 𝕜 A) norm_inr


@[fun_prop]
theorem continuous_inr : Continuous (inr : A → Unitization 𝕜 A) :=
  isometry_inr.continuous


lemma dist_inr (a b : A) : dist (a : Unitization 𝕜 A) (b : Unitization 𝕜 A) = dist a b :=
  isometry_inr.dist_eq a b


lemma nndist_inr (a b : A) : nndist (a : Unitization 𝕜 A) (b : Unitization 𝕜 A) = nndist a b :=
  isometry_inr.nndist_eq a b

/- These examples verify that the bornology and uniformity (hence also the topology) are the
correct ones. -/

