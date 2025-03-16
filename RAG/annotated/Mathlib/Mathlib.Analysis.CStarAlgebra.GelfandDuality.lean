/-- Every maximal ideal in a commutative complex Banach algebra gives rise to a character on that
algebra. In particular, the character, which may be identified as an algebra homomorphism due to
`WeakDual.CharacterSpace.equivAlgHom`, is given by the composition of the quotient map and
the Gelfand-Mazur isomorphism `NormedRing.algEquivComplexOfComplete`. -/
noncomputable def Ideal.toCharacterSpace : characterSpace ℂ A :=
  CharacterSpace.equivAlgHom.symm <|
    ((NormedRing.algEquivComplexOfComplete
      (letI := Quotient.field I; isUnit_iff_ne_zero (G₀ := A ⧸ I))).symm : A ⧸ I →ₐ[ℂ] ℂ).comp <|
    Quotient.mkₐ ℂ I


theorem Ideal.toCharacterSpace_apply_eq_zero_of_mem {a : A} (ha : a ∈ I) :
    I.toCharacterSpace a = 0 := by
  /-
    A : Type u_1
    inst✝³ : NormedCommRing A
    inst✝² : NormedAlgebra Complex A
    inst✝¹ : CompleteSpace A
    I : Ideal A
    inst✝ : I.IsMaximal
    a : A
    ha : Membership.mem I a
    ⊢ Eq (I.toCharacterSpace a) 0
  -/
  unfold Ideal.toCharacterSpace
  simp only [CharacterSpace.equivAlgHom_symm_coe, AlgHom.coe_comp, AlgHom.coe_coe,
    Quotient.mkₐ_eq_mk, Function.comp_apply, NormedRing.algEquivComplexOfComplete_symm_apply]
  /-
    A : Type u_1
    inst✝³ : NormedCommRing A
    inst✝² : NormedAlgebra Complex A
    inst✝¹ : CompleteSpace A
    I : Ideal A
    inst✝ : I.IsMaximal
    a : A
    ha : Membership.mem I a
    ⊢ Eq ⋯.some 0
  -/
  simp_rw [Quotient.eq_zero_iff_mem.mpr ha, spectrum.zero_eq]
  /-
    A : Type u_1
    inst✝³ : NormedCommRing A
    inst✝² : NormedAlgebra Complex A
    inst✝¹ : CompleteSpace A
    I : Ideal A
    inst✝ : I.IsMaximal
    a : A
    ha : Membership.mem I a
    ⊢ Eq ⋯.some 0
  -/
  exact Set.eq_of_mem_singleton (Set.singleton_nonempty (0 : ℂ)).some_mem
  /-
    🎉 no goals
  -/


/-- If `a : A` is not a unit, then some character takes the value zero at `a`. This is equivalent
to `gelfandTransform ℂ A a` takes the value zero at some character. -/
theorem WeakDual.CharacterSpace.exists_apply_eq_zero {a : A} (ha : ¬IsUnit a) :
    ∃ f : characterSpace ℂ A, f a = 0 := by
  /-
    A : Type u_1
    inst✝² : NormedCommRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    ha : Not (IsUnit a)
    ⊢ Exists fun f => Eq (f a) 0
  -/
  obtain ⟨M, hM, haM⟩ := (span {a}).exists_le_maximal (span_singleton_ne_top ha)
  exact
    ⟨M.toCharacterSpace,
      M.toCharacterSpace_apply_eq_zero_of_mem
        (haM (mem_span_singleton.mpr ⟨1, (mul_one a).symm⟩))⟩


theorem WeakDual.CharacterSpace.mem_spectrum_iff_exists {a : A} {z : ℂ} :
    z ∈ spectrum ℂ a ↔ ∃ f : characterSpace ℂ A, f a = z := by
  /-
    A : Type u_1
    inst✝² : NormedCommRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    z : Complex
    ⊢ Iff (Membership.mem (spectrum Complex a) z) (Exists fun f => Eq (f a) z)
  -/
  refine ⟨fun hz => ?_, ?_⟩
    /-
      case refine_1
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra Complex A
      inst✝ : CompleteSpace A
      a : A
      z : Complex
      hz : Membership.mem (spectrum Complex a) z
      ⊢ Exists fun f => Eq (f a) z
    -/
  · obtain ⟨f, hf⟩ := WeakDual.CharacterSpace.exists_apply_eq_zero hz
    /-
      case refine_1.intro
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra Complex A
      inst✝ : CompleteSpace A
      a : A
      z : Complex
      hz : Membership.mem (spectrum Complex a) z
      f : ↑(WeakDual.characterSpace Complex A)
      hf : Eq (f (HSub.hSub ((algebraMap Complex A) z) a)) 0
      ⊢ Exists fun f => Eq (f a) z
    -/
    simp only [map_sub, sub_eq_zero, AlgHomClass.commutes] at hf
    /-
      case refine_1.intro
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra Complex A
      inst✝ : CompleteSpace A
      a : A
      z : Complex
      hz : Membership.mem (spectrum Complex a) z
      f : ↑(WeakDual.characterSpace Complex A)
      hf : Eq ((algebraMap Complex Complex) z) (f a)
      ⊢ Exists fun f => Eq (f a) z
    -/
    exact ⟨_, hf.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra Complex A
      inst✝ : CompleteSpace A
      a : A
      z : Complex
      ⊢ (Exists fun f => Eq (f a) z) → Membership.mem (spectrum Complex a) z
    -/
  · rintro ⟨f, rfl⟩
    /-
      case refine_2.intro
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra Complex A
      inst✝ : CompleteSpace A
      a : A
      f : ↑(WeakDual.characterSpace Complex A)
      ⊢ Membership.mem (spectrum Complex a) (f a)
    -/
    exact AlgHom.apply_mem_spectrum f a
    /-
      🎉 no goals
    -/


/-- The Gelfand transform is spectrum-preserving. -/
theorem spectrum.gelfandTransform_eq (a : A) :
    spectrum ℂ (gelfandTransform ℂ A a) = spectrum ℂ a := by
  /-
    A : Type u_1
    inst✝² : NormedCommRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    ⊢ Eq (spectrum Complex ((WeakDual.gelfandTransform Complex A) a)) (spectrum Co …
  -/
  ext z
  /-
    case h
    A : Type u_1
    inst✝² : NormedCommRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    z : Complex
    ⊢ Iff (Membership.mem (spectrum Complex ((WeakDual.gelfandTransform Complex A) …
  -/
  rw [ContinuousMap.spectrum_eq_range, WeakDual.CharacterSpace.mem_spectrum_iff_exists]
  /-
    case h
    A : Type u_1
    inst✝² : NormedCommRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    z : Complex
    ⊢ Iff (Membership.mem (Set.range ⇑((WeakDual.gelfandTransform Complex A) a)) z …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


instance [Nontrivial A] : Nonempty (characterSpace ℂ A) :=
  ⟨Classical.choose <|
      WeakDual.CharacterSpace.exists_apply_eq_zero <| zero_mem_nonunits.2 zero_ne_one⟩


theorem gelfandTransform_map_star (a : A) :
    gelfandTransform ℂ A (star a) = star (gelfandTransform ℂ A a) :=
  ContinuousMap.ext fun φ => map_star φ a


/-- The Gelfand transform is an isometry when the algebra is a C⋆-algebra over `ℂ`. -/
theorem gelfandTransform_isometry : Isometry (gelfandTransform ℂ A) := by
  /-
    A : Type u_1
    inst✝ : CommCStarAlgebra A
    ⊢ Isometry ⇑(WeakDual.gelfandTransform Complex A)
  -/
  nontriviality A
  /-
    A : Type u_1
    inst✝ : CommCStarAlgebra A
    a✝ : Nontrivial A
    ⊢ Isometry ⇑(WeakDual.gelfandTransform Complex A)
  -/
  refine AddMonoidHomClass.isometry_of_norm (gelfandTransform ℂ A) fun a => ?_
  /- By `spectrum.gelfandTransform_eq`, the spectra of `star a * a` and its
    `gelfandTransform` coincide. Therefore, so do their spectral radii, and since they are
    self-adjoint, so also do their norms. Applying the C⋆-property of the norm and taking square
    roots shows that the norm is preserved. -/
  have : spectralRadius ℂ (gelfandTransform ℂ A (star a * a)) = spectralRadius ℂ (star a * a) := by
    unfold spectralRadius; rw [spectrum.gelfandTransform_eq]
  rw [map_mul, (IsSelfAdjoint.star_mul_self a).spectralRadius_eq_nnnorm, gelfandTransform_map_star,
    (IsSelfAdjoint.star_mul_self (gelfandTransform ℂ A a)).spectralRadius_eq_nnnorm] at this
  /-
    A : Type u_1
    inst✝ : CommCStarAlgebra A
    a✝ : Nontrivial A
    a : A
    this : Eq ↑(NNNorm.nnnorm (HMul.hMul (Star.star ((WeakDual.gelfandTransform Co …
    ⊢ Eq (Norm.norm ((WeakDual.gelfandTransform Complex A) a)) (Norm.norm a)
  -/
  simp only [ENNReal.coe_inj, CStarRing.nnnorm_star_mul_self, ← sq] at this
  simpa only [Function.comp_apply, NNReal.sqrt_sq] using
    congr_arg (((↑) : ℝ≥0 → ℝ) ∘ ⇑NNReal.sqrt) this


/-- The Gelfand transform is bijective when the algebra is a C⋆-algebra over `ℂ`. -/
theorem gelfandTransform_bijective : Function.Bijective (gelfandTransform ℂ A) := by
  /-
    A : Type u_1
    inst✝ : CommCStarAlgebra A
    ⊢ Function.Bijective ⇑(WeakDual.gelfandTransform Complex A)
  -/
  refine ⟨(gelfandTransform_isometry A).injective, ?_⟩
  /- The range of `gelfandTransform ℂ A` is actually a `StarSubalgebra`. The key lemma below may be
    hard to spot; it's `map_star` coming from `WeakDual.Complex.instStarHomClass`, which is a
    nontrivial result. -/
  let rng : StarSubalgebra ℂ C(characterSpace ℂ A, ℂ) :=
    { toSubalgebra := (gelfandTransform ℂ A).range
      star_mem' := by
        rintro - ⟨a, rfl⟩
        use star a
        ext1 φ
        dsimp
        simp only [map_star, RCLike.star_def] }
  suffices rng = ⊤ from
    fun x => show x ∈ rng from this.symm ▸ StarSubalgebra.mem_top
  /- Because the `gelfandTransform ℂ A` is an isometry, it has closed range, and so by the
    Stone-Weierstrass theorem, it suffices to show that the image of the Gelfand transform separates
    points in `C(characterSpace ℂ A, ℂ)` and is closed under `star`. -/
  have h : rng.topologicalClosure = rng := le_antisymm
    (StarSubalgebra.topologicalClosure_minimal le_rfl
      (gelfandTransform_isometry A).isClosedEmbedding.isClosed_range)
    (StarSubalgebra.le_topologicalClosure _)
  refine h ▸ ContinuousMap.starSubalgebra_topologicalClosure_eq_top_of_separatesPoints
    _ (fun _ _ => ?_)
  /- Separating points just means that elements of the `characterSpace` which agree at all points
    of `A` are the same functional, which is just extensionality. -/
  /-
    A : Type u_1
    inst✝ : CommCStarAlgebra A
    rng : StarSubalgebra Complex (ContinuousMap (↑(WeakDual.characterSpace Complex …
    h : Eq rng.topologicalClosure rng
    x✝¹ x✝ : ↑(WeakDual.characterSpace Complex A)
    ⊢ Ne x✝¹ x✝ → Exists fun f => And (Membership.mem (Set.image (fun f => ⇑f) ↑rn …
  -/
  contrapose!
  exact fun h => Subtype.ext (ContinuousLinearMap.ext fun a =>
    h (gelfandTransform ℂ A a) ⟨gelfandTransform ℂ A a, ⟨a, rfl⟩, rfl⟩)


/-- The Gelfand transform as a `StarAlgEquiv` between a commutative unital C⋆-algebra over `ℂ`
and the continuous functions on its `characterSpace`. -/
@[simps!]
noncomputable def gelfandStarTransform : A ≃⋆ₐ[ℂ] C(characterSpace ℂ A, ℂ) :=
  StarAlgEquiv.ofBijective
    (show A →⋆ₐ[ℂ] C(characterSpace ℂ A, ℂ) from
      { gelfandTransform ℂ A with map_star' := fun x => gelfandTransform_map_star x })
    (gelfandTransform_bijective A)


/-- The functorial map taking `ψ : A →⋆ₐ[ℂ] B` to a continuous function
`characterSpace ℂ B → characterSpace ℂ A` obtained by pre-composition with `ψ`. -/
@[simps]
noncomputable def compContinuousMap (ψ : A →⋆ₐ[𝕜] B) :
    C(characterSpace 𝕜 B, characterSpace 𝕜 A) where
  toFun φ := equivAlgHom.symm ((equivAlgHom φ).comp ψ.toAlgHom)
  continuous_toFun :=
    Continuous.subtype_mk
      (continuous_of_continuous_eval fun a => map_continuous <| gelfandTransform 𝕜 B (ψ a)) _


/-- `WeakDual.CharacterSpace.compContinuousMap` sends the identity to the identity. -/
@[simp]
theorem compContinuousMap_id :
    compContinuousMap (StarAlgHom.id 𝕜 A) = ContinuousMap.id (characterSpace 𝕜 A) :=
  ContinuousMap.ext fun _a => ext fun _x => rfl


/-- `WeakDual.CharacterSpace.compContinuousMap` is functorial. -/
@[simp]
theorem compContinuousMap_comp (ψ₂ : B →⋆ₐ[𝕜] C) (ψ₁ : A →⋆ₐ[𝕜] B) :
    compContinuousMap (ψ₂.comp ψ₁) = (compContinuousMap ψ₁).comp (compContinuousMap ψ₂) :=
  ContinuousMap.ext fun _a => ext fun _x => rfl


open CharacterSpace in
/--
Consider the contravariant functors between compact Hausdorff spaces and commutative unital
C⋆algebras `F : Cpct → CommCStarAlg := X ↦ C(X, ℂ)` and
`G : CommCStarAlg → Cpct := A → characterSpace ℂ A` whose actions on morphisms are given by
`WeakDual.CharacterSpace.compContinuousMap` and `ContinuousMap.compStarAlgHom'`, respectively.

Then `η : id → F ∘ G := gelfandStarTransform` is a natural isomorphism implementing (half of)
the duality between these categories. That is, for commutative unital C⋆-algebras `A` and `B` and
`φ : A →⋆ₐ[ℂ] B` the following diagram commutes:

```
A  --- η A ---> C(characterSpace ℂ A, ℂ)

|                     |

φ                  (F ∘ G) φ

|                     |
V                     V

B  --- η B ---> C(characterSpace ℂ B, ℂ)
```
-/
theorem gelfandStarTransform_naturality {A B : Type*} [CommCStarAlgebra A] [CommCStarAlgebra B]
    (φ : A →⋆ₐ[ℂ] B) :
    (gelfandStarTransform B : _ →⋆ₐ[ℂ] _).comp φ =
      (compContinuousMap φ |>.compStarAlgHom' ℂ ℂ).comp (gelfandStarTransform A : _ →⋆ₐ[ℂ] _) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommCStarAlgebra A
    inst✝ : CommCStarAlgebra B
    φ : StarAlgHom Complex A B
    ⊢ Eq ((↑(gelfandStarTransform B)).comp φ) ((ContinuousMap.compStarAlgHom' Comp …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
Consider the contravariant functors between compact Hausdorff spaces and commutative unital
C⋆algebras `F : Cpct → CommCStarAlg := X ↦ C(X, ℂ)` and
`G : CommCStarAlg → Cpct := A → characterSpace ℂ A` whose actions on morphisms are given by
`WeakDual.CharacterSpace.compContinuousMap` and `ContinuousMap.compStarAlgHom'`, respectively.

Then `η : id → G ∘ F := WeakDual.CharacterSpace.homeoEval` is a natural isomorphism implementing
(half of) the duality between these categories. That is, for compact Hausdorff spaces `X` and `Y`,
`f : C(X, Y)` the following diagram commutes:

```
X  --- η X ---> characterSpace ℂ C(X, ℂ)

|                     |

f                  (G ∘ F) f

|                     |
V                     V

Y  --- η Y ---> characterSpace ℂ C(Y, ℂ)
```
-/
lemma WeakDual.CharacterSpace.homeoEval_naturality {X Y 𝕜 : Type*} [RCLike 𝕜] [TopologicalSpace X]
    [CompactSpace X] [T2Space X] [TopologicalSpace Y] [CompactSpace Y] [T2Space Y] (f : C(X, Y)) :
    (homeoEval Y 𝕜 : C(_, _)).comp f =
      (f.compStarAlgHom' 𝕜 𝕜 |> compContinuousMap).comp (homeoEval X 𝕜 : C(_, _)) :=
  rfl

