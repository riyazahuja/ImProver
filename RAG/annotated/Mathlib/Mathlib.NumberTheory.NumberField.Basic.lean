/-- A number field is a field which has characteristic zero and is finite
dimensional over ℚ. -/
@[stacks 09GA]
class NumberField (K : Type*) [Field K] : Prop where
  [to_charZero : CharZero K]
  [to_finiteDimensional : FiniteDimensional ℚ K]


/-- `ℤ` with its usual ring structure is not a field. -/
theorem Int.not_isField : ¬IsField ℤ := fun h =>
  Int.not_even_one <|
                                                   /-
                                                     h : IsField Int
                                                     a : Int
                                                     ⊢ Eq (HMul.hMul 2 a) 1 → Eq 1 (HAdd.hAdd a a)
                                                   -/
    (h.mul_inv_cancel two_ne_zero).imp fun a => by rw [← two_mul]; exact Eq.symm
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


protected theorem isAlgebraic [NumberField K] : Algebra.IsAlgebraic ℚ K :=
  Algebra.IsAlgebraic.of_finite _ _


instance [NumberField K] [NumberField L] [Algebra K L] : FiniteDimensional K L :=
  Module.Finite.of_restrictScalars_finite ℚ K L


/-- A finite extension of a number field is a number field. -/
theorem of_module_finite [NumberField K] [Algebra K L] [Module.Finite K L] : NumberField L where
  to_charZero := charZero_of_injective_algebraMap (algebraMap K L).injective
  to_finiteDimensional :=
    letI := charZero_of_injective_algebraMap (algebraMap K L).injective
    Module.Finite.trans K L


variable {K} {L} in
instance of_intermediateField [NumberField K] [NumberField L] [Algebra K L]
    (E : IntermediateField K L) : NumberField E :=
  of_module_finite K E


theorem of_tower [NumberField K] [NumberField L] [Algebra K L] (E : Type*) [Field E]
    [Algebra K E] [Algebra E L] [IsScalarTower K E L] : NumberField E :=
  letI := Module.Finite.left K E L
  of_module_finite K E


/-- The ring of integers (or number ring) corresponding to a number field
is the integral closure of ℤ in the number field.

This is defined as its own type, rather than a `Subalgebra`, for performance reasons:
looking for instances of the form `SMul (RingOfIntegers _) (RingOfIntegers _)` makes
much more effective use of the discrimination tree than instances of the form
`SMul (Subtype _) (Subtype _)`.
The drawback is we have to copy over instances manually.
-/
def RingOfIntegers : Type _ :=
  integralClosure ℤ K


@[inherit_doc] scoped notation "𝓞" => NumberField.RingOfIntegers


instance : CommRing (𝓞 K) :=
  inferInstanceAs (CommRing (integralClosure _ _))

instance : IsDomain (𝓞 K) :=
  inferInstanceAs (IsDomain (integralClosure _ _))

instance [NumberField K] : CharZero (𝓞 K) :=
  inferInstanceAs (CharZero (integralClosure _ _))

instance : Algebra (𝓞 K) K :=
  inferInstanceAs (Algebra (integralClosure _ _) _)

instance : NoZeroSMulDivisors (𝓞 K) K :=
  inferInstanceAs (NoZeroSMulDivisors (integralClosure _ _) _)

instance : Nontrivial (𝓞 K) :=
  inferInstanceAs (Nontrivial (integralClosure _ _))

instance {L : Type*} [Ring L] [Algebra K L] : Algebra (𝓞 K) L :=
  inferInstanceAs (Algebra (integralClosure _ _) L)

instance {L : Type*} [Ring L] [Algebra K L] : IsScalarTower (𝓞 K) K L :=
  inferInstanceAs (IsScalarTower (integralClosure _ _) K L)


/-- The canonical coercion from `𝓞 K` to `K`. -/
@[coe]
abbrev val (x : 𝓞 K) : K := algebraMap _ _ x


/-- This instance has to be `CoeHead` because we only want to apply it from `𝓞 K` to `K`. -/
instance : CoeHead (𝓞 K) K := ⟨val⟩


lemma coe_eq_algebraMap (x : 𝓞 K) : (x : K) = algebraMap _ _ x := rfl


@[ext] theorem ext {x y : 𝓞 K} (h : (x : K) = (y : K)) : x = y :=
  Subtype.ext h


@[norm_cast]
theorem eq_iff {x y : 𝓞 K} : (x : K) = (y : K) ↔ x = y :=
  NumberField.RingOfIntegers.ext_iff.symm


@[simp] lemma map_mk (x : K) (hx) : algebraMap (𝓞 K) K ⟨x, hx⟩ = x := rfl


lemma coe_mk {x : K} (hx) : ((⟨x, hx⟩ : 𝓞 K) : K) = x := rfl


                                                                           /-
                                                                             K : Type u_1
                                                                             inst✝ : Field K
                                                                             x y : K
                                                                             hx : Membership.mem (integralClosure Int K) x
                                                                             hy : Membership.mem (integralClosure Int K) y
                                                                             ⊢ Iff (Eq ⟨x, hx⟩ ⟨y, hy⟩) (Eq x y)
                                                                           -/
lemma mk_eq_mk (x y : K) (hx hy) : (⟨x, hx⟩ : 𝓞 K) = ⟨y, hy⟩ ↔ x = y := by simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp] lemma mk_one : (⟨1, one_mem _⟩ : 𝓞 K) = 1 :=
  rfl


@[simp] lemma mk_zero : (⟨0, zero_mem _⟩ : 𝓞 K) = 0 :=
  rfl
-- TODO: these lemmas don't seem to fire?

@[simp] lemma mk_add_mk (x y : K) (hx hy) : (⟨x, hx⟩ : 𝓞 K) + ⟨y, hy⟩ = ⟨x + y, add_mem hx hy⟩ :=
  rfl


@[simp] lemma mk_mul_mk (x y : K) (hx hy) : (⟨x, hx⟩ : 𝓞 K) * ⟨y, hy⟩ = ⟨x * y, mul_mem hx hy⟩ :=
  rfl


@[simp] lemma mk_sub_mk (x y : K) (hx hy) : (⟨x, hx⟩ : 𝓞 K) - ⟨y, hy⟩ = ⟨x - y, sub_mem hx hy⟩ :=
  rfl


@[simp] lemma neg_mk (x : K) (hx) : (-⟨x, hx⟩ : 𝓞 K) = ⟨-x, neg_mem hx⟩ :=
  rfl


/-- The ring homomorphism `(𝓞 K) →+* (𝓞 L)` given by restricting a ring homomorphism
  `f : K →+* L` to `𝓞 K`. -/
def mapRingHom {K L F : Type*} [Field K] [Field L] [FunLike F K L]
    [RingHomClass F K L] (f : F) : (𝓞 K) →+* (𝓞 L) where
  toFun k := ⟨f k.val, map_isIntegral_int f k.2⟩
                  /-
                    K✝ : Type u_1
                    L✝ : Type u_2
                    inst✝⁵ : Field K✝
                    inst✝⁴ : Field L✝
                    K : Type u_3
                    L : Type u_4
                    F : Type u_5
                    inst✝³ : Field K
                    inst✝² : Field L
                    inst✝¹ : FunLike F K L
                    inst✝ : RingHomClass F K L
                    f : F
                    ⊢ Eq ((↑{ toFun := fun k => ⟨f ↑k, ⋯⟩, map_one' := ⋯, map_mul' := ⋯ }).toFun 0 …
                  -/
                 /-
                   K✝ : Type u_1
                   L✝ : Type u_2
                   inst✝⁵ : Field K✝
                   inst✝⁴ : Field L✝
                   K : Type u_3
                   L : Type u_4
                   F : Type u_5
                   inst✝³ : Field K
                   inst✝² : Field L
                   inst✝¹ : FunLike F K L
                   inst✝ : RingHomClass F K L
                   f : F
                   ⊢ Eq ((fun k => ⟨f ↑k, ⋯⟩) 1) 1
                 -/
  map_zero' := by ext; simp only [map_mk, map_zero]
                      /-
                        🎉 no goals
                      -/
                       /-
                         🎉 no goals
                       -/
                     /-
                       K✝ : Type u_1
                       L✝ : Type u_2
                       inst✝⁵ : Field K✝
                       inst✝⁴ : Field L✝
                       K : Type u_3
                       L : Type u_4
                       F : Type u_5
                       inst✝³ : Field K
                       inst✝² : Field L
                       inst✝¹ : FunLike F K L
                       inst✝ : RingHomClass F K L
                       f : F
                       x y : NumberField.RingOfIntegers K
                       ⊢ Eq ({ toFun := fun k => ⟨f ↑k, ⋯⟩, map_one' := ⋯ }.toFun (HMul.hMul x y)) (H …
                     -/
  map_one' := by ext; simp only [map_mk, map_one]
                          /-
                            🎉 no goals
                          -/
                    /-
                      K✝ : Type u_1
                      L✝ : Type u_2
                      inst✝⁵ : Field K✝
                      inst✝⁴ : Field L✝
                      K : Type u_3
                      L : Type u_4
                      F : Type u_5
                      inst✝³ : Field K
                      inst✝² : Field L
                      inst✝¹ : FunLike F K L
                      inst✝ : RingHomClass F K L
                      f : F
                      x y : NumberField.RingOfIntegers K
                      ⊢ Eq ((↑{ toFun := fun k => ⟨f ↑k, ⋯⟩, map_one' := ⋯, map_mul' := ⋯ }).toFun ( …
                    -/
  map_add' x y:= by ext; simp only [map_mk, map_add]
                         /-
                           🎉 no goals
                         -/
  map_mul' x y := by ext; simp only [map_mk, map_mul]


/-- The ring isomorphsim `(𝓞 K) ≃+* (𝓞 L)` given by restricting
  a ring isomorphsim `e : K ≃+* L` to `𝓞 K`. -/
def mapRingEquiv {K L E : Type*} [Field K] [Field L] [EquivLike E K L]
    [RingEquivClass E K L] (e : E) : (𝓞 K) ≃+* (𝓞 L) :=
  RingEquiv.ofRingHom (mapRingHom e) (mapRingHom (e : K ≃+* L).symm)
    (RingHom.ext fun x => ext (EquivLike.right_inv e x.1))
      (RingHom.ext fun x => ext (EquivLike.left_inv e x.1))


/-- Given an algebra between two fields, create an algebra between their two rings of integers. -/
instance inst_ringOfIntegersAlgebra [Algebra K L] : Algebra (𝓞 K) (𝓞 L) :=
  (RingOfIntegers.mapRingHom (algebraMap K L)).toAlgebra

-- diamond at `reducible_and_instances` https://github.com/leanprover-community/mathlib4/issues/10906

/-- The algebra homomorphism `(𝓞 K) →ₐ[𝓞 k] (𝓞 L)` given by restricting a algebra homomorphism
  `f : K →ₐ[k] L` to `𝓞 K`. -/
def mapAlgHom {k K L F : Type*} [Field k] [Field K] [Field L] [Algebra k K]
    [Algebra k L] [FunLike F K L] [AlgHomClass F k K L] (f : F) : (𝓞 K) →ₐ[𝓞 k] (𝓞 L) where
  toRingHom := mapRingHom f
  commutes' x := SetCoe.ext (AlgHomClass.commutes ((f : K →ₐ[k] L).restrictScalars (𝓞 k)) x)


/-- The isomorphism of algebras `(𝓞 K) ≃ₐ[𝓞 k] (𝓞 L)` given by restricting
  an isomorphism of algebras `e : K ≃ₐ[k] L` to `𝓞 K`. -/
def mapAlgEquiv {k K L E : Type*} [Field k] [Field K] [Field L] [Algebra k K]
    [Algebra k L] [EquivLike E K L] [AlgEquivClass E k K L] (e : E) : (𝓞 K) ≃ₐ[𝓞 k] (𝓞 L) :=
  AlgEquiv.ofAlgHom (mapAlgHom e) (mapAlgHom (e : K ≃ₐ[k] L).symm)
    (AlgHom.ext fun x => ext (EquivLike.right_inv e x.1))
      (AlgHom.ext fun x => ext (EquivLike.left_inv e x.1))


instance inst_isScalarTower (k K L : Type*) [Field k] [Field K] [Field L]
    [Algebra k K] [Algebra k L] [Algebra K L] [IsScalarTower k K L] :
    IsScalarTower (𝓞 k) (𝓞 K) (𝓞 L) :=
  IsScalarTower.of_algHom (mapAlgHom (IsScalarTower.toAlgHom k K L))


/-- The canonical map from `𝓞 K` to `K` is injective.

This is a convenient abbreviation for `NoZeroSMulDivisors.algebraMap_injective`.
-/
lemma coe_injective : Function.Injective (algebraMap (𝓞 K) K) :=
  NoZeroSMulDivisors.algebraMap_injective _ _


/-- The canonical map from `𝓞 K` to `K` is injective.

This is a convenient abbreviation for `map_eq_zero_iff` applied to
`NoZeroSMulDivisors.algebraMap_injective`.
-/
lemma coe_eq_zero_iff {x : 𝓞 K} : algebraMap _ K x = 0 ↔ x = 0 :=
  map_eq_zero_iff _ coe_injective


/-- The canonical map from `𝓞 K` to `K` is injective.

This is a convenient abbreviation for `map_ne_zero_iff` applied to
`NoZeroSMulDivisors.algebraMap_injective`.
-/
lemma coe_ne_zero_iff {x : 𝓞 K} : algebraMap _ K x ≠ 0 ↔ x ≠ 0 :=
  map_ne_zero_iff _ coe_injective


theorem isIntegral_coe (x : 𝓞 K) : IsIntegral ℤ (algebraMap _ K x) :=
  x.2


theorem isIntegral (x : 𝓞 K) : IsIntegral ℤ x := by
  /-
    K : Type u_1
    inst✝ : Field K
    x : NumberField.RingOfIntegers K
    ⊢ IsIntegral Int x
  -/
  obtain ⟨P, hPm, hP⟩ := x.isIntegral_coe
  /-
    case intro.intro
    K : Type u_1
    inst✝ : Field K
    x : NumberField.RingOfIntegers K
    P : Polynomial Int
    hPm : P.Monic
    hP : Eq (Polynomial.eval₂ (algebraMap Int K) ((algebraMap (NumberField.RingOfI …
    ⊢ IsIntegral Int x
  -/
  refine ⟨P, hPm, ?_⟩
  /-
    case intro.intro
    K : Type u_1
    inst✝ : Field K
    x : NumberField.RingOfIntegers K
    P : Polynomial Int
    hPm : P.Monic
    hP : Eq (Polynomial.eval₂ (algebraMap Int K) ((algebraMap (NumberField.RingOfI …
    ⊢ Eq (Polynomial.eval₂ (algebraMap Int (NumberField.RingOfIntegers K)) x P) 0
  -/
  rwa [IsScalarTower.algebraMap_eq (S := 𝓞 K), ← Polynomial.hom_eval₂, coe_eq_zero_iff] at hP
  /-
    🎉 no goals
  -/


instance [NumberField K] : IsFractionRing (𝓞 K) K :=
  integralClosure.isFractionRing_of_finite_extension ℚ _


instance : IsIntegralClosure (𝓞 K) ℤ K :=
  integralClosure.isIntegralClosure _ _


instance : Algebra.IsIntegral ℤ (𝓞 K) :=
  IsIntegralClosure.isIntegral_algebra ℤ K


instance [NumberField K] : IsIntegrallyClosed (𝓞 K) :=
  integralClosure.isIntegrallyClosedOfFiniteExtension ℚ


/-- The ring of integers of `K` are equivalent to any integral closure of `ℤ` in `K` -/
protected noncomputable def equiv (R : Type*) [CommRing R] [Algebra R K]
    [IsIntegralClosure R ℤ K] : 𝓞 K ≃+* R :=
  (IsIntegralClosure.equiv ℤ R K _).symm.toRingEquiv


instance [CharZero K] : CharZero (𝓞 K) :=
  CharZero.of_module _ K


instance : IsNoetherian ℤ (𝓞 K) :=
  IsIntegralClosure.isNoetherian _ ℚ K _


/-- The ring of integers of a number field is not a field. -/
theorem not_isField : ¬IsField (𝓞 K) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Not (IsField (NumberField.RingOfIntegers K))
  -/
  have h_inj : Function.Injective (algebraMap ℤ (𝓞 K)) := RingHom.injective_int (algebraMap ℤ (𝓞 K))
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h_inj : Function.Injective ⇑(algebraMap Int (NumberField.RingOfIntegers K))
    ⊢ Not (IsField (NumberField.RingOfIntegers K))
  -/
  intro hf
  exact Int.not_isField
    (((IsIntegralClosure.isIntegral_algebra ℤ K).isField_iff_isField h_inj).mpr hf)


instance : IsDedekindDomain (𝓞 K) :=
  IsIntegralClosure.isDedekindDomain ℤ ℚ K _


instance : Free ℤ (𝓞 K) :=
  IsIntegralClosure.module_free ℤ ℚ K (𝓞 K)


instance : IsLocalization (Algebra.algebraMapSubmonoid (𝓞 K) ℤ⁰) K :=
  IsIntegralClosure.isLocalization_of_isSeparable ℤ ℚ K (𝓞 K)


/-- A ℤ-basis of the ring of integers of `K`. -/
noncomputable def basis : Basis (Free.ChooseBasisIndex ℤ (𝓞 K)) ℤ (𝓞 K) :=
  Free.chooseBasis ℤ (𝓞 K)


/-- Given `f : M → K` such that `∀ x, IsIntegral ℤ (f x)`, the corresponding function
`M → 𝓞 K`. -/
def restrict (f : M → K) (h : ∀ x, IsIntegral ℤ (f x)) (x : M) : 𝓞 K :=
  ⟨f x, h x⟩


/-- Given `f : M →+ K` such that `∀ x, IsIntegral ℤ (f x)`, the corresponding function
`M →+ 𝓞 K`. -/
def restrict_addMonoidHom [AddZeroClass M] (f : M →+ K) (h : ∀ x, IsIntegral ℤ (f x)) :
    M →+ 𝓞 K where
  toFun := restrict f h
                  /-
                    K : Type u_1
                    L : Type u_2
                    inst✝³ : Field K
                    inst✝² : Field L
                    inst✝¹ : NumberField K
                    M : Type u_3
                    inst✝ : AddZeroClass M
                    f : AddMonoidHom M K
                    h : ∀ (x : M), IsIntegral Int (f x)
                    ⊢ Eq (NumberField.RingOfIntegers.restrict (⇑f) h 0) 0
                  -/
  map_zero' := by simp only [restrict, map_zero, mk_zero]
                  /-
                    🎉 no goals
                  -/
                     /-
                       K : Type u_1
                       L : Type u_2
                       inst✝³ : Field K
                       inst✝² : Field L
                       inst✝¹ : NumberField K
                       M : Type u_3
                       inst✝ : AddZeroClass M
                       f : AddMonoidHom M K
                       h : ∀ (x : M), IsIntegral Int (f x)
                       x y : M
                       ⊢ Eq ({ toFun := NumberField.RingOfIntegers.restrict (⇑f) h, map_zero' := ⋯ }. …
                     -/
  map_add' x y := by simp only [restrict, map_add, mk_add_mk _]
                     /-
                       🎉 no goals
                     -/


/-- Given `f : M →* K` such that `∀ x, IsIntegral ℤ (f x)`, the corresponding function
`M →* 𝓞 K`. -/
@[to_additive existing] -- TODO: why doesn't it figure this out by itself?
def restrict_monoidHom [MulOneClass M] (f : M →* K) (h : ∀ x, IsIntegral ℤ (f x)) : M →* 𝓞 K where
  toFun := restrict f h
                 /-
                   K : Type u_1
                   L : Type u_2
                   inst✝³ : Field K
                   inst✝² : Field L
                   inst✝¹ : NumberField K
                   M : Type u_3
                   inst✝ : MulOneClass M
                   f : MonoidHom M K
                   h : ∀ (x : M), IsIntegral Int (f x)
                   ⊢ Eq (NumberField.RingOfIntegers.restrict (⇑f) h 1) 1
                 -/
  map_one' := by simp only [restrict, map_one, mk_one]
                 /-
                   🎉 no goals
                 -/
                     /-
                       K : Type u_1
                       L : Type u_2
                       inst✝³ : Field K
                       inst✝² : Field L
                       inst✝¹ : NumberField K
                       M : Type u_3
                       inst✝ : MulOneClass M
                       f : MonoidHom M K
                       h : ∀ (x : M), IsIntegral Int (f x)
                       x y : M
                       ⊢ Eq ({ toFun := NumberField.RingOfIntegers.restrict (⇑f) h, map_one' := ⋯ }.t …
                     -/
  map_mul' x y := by simp only [restrict, map_mul, mk_mul_mk _]
                     /-
                       🎉 no goals
                     -/


instance : IsScalarTower (𝓞 K) (𝓞 L) L :=
  IsScalarTower.of_algebraMap_eq' rfl


instance : IsIntegralClosure (𝓞 L) (𝓞 K) L :=
  IsIntegralClosure.tower_top (R := ℤ)


/-- The ring of integers of `L` is isomorphic to any integral closure of `𝓞 K` in `L` -/
protected noncomputable def algEquiv (R : Type*) [CommRing R] [Algebra (𝓞 K) R] [Algebra R L]
    [IsScalarTower (𝓞 K) R L] [IsIntegralClosure R (𝓞 K) L] : 𝓞 L ≃ₐ[𝓞 K] R :=
  (IsIntegralClosure.equiv (𝓞 K) R L _).symm


/-- Any extension between ring of integers is integral. -/
instance extension_algebra_isIntegral : Algebra.IsIntegral (𝓞 K) (𝓞 L) :=
  IsIntegralClosure.isIntegral_algebra (𝓞 K) L


/-- Any extension between ring of integers of number fields is noetherian. -/
instance extension_isNoetherian [NumberField K] [NumberField L] : IsNoetherian (𝓞 K) (𝓞 L) :=
  IsIntegralClosure.isNoetherian (𝓞 K) K L (𝓞 L)


/-- The kernel of the algebraMap between ring of integers is `⊥`. -/
theorem ker_algebraMap_eq_bot : RingHom.ker (algebraMap (𝓞 K) (𝓞 L)) = ⊥ :=
  (RingHom.ker_eq_bot_iff_eq_zero (algebraMap (𝓞 K) (𝓞 L))).mpr <| fun x hx => by
  /-
    K : Type u_4
    L : Type u_5
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : NumberField.RingOfIntegers K
    hx : Eq ((algebraMap (NumberField.RingOfIntegers K) (NumberField.RingOfInteger …
    ⊢ Eq x 0
  -/
  have h : (algebraMap K L) x = (algebraMap (𝓞 K) (𝓞 L)) x := rfl
  /-
    K : Type u_4
    L : Type u_5
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : NumberField.RingOfIntegers K
    hx : Eq ((algebraMap (NumberField.RingOfIntegers K) (NumberField.RingOfInteger …
    h : Eq ((algebraMap K L) ↑x) ↑((algebraMap (NumberField.RingOfIntegers K) (Num …
    ⊢ Eq x 0
  -/
  simp only [hx, map_zero, map_eq_zero, RingOfIntegers.coe_eq_zero_iff] at h
  /-
    K : Type u_4
    L : Type u_5
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : NumberField.RingOfIntegers K
    hx : Eq ((algebraMap (NumberField.RingOfIntegers K) (NumberField.RingOfInteger …
    h : Eq x 0
    ⊢ Eq x 0
  -/
  exact h
  /-
    🎉 no goals
  -/


/-- The algebraMap between ring of integers is injective. -/
theorem algebraMap.injective : Function.Injective (algebraMap (𝓞 K) (𝓞 L)) :=
  (RingHom.injective_iff_ker_eq_bot (algebraMap (𝓞 K) (𝓞 L))).mpr (ker_algebraMap_eq_bot K L)


instance : NoZeroSMulDivisors (𝓞 K) (𝓞 L) :=
  NoZeroSMulDivisors.of_algebraMap_injective (algebraMap.injective K L)


instance : NoZeroSMulDivisors (𝓞 K) L :=
  NoZeroSMulDivisors.trans (𝓞 K) (𝓞 L) L


/-- A basis of `K` over `ℚ` that is also a basis of `𝓞 K` over `ℤ`. -/
noncomputable def integralBasis : Basis (Free.ChooseBasisIndex ℤ (𝓞 K)) ℚ K :=
  Basis.localizationLocalization ℚ (nonZeroDivisors ℤ) K (RingOfIntegers.basis K)


@[simp]
theorem integralBasis_apply (i : Free.ChooseBasisIndex ℤ (𝓞 K)) :
    integralBasis K i = algebraMap (𝓞 K) K (RingOfIntegers.basis K i) :=
  Basis.localizationLocalization_apply ℚ (nonZeroDivisors ℤ) K (RingOfIntegers.basis K) i


@[simp]
theorem integralBasis_repr_apply (x : (𝓞 K)) (i : Free.ChooseBasisIndex ℤ (𝓞 K)) :
    (integralBasis K).repr (algebraMap _ _ x) i =
      (algebraMap ℤ ℚ) ((RingOfIntegers.basis K).repr x i) :=
  Basis.localizationLocalization_repr_algebraMap ℚ (nonZeroDivisors ℤ) K _ x i


theorem mem_span_integralBasis {x : K} :
    x ∈ Submodule.span ℤ (Set.range (integralBasis K)) ↔ x ∈ (algebraMap (𝓞 K) K).range := by
  rw [integralBasis, Basis.localizationLocalization_span, LinearMap.mem_range,
      IsScalarTower.coe_toAlgHom', RingHom.mem_range]


theorem RingOfIntegers.rank : Module.finrank ℤ (𝓞 K) = Module.finrank ℚ K :=
  IsIntegralClosure.rank ℤ ℚ K (𝓞 K)


instance numberField : NumberField ℚ where
  to_charZero := inferInstance
  to_finiteDimensional := by
  -- The vector space structure of `ℚ` over itself can arise in multiple ways:
  -- all fields are vector spaces over themselves (used in `Rat.finiteDimensional`)
  -- all char 0 fields have a canonical embedding of `ℚ` (used in `NumberField`).
  -- Show that these coincide:
    /-
      ⊢ FiniteDimensional Rat Rat
    -/
    convert (inferInstance : FiniteDimensional ℚ ℚ)
    /-
      🎉 no goals
    -/


/-- The ring of integers of `ℚ` as a number field is just `ℤ`. -/
noncomputable def ringOfIntegersEquiv : 𝓞 ℚ ≃+* ℤ :=
  RingOfIntegers.equiv ℤ


/-- The quotient of `ℚ[X]` by the ideal generated by an irreducible polynomial of `ℚ[X]`
is a number field. -/
instance {f : Polynomial ℚ} [hf : Fact (Irreducible f)] : NumberField (AdjoinRoot f) where
  to_charZero := charZero_of_injective_algebraMap (algebraMap ℚ _).injective
                             /-
                               f : Polynomial Rat
                               hf : Fact (Irreducible f)
                               ⊢ FiniteDimensional Rat (AdjoinRoot f)
                             -/
  to_finiteDimensional := by convert (AdjoinRoot.powerBasis hf.out.ne_zero).finite
                             /-
                               🎉 no goals
                             -/


