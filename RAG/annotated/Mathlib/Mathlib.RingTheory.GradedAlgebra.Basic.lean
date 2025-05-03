/-- An internally-graded `R`-algebra `A` is one that can be decomposed into a collection
of `Submodule R A`s indexed by `ι` such that the canonical map `A → ⨁ i, 𝒜 i` is bijective and
respects multiplication, i.e. the product of an element of degree `i` and an element of degree `j`
is an element of degree `i + j`.

Note that the fact that `A` is internally-graded, `GradedAlgebra 𝒜`, implies an externally-graded
algebra structure `DirectSum.GAlgebra R (fun i ↦ ↥(𝒜 i))`, which in turn makes available an
`Algebra R (⨁ i, 𝒜 i)` instance.
-/
class GradedRing (𝒜 : ι → σ) extends SetLike.GradedMonoid 𝒜, DirectSum.Decomposition 𝒜


/-- If `A` is graded by `ι` with degree `i` component `𝒜 i`, then it is isomorphic as
a ring to a direct sum of components. -/
def decomposeRingEquiv : A ≃+* ⨁ i, 𝒜 i :=
  RingEquiv.symm
    { (decomposeAddEquiv 𝒜).symm with
      map_mul' := (coeRingHom 𝒜).map_mul }


@[simp]
theorem decompose_one : decompose 𝒜 (1 : A) = 1 :=
  map_one (decomposeRingEquiv 𝒜)


@[simp]
theorem decompose_symm_one : (decompose 𝒜).symm 1 = (1 : A) :=
  map_one (decomposeRingEquiv 𝒜).symm


@[simp]
theorem decompose_mul (x y : A) : decompose 𝒜 (x * y) = decompose 𝒜 x * decompose 𝒜 y :=
  map_mul (decomposeRingEquiv 𝒜) x y


@[simp]
theorem decompose_symm_mul (x y : ⨁ i, 𝒜 i) :
    (decompose 𝒜).symm (x * y) = (decompose 𝒜).symm x * (decompose 𝒜).symm y :=
  map_mul (decomposeRingEquiv 𝒜).symm x y


/-- The projection maps of a graded ring -/
def GradedRing.proj (i : ι) : A →+ A :=
  (AddSubmonoidClass.subtype (𝒜 i)).comp <|
    (DFinsupp.evalAddMonoidHom i).comp <|
      RingHom.toAddMonoidHom <| RingEquiv.toRingHom <| DirectSum.decomposeRingEquiv 𝒜


@[simp]
theorem GradedRing.proj_apply (i : ι) (r : A) :
    GradedRing.proj 𝒜 i r = (decompose 𝒜 r : ⨁ i, 𝒜 i) i :=
  rfl


theorem GradedRing.proj_recompose (a : ⨁ i, 𝒜 i) (i : ι) :
    GradedRing.proj 𝒜 i ((decompose 𝒜).symm a) = (decompose 𝒜).symm (DirectSum.of _ i (a i)) := by
  /-
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : AddMonoid ι
    inst✝³ : Semiring A
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    a : DirectSum ι fun i => Subtype fun x => Membership.mem (𝒜 i) x
    i : ι
    ⊢ Eq ((GradedRing.proj 𝒜 i) ((DirectSum.decompose 𝒜).symm a)) ((DirectSum.deco …
  -/
  rw [GradedRing.proj_apply, decompose_symm_of, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


theorem GradedRing.mem_support_iff [∀ (i) (x : 𝒜 i), Decidable (x ≠ 0)] (r : A) (i : ι) :
    i ∈ (decompose 𝒜 r).support ↔ GradedRing.proj 𝒜 i r ≠ 0 :=
  DFinsupp.mem_support_iff.trans ZeroMemClass.coe_eq_zero.not.symm


theorem coe_decompose_mul_add_of_left_mem [AddLeftCancelMonoid ι] [GradedRing 𝒜] {a b : A}
    (a_mem : a ∈ 𝒜 i) : (decompose 𝒜 (a * b) (i + j) : A) = a * decompose 𝒜 b j := by
  /-
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Semiring A
    inst✝³ : SetLike σ A
    inst✝² : AddSubmonoidClass σ A
    𝒜 : ι → σ
    i j : ι
    inst✝¹ : AddLeftCancelMonoid ι
    inst✝ : GradedRing 𝒜
    a b : A
    a_mem : Membership.mem (𝒜 i) a
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a b)) (HAdd.hAdd i j))) (HMul.hMul …
  -/
  lift a to 𝒜 i using a_mem
  /-
    case intro
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Semiring A
    inst✝³ : SetLike σ A
    inst✝² : AddSubmonoidClass σ A
    𝒜 : ι → σ
    i j : ι
    inst✝¹ : AddLeftCancelMonoid ι
    inst✝ : GradedRing 𝒜
    b : A
    a : Subtype fun x => Membership.mem (𝒜 i) x
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul (↑a) b)) (HAdd.hAdd i j))) (HMul.h …
  -/
  rw [decompose_mul, decompose_coe, coe_of_mul_apply_add]
  /-
    🎉 no goals
  -/


theorem coe_decompose_mul_add_of_right_mem [AddRightCancelMonoid ι] [GradedRing 𝒜] {a b : A}
    (b_mem : b ∈ 𝒜 j) : (decompose 𝒜 (a * b) (i + j) : A) = decompose 𝒜 a i * b := by
  /-
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Semiring A
    inst✝³ : SetLike σ A
    inst✝² : AddSubmonoidClass σ A
    𝒜 : ι → σ
    i j : ι
    inst✝¹ : AddRightCancelMonoid ι
    inst✝ : GradedRing 𝒜
    a b : A
    b_mem : Membership.mem (𝒜 j) b
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a b)) (HAdd.hAdd i j))) (HMul.hMul …
  -/
  lift b to 𝒜 j using b_mem
  /-
    case intro
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Semiring A
    inst✝³ : SetLike σ A
    inst✝² : AddSubmonoidClass σ A
    𝒜 : ι → σ
    i j : ι
    inst✝¹ : AddRightCancelMonoid ι
    inst✝ : GradedRing 𝒜
    a : A
    b : Subtype fun x => Membership.mem (𝒜 j) x
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a ↑b)) (HAdd.hAdd i j))) (HMul.hMu …
  -/
  rw [decompose_mul, decompose_coe, coe_mul_of_apply_add]
  /-
    🎉 no goals
  -/


theorem decompose_mul_add_left [AddLeftCancelMonoid ι] [GradedRing 𝒜] (a : 𝒜 i) {b : A} :
    decompose 𝒜 (↑a * b) (i + j) =
      @GradedMonoid.GMul.mul ι (fun i => 𝒜 i) _ _ _ _ a (decompose 𝒜 b j) :=
  Subtype.ext <| coe_decompose_mul_add_of_left_mem 𝒜 a.2


theorem decompose_mul_add_right [AddRightCancelMonoid ι] [GradedRing 𝒜] {a : A} (b : 𝒜 j) :
    decompose 𝒜 (a * ↑b) (i + j) =
      @GradedMonoid.GMul.mul ι (fun i => 𝒜 i) _ _ _ _ (decompose 𝒜 a i) b :=
  Subtype.ext <| coe_decompose_mul_add_of_right_mem 𝒜 b.2


/-- A special case of `GradedRing` with `σ = Submodule R A`. This is useful both because it
can avoid typeclass search, and because it provides a more concise name. -/
abbrev GradedAlgebra :=
  GradedRing 𝒜


/-- A helper to construct a `GradedAlgebra` when the `SetLike.GradedMonoid` structure is already
available. This makes the `left_inv` condition easier to prove, and phrases the `right_inv`
condition in a way that allows custom `@[ext]` lemmas to apply.

See note [reducible non-instances]. -/
abbrev GradedAlgebra.ofAlgHom [SetLike.GradedMonoid 𝒜] (decompose : A →ₐ[R] ⨁ i, 𝒜 i)
    (right_inv : (DirectSum.coeAlgHom 𝒜).comp decompose = AlgHom.id R A)
    (left_inv : ∀ i (x : 𝒜 i), decompose (x : A) = DirectSum.of (fun i => ↥(𝒜 i)) i x) :
    GradedAlgebra 𝒜 where
  decompose' := decompose
  left_inv := AlgHom.congr_fun right_inv
  right_inv := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      σ : Type u_4
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : AddMonoid ι
      inst✝³ : CommSemiring R
      inst✝² : Semiring A
      inst✝¹ : Algebra R A
      𝒜 : ι → Submodule R A
      inst✝ : SetLike.GradedMonoid 𝒜
      decompose : AlgHom R A (DirectSum ι fun i => Subtype fun x => Membership.mem ( …
      right_inv : Eq ((DirectSum.coeAlgHom 𝒜).comp decompose) (AlgHom.id R A)
      left_inv : ∀ (i : ι) (x : Subtype fun x => Membership.mem (𝒜 i) x), Eq (decomp …
      ⊢ Function.RightInverse ⇑(DirectSum.coeAddMonoidHom 𝒜) ⇑decompose
    -/
    suffices decompose.comp (DirectSum.coeAlgHom 𝒜) = AlgHom.id _ _ from AlgHom.congr_fun this
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      σ : Type u_4
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : AddMonoid ι
      inst✝³ : CommSemiring R
      inst✝² : Semiring A
      inst✝¹ : Algebra R A
      𝒜 : ι → Submodule R A
      inst✝ : SetLike.GradedMonoid 𝒜
      decompose : AlgHom R A (DirectSum ι fun i => Subtype fun x => Membership.mem ( …
      right_inv : Eq ((DirectSum.coeAlgHom 𝒜).comp decompose) (AlgHom.id R A)
      left_inv : ∀ (i : ι) (x : Subtype fun x => Membership.mem (𝒜 i) x), Eq (decomp …
      ⊢ Eq (decompose.comp (DirectSum.coeAlgHom 𝒜)) (AlgHom.id R (DirectSum ι fun i  …
    -/
    ext i x : 2
    /-
      case h.h
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      σ : Type u_4
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : AddMonoid ι
      inst✝³ : CommSemiring R
      inst✝² : Semiring A
      inst✝¹ : Algebra R A
      𝒜 : ι → Submodule R A
      inst✝ : SetLike.GradedMonoid 𝒜
      decompose : AlgHom R A (DirectSum ι fun i => Subtype fun x => Membership.mem ( …
      right_inv : Eq ((DirectSum.coeAlgHom 𝒜).comp decompose) (AlgHom.id R A)
      left_inv : ∀ (i : ι) (x : Subtype fun x => Membership.mem (𝒜 i) x), Eq (decomp …
      i : ι
      x : Subtype fun x => Membership.mem (𝒜 i) x
      ⊢ Eq (((decompose.comp (DirectSum.coeAlgHom 𝒜)).toLinearMap.comp (DirectSum.lo …
    -/
    exact (decompose.congr_arg <| DirectSum.coeAlgHom_of _ _ _).trans (left_inv i x)
    /-
      🎉 no goals
    -/


/-- If `A` is graded by `ι` with degree `i` component `𝒜 i`, then it is isomorphic as
an algebra to a direct sum of components. -/
-- Porting note: deleted [simps] and added the corresponding lemmas by hand
def decomposeAlgEquiv : A ≃ₐ[R] ⨁ i, 𝒜 i :=
  AlgEquiv.symm
    { (decomposeAddEquiv 𝒜).symm with
      map_mul' := map_mul (coeAlgHom 𝒜)
      commutes' := (coeAlgHom 𝒜).commutes }


@[simp]
lemma decomposeAlgEquiv_apply (a : A) :
    decomposeAlgEquiv 𝒜 a = decompose 𝒜 a := rfl


@[simp]
lemma decomposeAlgEquiv_symm_apply (a : ⨁ i, 𝒜 i) :
    (decomposeAlgEquiv 𝒜).symm a = (decompose 𝒜).symm a := rfl


@[simp]
lemma decompose_algebraMap (r : R) :
    decompose 𝒜 (algebraMap R A r) = algebraMap R (⨁ i, 𝒜 i) r :=
  (decomposeAlgEquiv 𝒜).commutes r


@[simp]
lemma decompose_symm_algebraMap (r : R) :
    (decompose 𝒜).symm (algebraMap R (⨁ i, 𝒜 i) r) = algebraMap R A r :=
  (decomposeAlgEquiv 𝒜).symm.commutes r


/-- The projection maps of graded algebra -/
def GradedAlgebra.proj (𝒜 : ι → Submodule R A) [GradedAlgebra 𝒜] (i : ι) : A →ₗ[R] A :=
  (𝒜 i).subtype.comp <| (DFinsupp.lapply i).comp <| (decomposeAlgEquiv 𝒜).toAlgHom.toLinearMap


@[simp]
theorem GradedAlgebra.proj_apply (i : ι) (r : A) :
    GradedAlgebra.proj 𝒜 i r = (decompose 𝒜 r : ⨁ i, 𝒜 i) i :=
  rfl


theorem GradedAlgebra.proj_recompose (a : ⨁ i, 𝒜 i) (i : ι) :
    GradedAlgebra.proj 𝒜 i ((decompose 𝒜).symm a) = (decompose 𝒜).symm (of _ i (a i)) := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : AddMonoid ι
    inst✝³ : CommSemiring R
    inst✝² : Semiring A
    inst✝¹ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    a : DirectSum ι fun i => Subtype fun x => Membership.mem (𝒜 i) x
    i : ι
    ⊢ Eq ((GradedAlgebra.proj 𝒜 i) ((DirectSum.decompose 𝒜).symm a)) ((DirectSum.d …
  -/
  rw [GradedAlgebra.proj_apply, decompose_symm_of, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


theorem GradedAlgebra.mem_support_iff [DecidableEq A] (r : A) (i : ι) :
    i ∈ (decompose 𝒜 r).support ↔ GradedAlgebra.proj 𝒜 i r ≠ 0 :=
  DFinsupp.mem_support_iff.trans Submodule.coe_eq_zero.not.symm


/-- If `A` is graded by a canonically ordered add monoid, then the projection map `x ↦ x₀` is a ring
homomorphism.
-/
@[simps]
def GradedRing.projZeroRingHom : A →+* A where
  toFun a := decompose 𝒜 a 0
  map_one' :=
    -- Porting note: qualified `one_mem`
    decompose_of_mem_same 𝒜 SetLike.GradedOne.one_mem
  map_zero' := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      σ : Type u_4
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : CanonicallyOrderedAddCommMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      ⊢ Eq ((↑{ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one' := ⋯, m …
    -/
    simp only -- Porting note: added
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      σ : Type u_4
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : CanonicallyOrderedAddCommMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      ⊢ Eq (↑(((DirectSum.decompose 𝒜) 0) 0)) 0
    -/
    rw [decompose_zero]
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      σ : Type u_4
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : CanonicallyOrderedAddCommMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      ⊢ Eq (↑(0 0)) 0
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_add' _ _ := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      σ : Type u_4
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : CanonicallyOrderedAddCommMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      ⊢ ∀ (x y : A), Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_o …
    -/
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      σ : Type u_4
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : CanonicallyOrderedAddCommMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      x✝¹ x✝ : A
      ⊢ Eq ((↑{ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one' := ⋯, m …
    -/
      /-
        case refine_1
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        σ : Type u_4
        inst✝⁵ : Semiring A
        inst✝⁴ : DecidableEq ι
        inst✝³ : CanonicallyOrderedAddCommMonoid ι
        inst✝² : SetLike σ A
        inst✝¹ : AddSubmonoidClass σ A
        𝒜 : ι → σ
        inst✝ : GradedRing 𝒜
        x : A
        ⊢ Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one' := ⋯ }.to …
      -/
    simp only -- Porting note: added
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        σ : Type u_4
        inst✝⁵ : Semiring A
        inst✝⁴ : DecidableEq ι
        inst✝³ : CanonicallyOrderedAddCommMonoid ι
        inst✝² : SetLike σ A
        inst✝¹ : AddSubmonoidClass σ A
        𝒜 : ι → σ
        inst✝ : GradedRing 𝒜
        ⊢ ∀ {i : ι} (m : Subtype fun x => Membership.mem (𝒜 i) x) (y : A), Eq ({ toFun …
      -/
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      σ : Type u_4
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : CanonicallyOrderedAddCommMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      x✝¹ x✝ : A
      ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HAdd.hAdd x✝¹ x✝)) 0)) (HAdd.hAdd ↑(((Direct …
    -/
      /-
        case refine_2.mk
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        σ : Type u_4
        inst✝⁵ : Semiring A
        inst✝⁴ : DecidableEq ι
        inst✝³ : CanonicallyOrderedAddCommMonoid ι
        inst✝² : SetLike σ A
        inst✝¹ : AddSubmonoidClass σ A
        𝒜 : ι → σ
        inst✝ : GradedRing 𝒜
        i : ι
        c : A
        hc : Membership.mem (𝒜 i) c
        ⊢ ∀ (y : A), Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one …
      -/
    rw [decompose_add]
        /-
          case refine_2.mk.refine_1
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          σ : Type u_4
          inst✝⁵ : Semiring A
          inst✝⁴ : DecidableEq ι
          inst✝³ : CanonicallyOrderedAddCommMonoid ι
          inst✝² : SetLike σ A
          inst✝¹ : AddSubmonoidClass σ A
          𝒜 : ι → σ
          inst✝ : GradedRing 𝒜
          i : ι
          c : A
          hc : Membership.mem (𝒜 i) c
          ⊢ Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one' := ⋯ }.to …
        -/
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      σ : Type u_4
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : CanonicallyOrderedAddCommMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      x✝¹ x✝ : A
      ⊢ Eq (↑((HAdd.hAdd ((DirectSum.decompose 𝒜) x✝¹) ((DirectSum.decompose 𝒜) x✝)) …
    -/
        /-
          🎉 no goals
        -/
        /-
          case refine_2.mk.refine_2
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          σ : Type u_4
          inst✝⁵ : Semiring A
          inst✝⁴ : DecidableEq ι
          inst✝³ : CanonicallyOrderedAddCommMonoid ι
          inst✝² : SetLike σ A
          inst✝¹ : AddSubmonoidClass σ A
          𝒜 : ι → σ
          inst✝ : GradedRing 𝒜
          i : ι
          c : A
          hc : Membership.mem (𝒜 i) c
          ⊢ ∀ {i_1 : ι} (m : Subtype fun x => Membership.mem (𝒜 i_1) x), Eq ({ toFun :=  …
        -/
    rfl
        /-
          case refine_2.mk.refine_2.mk
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          σ : Type u_4
          inst✝⁵ : Semiring A
          inst✝⁴ : DecidableEq ι
          inst✝³ : CanonicallyOrderedAddCommMonoid ι
          inst✝² : SetLike σ A
          inst✝¹ : AddSubmonoidClass σ A
          𝒜 : ι → σ
          inst✝ : GradedRing 𝒜
          i : ι
          c : A
          hc : Membership.mem (𝒜 i) c
          j : ι
          c' : A
          hc' : Membership.mem (𝒜 j) c'
          ⊢ Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one' := ⋯ }.to …
        -/
    /-
      🎉 no goals
    -/
        /-
          case refine_2.mk.refine_2.mk
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          σ : Type u_4
          inst✝⁵ : Semiring A
          inst✝⁴ : DecidableEq ι
          inst✝³ : CanonicallyOrderedAddCommMonoid ι
          inst✝² : SetLike σ A
          inst✝¹ : AddSubmonoidClass σ A
          𝒜 : ι → σ
          inst✝ : GradedRing 𝒜
          i : ι
          c : A
          hc : Membership.mem (𝒜 i) c
          j : ι
          c' : A
          hc' : Membership.mem (𝒜 j) c'
          ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul c c')) 0)) (HMul.hMul ↑(((DirectSu …
        -/
  map_mul' := by
    refine DirectSum.Decomposition.inductionOn 𝒜 (fun x => ?_) ?_ ?_
    · simp only [zero_mul, decompose_zero, zero_apply, ZeroMemClass.coe_zero]
    · rintro i ⟨c, hc⟩
      refine DirectSum.Decomposition.inductionOn 𝒜 ?_ ?_ ?_
          /-
            case neg
            ι : Type u_1
            R : Type u_2
            A : Type u_3
            σ : Type u_4
            inst✝⁵ : Semiring A
            inst✝⁴ : DecidableEq ι
            inst✝³ : CanonicallyOrderedAddCommMonoid ι
            inst✝² : SetLike σ A
            inst✝¹ : AddSubmonoidClass σ A
            𝒜 : ι → σ
            inst✝ : GradedRing 𝒜
            i : ι
            c : A
            hc : Membership.mem (𝒜 i) c
            j : ι
            c' : A
            hc' : Membership.mem (𝒜 j) c'
            h : Not (Eq (HAdd.hAdd i j) 0)
            ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul c c')) 0)) (HMul.hMul ↑(((DirectSu …
          -/
      · simp only [mul_zero, decompose_zero, zero_apply, ZeroMemClass.coe_zero]
          /-
            case neg
            ι : Type u_1
            R : Type u_2
            A : Type u_3
            σ : Type u_4
            inst✝⁵ : Semiring A
            inst✝⁴ : DecidableEq ι
            inst✝³ : CanonicallyOrderedAddCommMonoid ι
            inst✝² : SetLike σ A
            inst✝¹ : AddSubmonoidClass σ A
            𝒜 : ι → σ
            inst✝ : GradedRing 𝒜
            i : ι
            c : A
            hc : Membership.mem (𝒜 i) c
            j : ι
            c' : A
            hc' : Membership.mem (𝒜 j) c'
            h : Not (Eq (HAdd.hAdd i j) 0)
            ⊢ Eq 0 (HMul.hMul ↑(((DirectSum.decompose 𝒜) c) 0) ↑(((DirectSum.decompose 𝒜)  …
          -/
      · rintro j ⟨c', hc'⟩
            /-
              case neg.inl
              ι : Type u_1
              R : Type u_2
              A : Type u_3
              σ : Type u_4
              inst✝⁵ : Semiring A
              inst✝⁴ : DecidableEq ι
              inst✝³ : CanonicallyOrderedAddCommMonoid ι
              inst✝² : SetLike σ A
              inst✝¹ : AddSubmonoidClass σ A
              𝒜 : ι → σ
              inst✝ : GradedRing 𝒜
              i : ι
              c : A
              hc : Membership.mem (𝒜 i) c
              j : ι
              c' : A
              hc' : Membership.mem (𝒜 j) c'
              h : Not (Eq (HAdd.hAdd i j) 0)
              h' : Ne i 0
              ⊢ Eq 0 (HMul.hMul ↑(((DirectSum.decompose 𝒜) c) 0) ↑(((DirectSum.decompose 𝒜)  …
            -/
        simp only [Subtype.coe_mk]
            /-
              🎉 no goals
            -/
            /-
              case neg.inr
              ι : Type u_1
              R : Type u_2
              A : Type u_3
              σ : Type u_4
              inst✝⁵ : Semiring A
              inst✝⁴ : DecidableEq ι
              inst✝³ : CanonicallyOrderedAddCommMonoid ι
              inst✝² : SetLike σ A
              inst✝¹ : AddSubmonoidClass σ A
              𝒜 : ι → σ
              inst✝ : GradedRing 𝒜
              i : ι
              c : A
              hc : Membership.mem (𝒜 i) c
              j : ι
              c' : A
              hc' : Membership.mem (𝒜 j) c'
              h : Not (Eq (HAdd.hAdd i j) 0)
              h' : Ne j 0
              ⊢ Eq 0 (HMul.hMul ↑(((DirectSum.decompose 𝒜) c) 0) ↑(((DirectSum.decompose 𝒜)  …
            -/
        by_cases h : i + j = 0
            /-
              🎉 no goals
            -/
        /-
          case refine_2.mk.refine_3
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          σ : Type u_4
          inst✝⁵ : Semiring A
          inst✝⁴ : DecidableEq ι
          inst✝³ : CanonicallyOrderedAddCommMonoid ι
          inst✝² : SetLike σ A
          inst✝¹ : AddSubmonoidClass σ A
          𝒜 : ι → σ
          inst✝ : GradedRing 𝒜
          i : ι
          c : A
          hc : Membership.mem (𝒜 i) c
          ⊢ ∀ (m m' : A), Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_ …
        -/
        · rw [decompose_of_mem_same 𝒜
        /-
          case refine_2.mk.refine_3
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          σ : Type u_4
          inst✝⁵ : Semiring A
          inst✝⁴ : DecidableEq ι
          inst✝³ : CanonicallyOrderedAddCommMonoid ι
          inst✝² : SetLike σ A
          inst✝¹ : AddSubmonoidClass σ A
          𝒜 : ι → σ
          inst✝ : GradedRing 𝒜
          i : ι
          c : A
          hc : Membership.mem (𝒜 i) c
          m✝ m'✝ : A
          hd : Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one' := ⋯ } …
          he : Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one' := ⋯ } …
          ⊢ Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one' := ⋯ }.to …
        -/
              (show c * c' ∈ 𝒜 0 from h ▸ SetLike.GradedMul.mul_mem hc hc'),
        /-
          case refine_2.mk.refine_3
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          σ : Type u_4
          inst✝⁵ : Semiring A
          inst✝⁴ : DecidableEq ι
          inst✝³ : CanonicallyOrderedAddCommMonoid ι
          inst✝² : SetLike σ A
          inst✝¹ : AddSubmonoidClass σ A
          𝒜 : ι → σ
          inst✝ : GradedRing 𝒜
          i : ι
          c : A
          hc : Membership.mem (𝒜 i) c
          m✝ m'✝ : A
          hd : Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul c m✝)) 0)) (HMul.hMul ↑(((Direc …
          he : Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul c m'✝)) 0)) (HMul.hMul ↑(((Dire …
          ⊢ Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one' := ⋯ }.to …
        -/
            decompose_of_mem_same 𝒜 (show c ∈ 𝒜 0 from (add_eq_zero.mp h).1 ▸ hc),
        /-
          🎉 no goals
        -/
      /-
        case refine_3
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        σ : Type u_4
        inst✝⁵ : Semiring A
        inst✝⁴ : DecidableEq ι
        inst✝³ : CanonicallyOrderedAddCommMonoid ι
        inst✝² : SetLike σ A
        inst✝¹ : AddSubmonoidClass σ A
        𝒜 : ι → σ
        inst✝ : GradedRing 𝒜
        ⊢ ∀ (m m' : A), (∀ (y : A), Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) …
      -/
            decompose_of_mem_same 𝒜 (show c' ∈ 𝒜 0 from (add_eq_zero.mp h).2 ▸ hc')]
      /-
        case refine_3
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        σ : Type u_4
        inst✝⁵ : Semiring A
        inst✝⁴ : DecidableEq ι
        inst✝³ : CanonicallyOrderedAddCommMonoid ι
        inst✝² : SetLike σ A
        inst✝¹ : AddSubmonoidClass σ A
        𝒜 : ι → σ
        inst✝ : GradedRing 𝒜
        m✝ m'✝ : A
        ha : ∀ (y : A), Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_ …
        hb : ∀ (y : A), Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_ …
        y✝ : A
        ⊢ Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one' := ⋯ }.to …
      -/
        · rw [decompose_of_mem_ne 𝒜 (SetLike.GradedMul.mul_mem hc hc') h]
      /-
        case refine_3
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        σ : Type u_4
        inst✝⁵ : Semiring A
        inst✝⁴ : DecidableEq ι
        inst✝³ : CanonicallyOrderedAddCommMonoid ι
        inst✝² : SetLike σ A
        inst✝¹ : AddSubmonoidClass σ A
        𝒜 : ι → σ
        inst✝ : GradedRing 𝒜
        m✝ m'✝ : A
        ha : ∀ (y : A), Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul m✝ y)) 0)) (HMul.hMu …
        hb : ∀ (y : A), Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul m'✝ y)) 0)) (HMul.hM …
        y✝ : A
        ⊢ Eq ({ toFun := fun a => ↑(((DirectSum.decompose 𝒜) a) 0), map_one' := ⋯ }.to …
      -/
          cases' show i ≠ 0 ∨ j ≠ 0 by rwa [add_eq_zero, not_and_or] at h with h' h'
      /-
        🎉 no goals
      -/
          · simp only [decompose_of_mem_ne 𝒜 hc h', zero_mul]
          · simp only [decompose_of_mem_ne 𝒜 hc' h', mul_zero]
      · intro _ _ hd he
        simp only at hd he -- Porting note: added
        simp only [mul_add, decompose_add, add_apply, AddMemClass.coe_add, hd, he]
    · rintro _ _ ha hb _
      simp only at ha hb -- Porting note: added
      simp only [add_mul, decompose_add, add_apply, AddMemClass.coe_add, ha, hb]


/-- The ring homomorphism from `A` to `𝒜 0` sending every `a : A` to `a₀`. -/
def GradedRing.projZeroRingHom' : A →+* 𝒜 0 :=
  ((GradedRing.projZeroRingHom 𝒜).codRestrict _ fun _x => SetLike.coe_mem _ :
  A →+* SetLike.GradeZero.subsemiring 𝒜)


@[simp] lemma GradedRing.coe_projZeroRingHom'_apply (a : A) :
    (GradedRing.projZeroRingHom' 𝒜 a : A) = GradedRing.projZeroRingHom 𝒜 a := rfl


@[simp] lemma GradedRing.projZeroRingHom'_apply_coe (a : 𝒜 0) :
    GradedRing.projZeroRingHom' 𝒜 a = a := by
  /-
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : CanonicallyOrderedAddCommMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    a : Subtype fun x => Membership.mem (𝒜 0) x
    ⊢ Eq ((GradedRing.projZeroRingHom' 𝒜) ↑a) a
  -/
  ext; simp only [coe_projZeroRingHom'_apply, projZeroRingHom_apply, decompose_coe, of_eq_same]
       /-
         🎉 no goals
       -/


/-- The ring homomorphism `GradedRing.projZeroRingHom' 𝒜` is surjective. -/
lemma GradedRing.projZeroRingHom'_surjective :
    Function.Surjective (GradedRing.projZeroRingHom' 𝒜) :=
  Function.RightInverse.surjective (GradedRing.projZeroRingHom'_apply_coe 𝒜)


theorem coe_decompose_mul_of_left_mem_of_not_le (a_mem : a ∈ 𝒜 i) (h : ¬i ≤ n) :
    (decompose 𝒜 (a * b) n : A) = 0 := by
  /-
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : CanonicallyOrderedAddCommMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    a b : A
    n i : ι
    a_mem : Membership.mem (𝒜 i) a
    h : Not (LE.le i n)
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a b)) n)) 0
  -/
  lift a to 𝒜 i using a_mem
  /-
    case intro
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : CanonicallyOrderedAddCommMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    b : A
    n i : ι
    h : Not (LE.le i n)
    a : Subtype fun x => Membership.mem (𝒜 i) x
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul (↑a) b)) n)) 0
  -/
  rwa [decompose_mul, decompose_coe, coe_of_mul_apply_of_not_le]
  /-
    🎉 no goals
  -/


theorem coe_decompose_mul_of_right_mem_of_not_le (b_mem : b ∈ 𝒜 i) (h : ¬i ≤ n) :
    (decompose 𝒜 (a * b) n : A) = 0 := by
  /-
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : CanonicallyOrderedAddCommMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    a b : A
    n i : ι
    b_mem : Membership.mem (𝒜 i) b
    h : Not (LE.le i n)
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a b)) n)) 0
  -/
  lift b to 𝒜 i using b_mem
  /-
    case intro
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : CanonicallyOrderedAddCommMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    a : A
    n i : ι
    h : Not (LE.le i n)
    b : Subtype fun x => Membership.mem (𝒜 i) x
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a ↑b)) n)) 0
  -/
  rwa [decompose_mul, decompose_coe, coe_mul_of_apply_of_not_le]
  /-
    🎉 no goals
  -/


theorem coe_decompose_mul_of_left_mem_of_le (a_mem : a ∈ 𝒜 i) (h : i ≤ n) :
    (decompose 𝒜 (a * b) n : A) = a * decompose 𝒜 b (n - i) := by
  /-
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁸ : Semiring A
    inst✝⁷ : DecidableEq ι
    inst✝⁶ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁵ : SetLike σ A
    inst✝⁴ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝³ : GradedRing 𝒜
    a b : A
    n i : ι
    inst✝² : Sub ι
    inst✝¹ : OrderedSub ι
    inst✝ : AddLeftReflectLE ι
    a_mem : Membership.mem (𝒜 i) a
    h : LE.le i n
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a b)) n)) (HMul.hMul a ↑(((DirectS …
  -/
  lift a to 𝒜 i using a_mem
  /-
    case intro
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁸ : Semiring A
    inst✝⁷ : DecidableEq ι
    inst✝⁶ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁵ : SetLike σ A
    inst✝⁴ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝³ : GradedRing 𝒜
    b : A
    n i : ι
    inst✝² : Sub ι
    inst✝¹ : OrderedSub ι
    inst✝ : AddLeftReflectLE ι
    h : LE.le i n
    a : Subtype fun x => Membership.mem (𝒜 i) x
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul (↑a) b)) n)) (HMul.hMul ↑a ↑(((Dir …
  -/
  rwa [decompose_mul, decompose_coe, coe_of_mul_apply_of_le]
  /-
    🎉 no goals
  -/


theorem coe_decompose_mul_of_right_mem_of_le (b_mem : b ∈ 𝒜 i) (h : i ≤ n) :
    (decompose 𝒜 (a * b) n : A) = decompose 𝒜 a (n - i) * b := by
  /-
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁸ : Semiring A
    inst✝⁷ : DecidableEq ι
    inst✝⁶ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁵ : SetLike σ A
    inst✝⁴ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝³ : GradedRing 𝒜
    a b : A
    n i : ι
    inst✝² : Sub ι
    inst✝¹ : OrderedSub ι
    inst✝ : AddLeftReflectLE ι
    b_mem : Membership.mem (𝒜 i) b
    h : LE.le i n
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a b)) n)) (HMul.hMul (↑(((DirectSu …
  -/
  lift b to 𝒜 i using b_mem
  /-
    case intro
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁸ : Semiring A
    inst✝⁷ : DecidableEq ι
    inst✝⁶ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁵ : SetLike σ A
    inst✝⁴ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝³ : GradedRing 𝒜
    a : A
    n i : ι
    inst✝² : Sub ι
    inst✝¹ : OrderedSub ι
    inst✝ : AddLeftReflectLE ι
    h : LE.le i n
    b : Subtype fun x => Membership.mem (𝒜 i) x
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a ↑b)) n)) (HMul.hMul ↑(((DirectSu …
  -/
  rwa [decompose_mul, decompose_coe, coe_mul_of_apply_of_le]
  /-
    🎉 no goals
  -/


theorem coe_decompose_mul_of_left_mem (n) [Decidable (i ≤ n)] (a_mem : a ∈ 𝒜 i) :
    (decompose 𝒜 (a * b) n : A) = if i ≤ n then a * decompose 𝒜 b (n - i) else 0 := by
  /-
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁹ : Semiring A
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁶ : SetLike σ A
    inst✝⁵ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝⁴ : GradedRing 𝒜
    a b : A
    i : ι
    inst✝³ : Sub ι
    inst✝² : OrderedSub ι
    inst✝¹ : AddLeftReflectLE ι
    n : ι
    inst✝ : Decidable (LE.le i n)
    a_mem : Membership.mem (𝒜 i) a
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a b)) n)) (ite (LE.le i n) (HMul.h …
  -/
  lift a to 𝒜 i using a_mem
  /-
    case intro
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁹ : Semiring A
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁶ : SetLike σ A
    inst✝⁵ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝⁴ : GradedRing 𝒜
    b : A
    i : ι
    inst✝³ : Sub ι
    inst✝² : OrderedSub ι
    inst✝¹ : AddLeftReflectLE ι
    n : ι
    inst✝ : Decidable (LE.le i n)
    a : Subtype fun x => Membership.mem (𝒜 i) x
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul (↑a) b)) n)) (ite (LE.le i n) (HMu …
  -/
  rw [decompose_mul, decompose_coe, coe_of_mul_apply]
  /-
    🎉 no goals
  -/


theorem coe_decompose_mul_of_right_mem (n) [Decidable (i ≤ n)] (b_mem : b ∈ 𝒜 i) :
    (decompose 𝒜 (a * b) n : A) = if i ≤ n then decompose 𝒜 a (n - i) * b else 0 := by
  /-
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁹ : Semiring A
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁶ : SetLike σ A
    inst✝⁵ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝⁴ : GradedRing 𝒜
    a b : A
    i : ι
    inst✝³ : Sub ι
    inst✝² : OrderedSub ι
    inst✝¹ : AddLeftReflectLE ι
    n : ι
    inst✝ : Decidable (LE.le i n)
    b_mem : Membership.mem (𝒜 i) b
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a b)) n)) (ite (LE.le i n) (HMul.h …
  -/
  lift b to 𝒜 i using b_mem
  /-
    case intro
    ι : Type u_1
    A : Type u_3
    σ : Type u_4
    inst✝⁹ : Semiring A
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁶ : SetLike σ A
    inst✝⁵ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝⁴ : GradedRing 𝒜
    a : A
    i : ι
    inst✝³ : Sub ι
    inst✝² : OrderedSub ι
    inst✝¹ : AddLeftReflectLE ι
    n : ι
    inst✝ : Decidable (LE.le i n)
    b : Subtype fun x => Membership.mem (𝒜 i) x
    ⊢ Eq (↑(((DirectSum.decompose 𝒜) (HMul.hMul a ↑b)) n)) (ite (LE.le i n) (HMul. …
  -/
  rw [decompose_mul, decompose_coe, coe_mul_of_apply]
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism of an internal direct sum with the ambient algebra -/
noncomputable def coeAlgEquiv (hM : DirectSum.IsInternal M) :
    (DirectSum ι fun i => ↥(M i)) ≃ₐ[R] A :=
                                                                                   /-
                                                                                     ι✝ : Type u_1
                                                                                     R✝ : Type u_2
                                                                                     A✝ : Type u_3
                                                                                     σ : Type u_4
                                                                                     R : Type u_5
                                                                                     inst✝⁵ : CommSemiring R
                                                                                     A : Type u_6
                                                                                     inst✝⁴ : Semiring A
                                                                                     inst✝³ : Algebra R A
                                                                                     ι : Type u_7
                                                                                     inst✝² : DecidableEq ι
                                                                                     inst✝¹ : AddMonoid ι
                                                                                     M : ι → Submodule R A
                                                                                     inst✝ : SetLike.GradedMonoid M
                                                                                     hM : DirectSum.IsInternal M
                                                                                     r : R
                                                                                     ⊢ Eq (__src✝.toFun ((algebraMap R (DirectSum ι fun i => Subtype fun x => Membe …
                                                                                   -/
  { RingEquiv.ofBijective (DirectSum.coeAlgHom M) hM with commutes' := fun r => by simp }
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- Given an `R`-algebra `A` and a family `ι → Submodule R A` of submodules
parameterized by an additive monoid `ι`
and satisfying `SetLike.GradedMonoid M` (essentially, is multiplicative)
such that `DirectSum.IsInternal M` (`A` is the direct sum of the `M i`),
we endow `A` with the structure of a graded algebra.
The submodules are the *homogeneous* parts. -/
noncomputable def gradedAlgebra (hM : DirectSum.IsInternal M) : GradedAlgebra M :=
  { (inferInstance : SetLike.GradedMonoid M) with
    decompose' := hM.coeAlgEquiv.symm
    left_inv := hM.coeAlgEquiv.symm.left_inv
    right_inv := hM.coeAlgEquiv.left_inv }


