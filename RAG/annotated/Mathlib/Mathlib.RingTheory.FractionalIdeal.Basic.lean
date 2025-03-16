/-- A submodule `I` is a fractional ideal if `a I ⊆ R` for some `a ≠ 0`. -/
def IsFractional (I : Submodule R P) :=
  ∃ a ∈ S, ∀ b ∈ I, IsInteger R (a • b)


/-- The fractional ideals of a domain `R` are ideals of `R` divided by some `a ∈ R`.

More precisely, let `P` be a localization of `R` at some submonoid `S`,
then a fractional ideal `I ⊆ P` is an `R`-submodule of `P`,
such that there is a nonzero `a : R` with `a I ⊆ R`.
-/
def FractionalIdeal :=
  { I : Submodule R P // IsFractional S I }


/-- Map a fractional ideal `I` to a submodule by forgetting that `∃ a, a I ⊆ R`.

This implements the coercion `FractionalIdeal S P → Submodule R P`.
-/
@[coe]
def coeToSubmodule (I : FractionalIdeal S P) : Submodule R P :=
  I.val


/-- Map a fractional ideal `I` to a submodule by forgetting that `∃ a, a I ⊆ R`.

This coercion is typically called `coeToSubmodule` in lemma names
(or `coe` when the coercion is clear from the context),
not to be confused with `IsLocalization.coeSubmodule : Ideal R → Submodule R P`
(which we use to define `coe : Ideal R → FractionalIdeal S P`).
-/
instance : CoeOut (FractionalIdeal S P) (Submodule R P) :=
  ⟨coeToSubmodule⟩


protected theorem isFractional (I : FractionalIdeal S P) : IsFractional S (I : Submodule R P) :=
  I.prop


/-- An element of `S` such that `I.den • I = I.num`, see `FractionalIdeal.num` and
`FractionalIdeal.den_mul_self_eq_num`. -/
noncomputable def den (I : FractionalIdeal S P) : S :=
  ⟨I.2.choose, I.2.choose_spec.1⟩


/-- An ideal of `R` such that `I.den • I = I.num`, see `FractionalIdeal.den` and
`FractionalIdeal.den_mul_self_eq_num`. -/
noncomputable def num (I : FractionalIdeal S P) : Ideal R :=
  (I.den • (I : Submodule R P)).comap (Algebra.linearMap R P)


theorem den_mul_self_eq_num (I : FractionalIdeal S P) :
    I.den • (I : Submodule R P) = Submodule.map (Algebra.linearMap R P) I.num := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    ⊢ Eq (HSMul.hSMul I.den ↑I) (Submodule.map (Algebra.linearMap R P) I.num)
  -/
  rw [den, num, Submodule.map_comap_eq]
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    ⊢ Eq (HSMul.hSMul ⟨Exists.choose ⋯, ⋯⟩ ↑I) (Min.min (LinearMap.range (Algebra. …
  -/
  refine (inf_of_le_right ?_).symm
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    ⊢ LE.le (HSMul.hSMul ⟨Exists.choose ⋯, ⋯⟩ ↑I) (LinearMap.range (Algebra.linear …
  -/
  rintro _ ⟨a, ha, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    a : P
    ha : Membership.mem (↑↑I) a
    ⊢ Membership.mem (LinearMap.range (Algebra.linearMap R P)) ((DistribMulAction. …
  -/
  exact I.2.choose_spec.2 a ha
  /-
    🎉 no goals
  -/


/-- The linear equivalence between the fractional ideal `I` and the integral ideal `I.num`
defined by mapping `x` to `den I • x`. -/
noncomputable def equivNum [Nontrivial P] [NoZeroSMulDivisors R P]
    {I : FractionalIdeal S P} (h_nz : (I.den : R) ≠ 0) : I ≃ₗ[R] I.num := by
  refine LinearEquiv.trans
    (LinearEquiv.ofBijective ((DistribMulAction.toLinearMap R P I.den).restrict fun _ hx ↦ ?_)
      ⟨fun _ _ hxy ↦ ?_, fun ⟨y, hy⟩ ↦ ?_⟩)
    (Submodule.equivMapOfInjective (Algebra.linearMap R P)
      (NoZeroSMulDivisors.algebraMap_injective R P) (num I)).symm
    /-
      case refine_1
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝³ : CommRing P
      inst✝² : Algebra R P
      inst✝¹ : Nontrivial P
      inst✝ : NoZeroSMulDivisors R P
      I : FractionalIdeal S P
      h_nz : Ne (↑I.den) 0
      x✝ : P
      hx : Membership.mem (↑I) x✝
      ⊢ Membership.mem (Submodule.map (Algebra.linearMap R P) I.num) ((DistribMulAct …
    -/
  · rw [← den_mul_self_eq_num]
    /-
      case refine_1
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝³ : CommRing P
      inst✝² : Algebra R P
      inst✝¹ : Nontrivial P
      inst✝ : NoZeroSMulDivisors R P
      I : FractionalIdeal S P
      h_nz : Ne (↑I.den) 0
      x✝ : P
      hx : Membership.mem (↑I) x✝
      ⊢ Membership.mem (HSMul.hSMul I.den ↑I) ((DistribMulAction.toLinearMap R P I.d …
    -/
    exact Submodule.smul_mem_pointwise_smul _ _ _ hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝³ : CommRing P
      inst✝² : Algebra R P
      inst✝¹ : Nontrivial P
      inst✝ : NoZeroSMulDivisors R P
      I : FractionalIdeal S P
      h_nz : Ne (↑I.den) 0
      x✝¹ x✝ : Subtype fun x => Membership.mem (↑I) x
      hxy : Eq (((DistribMulAction.toLinearMap R P I.den).restrict ⋯) x✝¹) (((Distri …
      ⊢ Eq x✝¹ x✝
    -/
  · simp_rw [LinearMap.restrict_apply, DistribMulAction.toLinearMap_apply, Subtype.mk.injEq] at hxy
    /-
      case refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝³ : CommRing P
      inst✝² : Algebra R P
      inst✝¹ : Nontrivial P
      inst✝ : NoZeroSMulDivisors R P
      I : FractionalIdeal S P
      h_nz : Ne (↑I.den) 0
      x✝¹ x✝ : Subtype fun x => Membership.mem (↑I) x
      hxy : Eq (HSMul.hSMul I.den ↑x✝¹) (HSMul.hSMul I.den ↑x✝)
      ⊢ Eq x✝¹ x✝
    -/
    rwa [Submonoid.smul_def, Submonoid.smul_def, smul_right_inj h_nz, SetCoe.ext_iff] at hxy
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝³ : CommRing P
      inst✝² : Algebra R P
      inst✝¹ : Nontrivial P
      inst✝ : NoZeroSMulDivisors R P
      I : FractionalIdeal S P
      h_nz : Ne (↑I.den) 0
      x✝ : Subtype fun x => Membership.mem (Submodule.map (Algebra.linearMap R P) I. …
      y : P
      hy : Membership.mem (Submodule.map (Algebra.linearMap R P) I.num) y
      ⊢ Exists fun a => Eq (((DistribMulAction.toLinearMap R P I.den).restrict ⋯) a) …
    -/
  · rw [← den_mul_self_eq_num] at hy
    /-
      case refine_3
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝³ : CommRing P
      inst✝² : Algebra R P
      inst✝¹ : Nontrivial P
      inst✝ : NoZeroSMulDivisors R P
      I : FractionalIdeal S P
      h_nz : Ne (↑I.den) 0
      x✝ : Subtype fun x => Membership.mem (Submodule.map (Algebra.linearMap R P) I. …
      y : P
      hy✝ : Membership.mem (Submodule.map (Algebra.linearMap R P) I.num) y
      hy : Membership.mem (HSMul.hSMul I.den ↑I) y
      ⊢ Exists fun a => Eq (((DistribMulAction.toLinearMap R P I.den).restrict ⋯) a) …
    -/
    obtain ⟨x, hx, hxy⟩ := hy
    /-
      case refine_3.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝³ : CommRing P
      inst✝² : Algebra R P
      inst✝¹ : Nontrivial P
      inst✝ : NoZeroSMulDivisors R P
      I : FractionalIdeal S P
      h_nz : Ne (↑I.den) 0
      x✝ : Subtype fun x => Membership.mem (Submodule.map (Algebra.linearMap R P) I. …
      y : P
      hy : Membership.mem (Submodule.map (Algebra.linearMap R P) I.num) y
      x : P
      hx : Membership.mem (↑↑I) x
      hxy : Eq ((DistribMulAction.toLinearMap R P (S.subtype I.den)) x) y
      ⊢ Exists fun a => Eq (((DistribMulAction.toLinearMap R P I.den).restrict ⋯) a) …
    -/
    exact ⟨⟨x, hx⟩, by simp_rw [LinearMap.restrict_apply, Subtype.ext_iff, ← hxy]; rfl⟩
    /-
      🎉 no goals
    -/


instance : SetLike (FractionalIdeal S P) P where
  coe I := ↑(I : Submodule R P)
  coe_injective' := SetLike.coe_injective.comp Subtype.coe_injective


@[simp]
theorem mem_coe {I : FractionalIdeal S P} {x : P} : x ∈ (I : Submodule R P) ↔ x ∈ I :=
  Iff.rfl


@[ext]
theorem ext {I J : FractionalIdeal S P} : (∀ x, x ∈ I ↔ x ∈ J) → I = J :=
  SetLike.ext


@[simp]
 theorem equivNum_apply [Nontrivial P] [NoZeroSMulDivisors R P] {I : FractionalIdeal S P}
    (h_nz : (I.den : R) ≠ 0) (x : I) :
    algebraMap R P (equivNum h_nz x) = I.den • x := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝³ : CommRing P
    inst✝² : Algebra R P
    inst✝¹ : Nontrivial P
    inst✝ : NoZeroSMulDivisors R P
    I : FractionalIdeal S P
    h_nz : Ne (↑I.den) 0
    x : Subtype fun x => Membership.mem I x
    ⊢ Eq ((algebraMap R P) ↑((FractionalIdeal.equivNum h_nz) x)) (HSMul.hSMul I.de …
  -/
  change Algebra.linearMap R P _ = _
  rw [equivNum, LinearEquiv.trans_apply, LinearEquiv.ofBijective_apply, LinearMap.restrict_apply,
    Submodule.map_equivMapOfInjective_symm_apply, Subtype.coe_mk,
    DistribMulAction.toLinearMap_apply]


/-- Copy of a `FractionalIdeal` with a new underlying set equal to the old one.
Useful to fix definitional equalities. -/
protected def copy (p : FractionalIdeal S P) (s : Set P) (hs : s = ↑p) : FractionalIdeal S P :=
  ⟨Submodule.copy p s hs, by
    /-
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      p : FractionalIdeal S P
      s : Set P
      hs : Eq s ↑p
      ⊢ IsFractional S ((↑p).copy s hs)
    -/
    convert p.isFractional
    /-
      case h.e'_7
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      p : FractionalIdeal S P
      s : Set P
      hs : Eq s ↑p
      ⊢ Eq ((↑p).copy s hs) ↑p
    -/
    ext
    /-
      case h.e'_7.h
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      p : FractionalIdeal S P
      s : Set P
      hs : Eq s ↑p
      x✝ : P
      ⊢ Iff (Membership.mem ((↑p).copy s hs) x✝) (Membership.mem (↑p) x✝)
    -/
    simp only [hs]
    /-
      case h.e'_7.h
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      p : FractionalIdeal S P
      s : Set P
      hs : Eq s ↑p
      x✝ : P
      ⊢ Iff (Membership.mem ((↑p).copy ↑p ⋯) x✝) (Membership.mem (↑p) x✝)
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_copy (p : FractionalIdeal S P) (s : Set P) (hs : s = ↑p) : ↑(p.copy s hs) = s :=
  rfl


theorem coe_eq (p : FractionalIdeal S P) (s : Set P) (hs : s = ↑p) : p.copy s hs = p :=
  SetLike.coe_injective hs


lemma zero_mem (I : FractionalIdeal S P) : 0 ∈ I := I.coeToSubmodule.zero_mem

-- Porting note: this seems to be needed a lot more than in Lean 3

@[simp]
theorem val_eq_coe (I : FractionalIdeal S P) : I.val = I :=
  rfl

-- Porting note: had to rephrase this to make it clear to `simp` what was going on.

@[simp, norm_cast]
theorem coe_mk (I : Submodule R P) (hI : IsFractional S I) :
    coeToSubmodule ⟨I, hI⟩ = I :=
  rfl


theorem coeToSet_coeToSubmodule (I : FractionalIdeal S P) :
    ((I : Submodule R P) : Set P) = I :=
  rfl


instance (I : FractionalIdeal S P) : Module R I :=
  Submodule.module (I : Submodule R P)


theorem coeToSubmodule_injective :
    Function.Injective (fun (I : FractionalIdeal S P) ↦ (I : Submodule R P)) :=
  Subtype.coe_injective


theorem coeToSubmodule_inj {I J : FractionalIdeal S P} : (I : Submodule R P) = J ↔ I = J :=
  coeToSubmodule_injective.eq_iff


theorem isFractional_of_le_one (I : Submodule R P) (h : I ≤ 1) : IsFractional S I := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : Submodule R P
    h : LE.le I 1
    ⊢ IsFractional S I
  -/
  use 1, S.one_mem
  /-
    case right
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : Submodule R P
    h : LE.le I 1
    ⊢ ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul 1 b)
  -/
  intro b hb
  /-
    case right
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : Submodule R P
    h : LE.le I 1
    b : P
    hb : Membership.mem I b
    ⊢ IsLocalization.IsInteger R (HSMul.hSMul 1 b)
  -/
  rw [one_smul]
  /-
    case right
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : Submodule R P
    h : LE.le I 1
    b : P
    hb : Membership.mem I b
    ⊢ IsLocalization.IsInteger R b
  -/
  obtain ⟨b', b'_mem, rfl⟩ := mem_one.mp (h hb)
  /-
    case right.intro.refl
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : Submodule R P
    h : LE.le I 1
    b' : R
    hb : Membership.mem I ((algebraMap R P) b')
    ⊢ IsLocalization.IsInteger R ((algebraMap R P) b')
  -/
  exact Set.mem_range_self b'
  /-
    🎉 no goals
  -/


theorem isFractional_of_le {I : Submodule R P} {J : FractionalIdeal S P} (hIJ : I ≤ J) :
    IsFractional S I := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : Submodule R P
    J : FractionalIdeal S P
    hIJ : LE.le I ↑J
    ⊢ IsFractional S I
  -/
  obtain ⟨a, a_mem, ha⟩ := J.isFractional
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : Submodule R P
    J : FractionalIdeal S P
    hIJ : LE.le I ↑J
    a : R
    a_mem : Membership.mem S a
    ha : ∀ (b : P), Membership.mem (↑J) b → IsLocalization.IsInteger R (HSMul.hSMu …
    ⊢ IsFractional S I
  -/
  use a, a_mem
  /-
    case right
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : Submodule R P
    J : FractionalIdeal S P
    hIJ : LE.le I ↑J
    a : R
    a_mem : Membership.mem S a
    ha : ∀ (b : P), Membership.mem (↑J) b → IsLocalization.IsInteger R (HSMul.hSMu …
    ⊢ ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a b)
  -/
  intro b b_mem
  /-
    case right
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : Submodule R P
    J : FractionalIdeal S P
    hIJ : LE.le I ↑J
    a : R
    a_mem : Membership.mem S a
    ha : ∀ (b : P), Membership.mem (↑J) b → IsLocalization.IsInteger R (HSMul.hSMu …
    b : P
    b_mem : Membership.mem I b
    ⊢ IsLocalization.IsInteger R (HSMul.hSMul a b)
  -/
  exact ha b (hIJ b_mem)
  /-
    🎉 no goals
  -/


/-- Map an ideal `I` to a fractional ideal by forgetting `I` is integral.

This is the function that implements the coercion `Ideal R → FractionalIdeal S P`. -/
@[coe]
def coeIdeal (I : Ideal R) : FractionalIdeal S P :=
  ⟨coeSubmodule P I,
                                  /-
                                    R : Type u_1
                                    inst✝² : CommRing R
                                    S : Submonoid R
                                    P : Type u_2
                                    inst✝¹ : CommRing P
                                    inst✝ : Algebra R P
                                    I : Ideal R
                                    ⊢ LE.le (IsLocalization.coeSubmodule P I) 1
                                  -/
   isFractional_of_le_one _ <| by simpa using coeSubmodule_mono P (le_top : I ≤ ⊤)⟩
                                  /-
                                    🎉 no goals
                                  -/

-- Is a `CoeTC` rather than `Coe` to speed up failing inference, see library note [use has_coe_t]

/-- Map an ideal `I` to a fractional ideal by forgetting `I` is integral.

This is a bundled version of `IsLocalization.coeSubmodule : Ideal R → Submodule R P`,
which is not to be confused with the `coe : FractionalIdeal S P → Submodule R P`,
also called `coeToSubmodule` in theorem names.

This map is available as a ring hom, called `FractionalIdeal.coeIdealHom`.
-/
instance : CoeTC (Ideal R) (FractionalIdeal S P) :=
  ⟨fun I => coeIdeal I⟩


@[simp, norm_cast]
theorem coe_coeIdeal (I : Ideal R) :
    ((I : FractionalIdeal S P) : Submodule R P) = coeSubmodule P I :=
  rfl


@[simp]
theorem mem_coeIdeal {x : P} {I : Ideal R} :
    x ∈ (I : FractionalIdeal S P) ↔ ∃ x', x' ∈ I ∧ algebraMap R P x' = x :=
  mem_coeSubmodule _ _


theorem mem_coeIdeal_of_mem {x : R} {I : Ideal R} (hx : x ∈ I) :
    algebraMap R P x ∈ (I : FractionalIdeal S P) :=
  (mem_coeIdeal S).mpr ⟨x, hx, rfl⟩


theorem coeIdeal_le_coeIdeal' [IsLocalization S P] (h : S ≤ nonZeroDivisors R) {I J : Ideal R} :
    (I : FractionalIdeal S P) ≤ J ↔ I ≤ J :=
  coeSubmodule_le_coeSubmodule h


@[simp]
theorem coeIdeal_le_coeIdeal (K : Type*) [CommRing K] [Algebra R K] [IsFractionRing R K]
    {I J : Ideal R} : (I : FractionalIdeal R⁰ K) ≤ J ↔ I ≤ J :=
  IsFractionRing.coeSubmodule_le_coeSubmodule


instance : Zero (FractionalIdeal S P) :=
  ⟨(0 : Ideal R)⟩


@[simp]
theorem mem_zero_iff {x : P} : x ∈ (0 : FractionalIdeal S P) ↔ x = 0 :=
  ⟨fun ⟨x', x'_mem_zero, x'_eq_x⟩ => by
    /-
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      x : P
      x✝ : Membership.mem 0 x
      x' : R
      x'_mem_zero : Membership.mem (↑0) x'
      x'_eq_x : Eq ((Algebra.linearMap R P) x') x
      ⊢ Eq x 0
    -/
    have x'_eq_zero : x' = 0 := x'_mem_zero
    /-
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      x : P
      x✝ : Membership.mem 0 x
      x' : R
      x'_mem_zero : Membership.mem (↑0) x'
      x'_eq_x : Eq ((Algebra.linearMap R P) x') x
      x'_eq_zero : Eq x' 0
      ⊢ Eq x 0
    -/
    /-
      🎉 no goals
    -/
    simp [x'_eq_x.symm, x'_eq_zero], fun hx => ⟨0, rfl, by simp [hx]⟩⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp, norm_cast]
theorem coe_zero : ↑(0 : FractionalIdeal S P) = (⊥ : Submodule R P) :=
  Submodule.ext fun _ => mem_zero_iff S


@[simp, norm_cast]
theorem coeIdeal_bot : ((⊥ : Ideal R) : FractionalIdeal S P) = 0 :=
  rfl


variable (P) in
@[simp]
theorem exists_mem_algebraMap_eq {x : R} {I : Ideal R} (h : S ≤ nonZeroDivisors R) :
    (∃ x', x' ∈ I ∧ algebraMap R P x' = algebraMap R P x) ↔ x ∈ I :=
  ⟨fun ⟨_, hx', Eq⟩ => IsLocalization.injective _ h Eq ▸ hx', fun h => ⟨x, h, rfl⟩⟩


theorem coeIdeal_injective' (h : S ≤ nonZeroDivisors R) :
    Function.Injective (fun (I : Ideal R) ↦ (I : FractionalIdeal S P)) := fun _ _ h' =>
  ((coeIdeal_le_coeIdeal' S h).mp h'.le).antisymm ((coeIdeal_le_coeIdeal' S h).mp
    h'.ge)


theorem coeIdeal_inj' (h : S ≤ nonZeroDivisors R) {I J : Ideal R} :
    (I : FractionalIdeal S P) = J ↔ I = J :=
  (coeIdeal_injective' h).eq_iff

-- Porting note: doesn't need to be @[simp] because it can be proved by coeIdeal_eq_zero

theorem coeIdeal_eq_zero' {I : Ideal R} (h : S ≤ nonZeroDivisors R) :
    (I : FractionalIdeal S P) = 0 ↔ I = (⊥ : Ideal R) :=
  coeIdeal_inj' h


theorem coeIdeal_ne_zero' {I : Ideal R} (h : S ≤ nonZeroDivisors R) :
    (I : FractionalIdeal S P) ≠ 0 ↔ I ≠ (⊥ : Ideal R) :=
  not_iff_not.mpr <| coeIdeal_eq_zero' h


theorem coeToSubmodule_eq_bot {I : FractionalIdeal S P} : (I : Submodule R P) = ⊥ ↔ I = 0 :=
                                         /-
                                           R : Type u_1
                                           inst✝² : CommRing R
                                           S : Submonoid R
                                           P : Type u_2
                                           inst✝¹ : CommRing P
                                           inst✝ : Algebra R P
                                           I : FractionalIdeal S P
                                           h : Eq (↑I) Bot.bot
                                           ⊢ Eq ((fun I => ↑I) I) ((fun I => ↑I) 0)
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
  ⟨fun h => coeToSubmodule_injective (by simp [h]), fun h => by simp [h]⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem coeToSubmodule_ne_bot {I : FractionalIdeal S P} : ↑I ≠ (⊥ : Submodule R P) ↔ I ≠ 0 :=
  not_iff_not.mpr coeToSubmodule_eq_bot


instance : Inhabited (FractionalIdeal S P) :=
  ⟨0⟩


instance : One (FractionalIdeal S P) :=
  ⟨(⊤ : Ideal R)⟩


theorem zero_of_num_eq_bot [NoZeroSMulDivisors R P] (hS : 0 ∉ S) {I : FractionalIdeal S P}
    (hI : I.num = ⊥) : I = 0 := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : NoZeroSMulDivisors R P
    hS : Not (Membership.mem S 0)
    I : FractionalIdeal S P
    hI : Eq I.num Bot.bot
    ⊢ Eq I 0
  -/
  rw [← coeToSubmodule_eq_bot, eq_bot_iff]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : NoZeroSMulDivisors R P
    hS : Not (Membership.mem S 0)
    I : FractionalIdeal S P
    hI : Eq I.num Bot.bot
    ⊢ LE.le (↑I) Bot.bot
  -/
  intro x hx
  suffices (den I : R) • x = 0 from
    (smul_eq_zero.mp this).resolve_left (ne_of_mem_of_not_mem (SetLike.coe_mem _) hS)
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : NoZeroSMulDivisors R P
    hS : Not (Membership.mem S 0)
    I : FractionalIdeal S P
    hI : Eq I.num Bot.bot
    x : P
    hx : Membership.mem (↑I) x
    ⊢ Eq (HSMul.hSMul (↑I.den) x) 0
  -/
  have h_eq : I.den • (I : Submodule R P) = ⊥ := by rw [den_mul_self_eq_num, hI, Submodule.map_bot]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : NoZeroSMulDivisors R P
    hS : Not (Membership.mem S 0)
    I : FractionalIdeal S P
    hI : Eq I.num Bot.bot
    x : P
    hx : Membership.mem (↑I) x
    h_eq : Eq (HSMul.hSMul I.den ↑I) Bot.bot
    ⊢ Eq (HSMul.hSMul (↑I.den) x) 0
  -/
  exact (Submodule.eq_bot_iff _).mp h_eq (den I • x) ⟨x, hx, rfl⟩
  /-
    🎉 no goals
  -/


theorem num_zero_eq (h_inj : Function.Injective (algebraMap R P)) :
    num (0 : FractionalIdeal S P) = 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    h_inj : Function.Injective ⇑(algebraMap R P)
    ⊢ Eq (FractionalIdeal.num 0) 0
  -/
  simpa [num, LinearMap.ker_eq_bot] using h_inj
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coeIdeal_top : ((⊤ : Ideal R) : FractionalIdeal S P) = 1 :=
  rfl


theorem mem_one_iff {x : P} : x ∈ (1 : FractionalIdeal S P) ↔ ∃ x' : R, algebraMap R P x' = x :=
  Iff.intro (fun ⟨x', _, h⟩ => ⟨x', h⟩) fun ⟨x', h⟩ => ⟨x', ⟨⟩, h⟩


theorem coe_mem_one (x : R) : algebraMap R P x ∈ (1 : FractionalIdeal S P) :=
  (mem_one_iff S).mpr ⟨x, rfl⟩


theorem one_mem_one : (1 : P) ∈ (1 : FractionalIdeal S P) :=
  (mem_one_iff S).mpr ⟨1, RingHom.map_one _⟩


/-- `(1 : FractionalIdeal S P)` is defined as the R-submodule `f(R) ≤ P`.

However, this is not definitionally equal to `1 : Submodule R P`,
which is proved in the actual `simp` lemma `coe_one`. -/
theorem coe_one_eq_coeSubmodule_top : ↑(1 : FractionalIdeal S P) = coeSubmodule P (⊤ : Ideal R) :=
  rfl


@[simp, norm_cast]
theorem coe_one : (↑(1 : FractionalIdeal S P) : Submodule R P) = 1 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    ⊢ Eq (↑1) 1
  -/
  rw [coe_one_eq_coeSubmodule_top, coeSubmodule_top]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_le_coe {I J : FractionalIdeal S P} :
    (I : Submodule R P) ≤ (J : Submodule R P) ↔ I ≤ J :=
  Iff.rfl


theorem zero_le (I : FractionalIdeal S P) : 0 ≤ I := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    ⊢ LE.le 0 I
  -/
  intro x hx
  -- Porting note: changed the proof from convert; simp into rw; exact
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    x : P
    hx : Membership.mem 0 x
    ⊢ Membership.mem I x
  -/
  rw [(mem_zero_iff _).mp hx]
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    x : P
    hx : Membership.mem 0 x
    ⊢ Membership.mem I 0
  -/
  exact zero_mem I
  /-
    🎉 no goals
  -/


instance orderBot : OrderBot (FractionalIdeal S P) where
  bot := 0
  bot_le := zero_le


@[simp]
theorem bot_eq_zero : (⊥ : FractionalIdeal S P) = 0 :=
  rfl


@[simp]
theorem le_zero_iff {I : FractionalIdeal S P} : I ≤ 0 ↔ I = 0 :=
  le_bot_iff


theorem eq_zero_iff {I : FractionalIdeal S P} : I = 0 ↔ ∀ x ∈ I, x = (0 : P) :=
                    /-
                      R : Type u_1
                      inst✝² : CommRing R
                      S : Submonoid R
                      P : Type u_2
                      inst✝¹ : CommRing P
                      inst✝ : Algebra R P
                      I : FractionalIdeal S P
                      h : Eq I 0
                      x : P
                      hx : Membership.mem I x
                      ⊢ Eq x 0
                    -/
  ⟨fun h x hx => by simpa [h, mem_zero_iff] using hx, fun h =>
                    /-
                      🎉 no goals
                    -/
    le_bot_iff.mp fun x hx => (mem_zero_iff S).mpr (h x hx)⟩


theorem _root_.IsFractional.sup {I J : Submodule R P} :
    IsFractional S I → IsFractional S J → IsFractional S (I ⊔ J)
  | ⟨aI, haI, hI⟩, ⟨aJ, haJ, hJ⟩ =>
    ⟨aI * aJ, S.mul_mem haI haJ, fun b hb => by
      /-
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        I J : Submodule R P
        aI : R
        haI : Membership.mem S aI
        hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
        aJ : R
        haJ : Membership.mem S aJ
        hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
        b : P
        hb : Membership.mem (Max.max I J) b
        ⊢ IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) b)
      -/
      rcases mem_sup.mp hb with ⟨bI, hbI, bJ, hbJ, rfl⟩
      /-
        case intro.intro.intro.intro
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        I J : Submodule R P
        aI : R
        haI : Membership.mem S aI
        hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
        aJ : R
        haJ : Membership.mem S aJ
        hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
        bI : P
        hbI : Membership.mem I bI
        bJ : P
        hbJ : Membership.mem J bJ
        hb : Membership.mem (Max.max I J) (HAdd.hAdd bI bJ)
        ⊢ IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) (HAdd.hAdd bI bJ))
      -/
      rw [smul_add]
      /-
        case intro.intro.intro.intro
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        I J : Submodule R P
        aI : R
        haI : Membership.mem S aI
        hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
        aJ : R
        haJ : Membership.mem S aJ
        hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
        bI : P
        hbI : Membership.mem I bI
        bJ : P
        hbJ : Membership.mem J bJ
        hb : Membership.mem (Max.max I J) (HAdd.hAdd bI bJ)
        ⊢ IsLocalization.IsInteger R (HAdd.hAdd (HSMul.hSMul (HMul.hMul aI aJ) bI) (HS …
      -/
      apply isInteger_add
        /-
          case intro.intro.intro.intro.ha
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          bI : P
          hbI : Membership.mem I bI
          bJ : P
          hbJ : Membership.mem J bJ
          hb : Membership.mem (Max.max I J) (HAdd.hAdd bI bJ)
          ⊢ IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) bI)
        -/
      · rw [mul_smul, smul_comm]
        /-
          case intro.intro.intro.intro.ha
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          bI : P
          hbI : Membership.mem I bI
          bJ : P
          hbJ : Membership.mem J bJ
          hb : Membership.mem (Max.max I J) (HAdd.hAdd bI bJ)
          ⊢ IsLocalization.IsInteger R (HSMul.hSMul aJ (HSMul.hSMul aI bI))
        -/
        exact isInteger_smul (hI bI hbI)
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.intro.hb
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          bI : P
          hbI : Membership.mem I bI
          bJ : P
          hbJ : Membership.mem J bJ
          hb : Membership.mem (Max.max I J) (HAdd.hAdd bI bJ)
          ⊢ IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) bJ)
        -/
      · rw [mul_smul]
        /-
          case intro.intro.intro.intro.hb
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          bI : P
          hbI : Membership.mem I bI
          bJ : P
          hbJ : Membership.mem J bJ
          hb : Membership.mem (Max.max I J) (HAdd.hAdd bI bJ)
          ⊢ IsLocalization.IsInteger R (HSMul.hSMul aI (HSMul.hSMul aJ bJ))
        -/
        exact isInteger_smul (hJ bJ hbJ)⟩
        /-
          🎉 no goals
        -/


theorem _root_.IsFractional.inf_right {I : Submodule R P} :
    IsFractional S I → ∀ J, IsFractional S (I ⊓ J)
  | ⟨aI, haI, hI⟩, J =>
    ⟨aI, haI, fun b hb => by
      /-
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        I : Submodule R P
        aI : R
        haI : Membership.mem S aI
        hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
        J : Submodule R P
        b : P
        hb : Membership.mem (Min.min I J) b
        ⊢ IsLocalization.IsInteger R (HSMul.hSMul aI b)
      -/
      rcases mem_inf.mp hb with ⟨hbI, _⟩
      /-
        case intro
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        I : Submodule R P
        aI : R
        haI : Membership.mem S aI
        hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
        J : Submodule R P
        b : P
        hb : Membership.mem (Min.min I J) b
        hbI : Membership.mem I b
        right✝ : Membership.mem J b
        ⊢ IsLocalization.IsInteger R (HSMul.hSMul aI b)
      -/
      exact hI b hbI⟩
      /-
        🎉 no goals
      -/


instance : Min (FractionalIdeal S P) :=
  ⟨fun I J => ⟨I ⊓ J, I.isFractional.inf_right J⟩⟩


@[simp, norm_cast]
theorem coe_inf (I J : FractionalIdeal S P) : ↑(I ⊓ J) = (I ⊓ J : Submodule R P) :=
  rfl


instance : Max (FractionalIdeal S P) :=
  ⟨fun I J => ⟨I ⊔ J, I.isFractional.sup J.isFractional⟩⟩


@[norm_cast]
theorem coe_sup (I J : FractionalIdeal S P) : ↑(I ⊔ J) = (I ⊔ J : Submodule R P) :=
  rfl


instance lattice : Lattice (FractionalIdeal S P) :=
  Function.Injective.lattice _ Subtype.coe_injective coe_sup coe_inf


instance : SemilatticeSup (FractionalIdeal S P) :=
  { FractionalIdeal.lattice with }


instance : Add (FractionalIdeal S P) :=
  ⟨(· ⊔ ·)⟩


@[simp]
theorem sup_eq_add (I J : FractionalIdeal S P) : I ⊔ J = I + J :=
  rfl


@[simp, norm_cast]
theorem coe_add (I J : FractionalIdeal S P) : (↑(I + J) : Submodule R P) = I + J :=
  rfl


theorem mem_add (I J : FractionalIdeal S P) (x : P) :
    x ∈ I + J ↔ ∃ i ∈ I, ∃ j ∈ J, i + j = x := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J : FractionalIdeal S P
    x : P
    ⊢ Iff (Membership.mem (HAdd.hAdd I J) x) (Exists fun i => And (Membership.mem  …
  -/
  rw [← mem_coe, coe_add, Submodule.add_eq_sup]; exact Submodule.mem_sup
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp, norm_cast]
theorem coeIdeal_sup (I J : Ideal R) : ↑(I ⊔ J) = (I + J : FractionalIdeal S P) :=
  coeToSubmodule_injective <| coeSubmodule_sup _ _ _


theorem _root_.IsFractional.nsmul {I : Submodule R P} :
    ∀ n : ℕ, IsFractional S I → IsFractional S (n • I : Submodule R P)
  | 0, _ => by
    /-
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      I : Submodule R P
      x✝ : IsFractional S I
      ⊢ IsFractional S (HSMul.hSMul 0 I)
    -/
    rw [zero_smul]
    /-
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      I : Submodule R P
      x✝ : IsFractional S I
      ⊢ IsFractional S 0
    -/
    convert ((0 : Ideal R) : FractionalIdeal S P).isFractional
    /-
      case h.e'_7
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      I : Submodule R P
      x✝ : IsFractional S I
      ⊢ Eq 0 ↑↑0
    -/
    simp
    /-
      🎉 no goals
    -/
  | n + 1, h => by
    /-
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      I : Submodule R P
      n : Nat
      h : IsFractional S I
      ⊢ IsFractional S (HSMul.hSMul (HAdd.hAdd n 1) I)
    -/
    rw [succ_nsmul]
    /-
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      I : Submodule R P
      n : Nat
      h : IsFractional S I
      ⊢ IsFractional S (HAdd.hAdd (HSMul.hSMul n I) I)
    -/
    exact (IsFractional.nsmul n h).sup h
    /-
      🎉 no goals
    -/


instance : SMul ℕ (FractionalIdeal S P) where smul n I := ⟨n • ↑I, I.isFractional.nsmul n⟩


@[norm_cast]
theorem coe_nsmul (n : ℕ) (I : FractionalIdeal S P) :
    (↑(n • I) : Submodule R P) = n • (I : Submodule R P) :=
  rfl


theorem _root_.IsFractional.mul {I J : Submodule R P} :
    IsFractional S I → IsFractional S J → IsFractional S (I * J : Submodule R P)
  | ⟨aI, haI, hI⟩, ⟨aJ, haJ, hJ⟩ =>
    ⟨aI * aJ, S.mul_mem haI haJ, fun b hb => by
      /-
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        I J : Submodule R P
        aI : R
        haI : Membership.mem S aI
        hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
        aJ : R
        haJ : Membership.mem S aJ
        hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
        b : P
        hb : Membership.mem (HMul.hMul I J) b
        ⊢ IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) b)
      -/
      refine Submodule.mul_induction_on hb ?_ ?_
        /-
          case refine_1
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          b : P
          hb : Membership.mem (HMul.hMul I J) b
          ⊢ ∀ (m : P), Membership.mem I m → ∀ (n : P), Membership.mem J n → IsLocalizati …
        -/
      · intro m hm n hn
        /-
          case refine_1
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          b : P
          hb : Membership.mem (HMul.hMul I J) b
          m : P
          hm : Membership.mem I m
          n : P
          hn : Membership.mem J n
          ⊢ IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) (HMul.hMul m n))
        -/
        obtain ⟨n', hn'⟩ := hJ n hn
        /-
          case refine_1.intro
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          b : P
          hb : Membership.mem (HMul.hMul I J) b
          m : P
          hm : Membership.mem I m
          n : P
          hn : Membership.mem J n
          n' : R
          hn' : Eq ((algebraMap R P) n') (HSMul.hSMul aJ n)
          ⊢ IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) (HMul.hMul m n))
        -/
        rw [mul_smul, mul_comm m, ← smul_mul_assoc, ← hn', ← Algebra.smul_def]
        /-
          case refine_1.intro
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          b : P
          hb : Membership.mem (HMul.hMul I J) b
          m : P
          hm : Membership.mem I m
          n : P
          hn : Membership.mem J n
          n' : R
          hn' : Eq ((algebraMap R P) n') (HSMul.hSMul aJ n)
          ⊢ IsLocalization.IsInteger R (HSMul.hSMul aI (HSMul.hSMul n' m))
        -/
        apply hI
        /-
          case refine_1.intro.a
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          b : P
          hb : Membership.mem (HMul.hMul I J) b
          m : P
          hm : Membership.mem I m
          n : P
          hn : Membership.mem J n
          n' : R
          hn' : Eq ((algebraMap R P) n') (HSMul.hSMul aJ n)
          ⊢ Membership.mem I (HSMul.hSMul n' m)
        -/
        exact Submodule.smul_mem _ _ hm
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          b : P
          hb : Membership.mem (HMul.hMul I J) b
          ⊢ ∀ (x y : P), IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) x) →  …
        -/
      · intro x y hx hy
        /-
          case refine_2
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          b : P
          hb : Membership.mem (HMul.hMul I J) b
          x y : P
          hx : IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) x)
          hy : IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) y)
          ⊢ IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) (HAdd.hAdd x y))
        -/
        rw [smul_add]
        /-
          case refine_2
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          I J : Submodule R P
          aI : R
          haI : Membership.mem S aI
          hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
          aJ : R
          haJ : Membership.mem S aJ
          hJ : ∀ (b : P), Membership.mem J b → IsLocalization.IsInteger R (HSMul.hSMul a …
          b : P
          hb : Membership.mem (HMul.hMul I J) b
          x y : P
          hx : IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) x)
          hy : IsLocalization.IsInteger R (HSMul.hSMul (HMul.hMul aI aJ) y)
          ⊢ IsLocalization.IsInteger R (HAdd.hAdd (HSMul.hSMul (HMul.hMul aI aJ) x) (HSM …
        -/
        apply isInteger_add hx hy⟩
        /-
          🎉 no goals
        -/


theorem _root_.IsFractional.pow {I : Submodule R P} (h : IsFractional S I) :
    ∀ n : ℕ, IsFractional S (I ^ n : Submodule R P)
  | 0 => isFractional_of_le_one _ (pow_zero _).le
  | n + 1 => (pow_succ I n).symm ▸ (IsFractional.pow h n).mul h


/-- `FractionalIdeal.mul` is the product of two fractional ideals,
used to define the `Mul` instance.

This is only an auxiliary definition: the preferred way of writing `I.mul J` is `I * J`.

Elaborated terms involving `FractionalIdeal` tend to grow quite large,
so by making definitions irreducible, we hope to avoid deep unfolds.
-/
irreducible_def mul (lemma := mul_def') (I J : FractionalIdeal S P) : FractionalIdeal S P :=
  ⟨I * J, I.isFractional.mul J.isFractional⟩

-- local attribute [semireducible] mul

instance : Mul (FractionalIdeal S P) :=
  ⟨fun I J => mul I J⟩


@[simp]
theorem mul_eq_mul (I J : FractionalIdeal S P) : mul I J = I * J :=
  rfl


theorem mul_def (I J : FractionalIdeal S P) :
                                                             /-
                                                               R : Type u_1
                                                               inst✝² : CommRing R
                                                               S : Submonoid R
                                                               P : Type u_2
                                                               inst✝¹ : CommRing P
                                                               inst✝ : Algebra R P
                                                               I J : FractionalIdeal S P
                                                               ⊢ Eq (HMul.hMul I J) ⟨HMul.hMul ↑I ↑J, ⋯⟩
                                                             -/
    I * J = ⟨I * J, I.isFractional.mul J.isFractional⟩ := by simp only [← mul_eq_mul, mul_def']
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp, norm_cast]
theorem coe_mul (I J : FractionalIdeal S P) : (↑(I * J) : Submodule R P) = I * J := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J : FractionalIdeal S P
    ⊢ Eq (↑(HMul.hMul I J)) (HMul.hMul ↑I ↑J)
  -/
  simp only [mul_def, coe_mk]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coeIdeal_mul (I J : Ideal R) : (↑(I * J) : FractionalIdeal S P) = I * J := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J : Ideal R
    ⊢ Eq (↑(HMul.hMul I J)) (HMul.hMul ↑I ↑J)
  -/
  simp only [mul_def]
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J : Ideal R
    ⊢ Eq ↑(HMul.hMul I J) ⟨HMul.hMul ↑↑I ↑↑J, ⋯⟩
  -/
  exact coeToSubmodule_injective (coeSubmodule_mul _ _ _)
  /-
    🎉 no goals
  -/


theorem mul_left_mono (I : FractionalIdeal S P) : Monotone (I * ·) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    ⊢ Monotone fun x => HMul.hMul I x
  -/
  intro J J' h
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J J' : FractionalIdeal S P
    h : LE.le J J'
    ⊢ LE.le ((fun x => HMul.hMul I x) J) ((fun x => HMul.hMul I x) J')
  -/
  simp only [mul_def]
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J J' : FractionalIdeal S P
    h : LE.le J J'
    ⊢ LE.le ⟨HMul.hMul ↑I ↑J, ⋯⟩ ⟨HMul.hMul ↑I ↑J', ⋯⟩
  -/
  exact mul_le.mpr fun x hx y hy => mul_mem_mul hx (h hy)
  /-
    🎉 no goals
  -/


theorem mul_right_mono (I : FractionalIdeal S P) : Monotone fun J => J * I := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    ⊢ Monotone fun J => HMul.hMul J I
  -/
  intro J J' h
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J J' : FractionalIdeal S P
    h : LE.le J J'
    ⊢ LE.le ((fun J => HMul.hMul J I) J) ((fun J => HMul.hMul J I) J')
  -/
  simp only [mul_def]
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J J' : FractionalIdeal S P
    h : LE.le J J'
    ⊢ LE.le ⟨HMul.hMul ↑J ↑I, ⋯⟩ ⟨HMul.hMul ↑J' ↑I, ⋯⟩
  -/
  exact mul_le.mpr fun x hx y hy => mul_mem_mul (h hx) hy
  /-
    🎉 no goals
  -/


theorem mul_mem_mul {I J : FractionalIdeal S P} {i j : P} (hi : i ∈ I) (hj : j ∈ J) :
    i * j ∈ I * J := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J : FractionalIdeal S P
    i j : P
    hi : Membership.mem I i
    hj : Membership.mem J j
    ⊢ Membership.mem (HMul.hMul I J) (HMul.hMul i j)
  -/
  simp only [mul_def]
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J : FractionalIdeal S P
    i j : P
    hi : Membership.mem I i
    hj : Membership.mem J j
    ⊢ Membership.mem ⟨HMul.hMul ↑I ↑J, ⋯⟩ (HMul.hMul i j)
  -/
  exact Submodule.mul_mem_mul hi hj
  /-
    🎉 no goals
  -/


theorem mul_le {I J K : FractionalIdeal S P} : I * J ≤ K ↔ ∀ i ∈ I, ∀ j ∈ J, i * j ∈ K := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J K : FractionalIdeal S P
    ⊢ Iff (LE.le (HMul.hMul I J) K) (∀ (i : P), Membership.mem I i → ∀ (j : P), Me …
  -/
  simp only [mul_def]
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J K : FractionalIdeal S P
    ⊢ Iff (LE.le ⟨HMul.hMul ↑I ↑J, ⋯⟩ K) (∀ (i : P), Membership.mem I i → ∀ (j : P …
  -/
  exact Submodule.mul_le
  /-
    🎉 no goals
  -/


instance : Pow (FractionalIdeal S P) ℕ :=
  ⟨fun I n => ⟨(I : Submodule R P) ^ n, I.isFractional.pow n⟩⟩


@[simp, norm_cast]
theorem coe_pow (I : FractionalIdeal S P) (n : ℕ) : ↑(I ^ n) = (I : Submodule R P) ^ n :=
  rfl


@[elab_as_elim]
protected theorem mul_induction_on {I J : FractionalIdeal S P} {C : P → Prop} {r : P}
    (hr : r ∈ I * J) (hm : ∀ i ∈ I, ∀ j ∈ J, C (i * j)) (ha : ∀ x y, C x → C y → C (x + y)) :
    C r := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J : FractionalIdeal S P
    C : P → Prop
    r : P
    hr : Membership.mem (HMul.hMul I J) r
    hm : ∀ (i : P), Membership.mem I i → ∀ (j : P), Membership.mem J j → C (HMul.h …
    ha : ∀ (x y : P), C x → C y → C (HAdd.hAdd x y)
    ⊢ C r
  -/
  simp only [mul_def] at hr
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I J : FractionalIdeal S P
    C : P → Prop
    r : P
    hm : ∀ (i : P), Membership.mem I i → ∀ (j : P), Membership.mem J j → C (HMul.h …
    ha : ∀ (x y : P), C x → C y → C (HAdd.hAdd x y)
    hr : Membership.mem ⟨HMul.hMul ↑I ↑J, ⋯⟩ r
    ⊢ C r
  -/
  exact Submodule.mul_induction_on hr hm ha
  /-
    🎉 no goals
  -/


instance : NatCast (FractionalIdeal S P) :=
  ⟨Nat.unaryCast⟩


theorem coe_natCast (n : ℕ) : ((n : FractionalIdeal S P) : Submodule R P) = n :=
  show ((n.unaryCast : FractionalIdeal S P) : Submodule R P) = n
     /-
       R : Type u_1
       inst✝² : CommRing R
       S : Submonoid R
       P : Type u_2
       inst✝¹ : CommRing P
       inst✝ : Algebra R P
       n : Nat
       ⊢ Eq ↑n.unaryCast ↑n
     -/
                     /-
                       🎉 no goals
                     -/
  by induction n <;> simp [*, Nat.unaryCast]
                     /-
                       🎉 no goals
                     -/


@[deprecated (since := "2024-04-17")]
alias coe_nat_cast := coe_natCast


instance commSemiring : CommSemiring (FractionalIdeal S P) :=
  Function.Injective.commSemiring _ Subtype.coe_injective coe_zero coe_one coe_add coe_mul
    (fun _ _ => coe_nsmul _ _) coe_pow coe_natCast


/-- `FractionalIdeal.coeToSubmodule` as a bundled `RingHom`. -/
@[simps]
def coeSubmoduleHom : FractionalIdeal S P →+* Submodule R P where
  toFun := coeToSubmodule
  map_one' := coe_one
  map_mul' := coe_mul
  map_zero' := coe_zero (S := S)
  map_add' := coe_add


theorem add_le_add_left {I J : FractionalIdeal S P} (hIJ : I ≤ J) (J' : FractionalIdeal S P) :
    J' + I ≤ J' + J :=
  sup_le_sup_left hIJ J'


theorem mul_le_mul_left {I J : FractionalIdeal S P} (hIJ : I ≤ J) (J' : FractionalIdeal S P) :
    J' * I ≤ J' * J :=
  mul_le.mpr fun _ hk _ hj => mul_mem_mul hk (hIJ hj)


theorem le_self_mul_self {I : FractionalIdeal S P} (hI : 1 ≤ I) : I ≤ I * I := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    hI : LE.le 1 I
    ⊢ LE.le I (HMul.hMul I I)
  -/
  convert mul_left_mono I hI
  /-
    case h.e'_3
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    hI : LE.le 1 I
    ⊢ Eq I ((fun x => HMul.hMul I x) 1)
  -/
  exact (mul_one I).symm
  /-
    🎉 no goals
  -/


theorem mul_self_le_self {I : FractionalIdeal S P} (hI : I ≤ 1) : I * I ≤ I := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    hI : LE.le I 1
    ⊢ LE.le (HMul.hMul I I) I
  -/
  convert mul_left_mono I hI
  /-
    case h.e'_4
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    hI : LE.le I 1
    ⊢ Eq I ((fun x => HMul.hMul I x) 1)
  -/
  exact (mul_one I).symm
  /-
    🎉 no goals
  -/


theorem coeIdeal_le_one {I : Ideal R} : (I : FractionalIdeal S P) ≤ 1 := fun _ hx =>
  let ⟨y, _, hy⟩ := (mem_coeIdeal S).mp hx
  (mem_one_iff S).mpr ⟨y, hy⟩


theorem le_one_iff_exists_coeIdeal {J : FractionalIdeal S P} :
    J ≤ (1 : FractionalIdeal S P) ↔ ∃ I : Ideal R, ↑I = J := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    J : FractionalIdeal S P
    ⊢ Iff (LE.le J 1) (Exists fun I => Eq (↑I) J)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      J : FractionalIdeal S P
      ⊢ LE.le J 1 → Exists fun I => Eq (↑I) J
    -/
  · intro hJ
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      J : FractionalIdeal S P
      hJ : LE.le J 1
      ⊢ Exists fun I => Eq (↑I) J
    -/
    refine ⟨⟨⟨⟨{ x : R | algebraMap R P x ∈ J }, ?_⟩, ?_⟩, ?_⟩, ?_⟩
      /-
        case mp.refine_1
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        J : FractionalIdeal S P
        hJ : LE.le J 1
        ⊢ ∀ {a b : R}, Membership.mem (setOf fun x => Membership.mem J ((algebraMap R  …
      -/
    · intro a b ha hb
      /-
        case mp.refine_1
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        J : FractionalIdeal S P
        hJ : LE.le J 1
        a b : R
        ha : Membership.mem (setOf fun x => Membership.mem J ((algebraMap R P) x)) a
        hb : Membership.mem (setOf fun x => Membership.mem J ((algebraMap R P) x)) b
        ⊢ Membership.mem (setOf fun x => Membership.mem J ((algebraMap R P) x)) (HAdd. …
      -/
      rw [mem_setOf, RingHom.map_add]
      /-
        case mp.refine_1
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        J : FractionalIdeal S P
        hJ : LE.le J 1
        a b : R
        ha : Membership.mem (setOf fun x => Membership.mem J ((algebraMap R P) x)) a
        hb : Membership.mem (setOf fun x => Membership.mem J ((algebraMap R P) x)) b
        ⊢ Membership.mem J (HAdd.hAdd ((algebraMap R P) a) ((algebraMap R P) b))
      -/
      exact J.val.add_mem ha hb
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_2
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        J : FractionalIdeal S P
        hJ : LE.le J 1
        ⊢ Membership.mem { carrier := setOf fun x => Membership.mem J ((algebraMap R P …
      -/
    · rw [mem_setOf, RingHom.map_zero]
      /-
        case mp.refine_2
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        J : FractionalIdeal S P
        hJ : LE.le J 1
        ⊢ Membership.mem J 0
      -/
      exact J.zero_mem
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_3
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        J : FractionalIdeal S P
        hJ : LE.le J 1
        ⊢ ∀ (c : R) {x : R}, Membership.mem { carrier := setOf fun x => Membership.mem …
      -/
    · intro c x hx
      /-
        case mp.refine_3
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        J : FractionalIdeal S P
        hJ : LE.le J 1
        c x : R
        hx : Membership.mem { carrier := setOf fun x => Membership.mem J ((algebraMap  …
        ⊢ Membership.mem { carrier := setOf fun x => Membership.mem J ((algebraMap R P …
      -/
      rw [smul_eq_mul, mem_setOf, RingHom.map_mul, ← Algebra.smul_def]
      /-
        case mp.refine_3
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        J : FractionalIdeal S P
        hJ : LE.le J 1
        c x : R
        hx : Membership.mem { carrier := setOf fun x => Membership.mem J ((algebraMap  …
        ⊢ Membership.mem J (HSMul.hSMul c ((algebraMap R P) x))
      -/
      exact J.val.smul_mem c hx
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_4
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        J : FractionalIdeal S P
        hJ : LE.le J 1
        ⊢ Eq (↑{ carrier := setOf fun x => Membership.mem J ((algebraMap R P) x), add_ …
      -/
    · ext x
      /-
        case mp.refine_4.a
        R : Type u_1
        inst✝² : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝¹ : CommRing P
        inst✝ : Algebra R P
        J : FractionalIdeal S P
        hJ : LE.le J 1
        x : P
        ⊢ Iff (Membership.mem (↑{ carrier := setOf fun x => Membership.mem J ((algebra …
      -/
      constructor
        /-
          case mp.refine_4.a.mp
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          J : FractionalIdeal S P
          hJ : LE.le J 1
          x : P
          ⊢ Membership.mem (↑{ carrier := setOf fun x => Membership.mem J ((algebraMap R …
        -/
      · rintro ⟨y, hy, eq_y⟩
        /-
          case mp.refine_4.a.mp.intro.intro
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          J : FractionalIdeal S P
          hJ : LE.le J 1
          x : P
          y : R
          hy : Membership.mem (↑{ carrier := setOf fun x => Membership.mem J ((algebraMa …
          eq_y : Eq ((Algebra.linearMap R P) y) x
          ⊢ Membership.mem J x
        -/
        rwa [← eq_y]
        /-
          🎉 no goals
        -/
        /-
          case mp.refine_4.a.mpr
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          J : FractionalIdeal S P
          hJ : LE.le J 1
          x : P
          ⊢ Membership.mem J x → Membership.mem (↑{ carrier := setOf fun x => Membership …
        -/
      · intro hx
        /-
          case mp.refine_4.a.mpr
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          J : FractionalIdeal S P
          hJ : LE.le J 1
          x : P
          hx : Membership.mem J x
          ⊢ Membership.mem (↑{ carrier := setOf fun x => Membership.mem J ((algebraMap R …
        -/
        obtain ⟨y, rfl⟩ := (mem_one_iff S).mp (hJ hx)
        /-
          case mp.refine_4.a.mpr.intro
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          J : FractionalIdeal S P
          hJ : LE.le J 1
          y : R
          hx : Membership.mem J ((algebraMap R P) y)
          ⊢ Membership.mem (↑{ carrier := setOf fun x => Membership.mem J ((algebraMap R …
        -/
        exact mem_setOf.mpr ⟨y, hx, rfl⟩
        /-
          🎉 no goals
        -/
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      J : FractionalIdeal S P
      ⊢ (Exists fun I => Eq (↑I) J) → LE.le J 1
    -/
  · rintro ⟨I, hI⟩
    /-
      case mpr.intro
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      J : FractionalIdeal S P
      I : Ideal R
      hI : Eq (↑I) J
      ⊢ LE.le J 1
    -/
    rw [← hI]
    /-
      case mpr.intro
      R : Type u_1
      inst✝² : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝¹ : CommRing P
      inst✝ : Algebra R P
      J : FractionalIdeal S P
      I : Ideal R
      hI : Eq (↑I) J
      ⊢ LE.le (↑I) 1
    -/
    apply coeIdeal_le_one
    /-
      🎉 no goals
    -/


@[simp]
theorem one_le {I : FractionalIdeal S P} : 1 ≤ I ↔ (1 : P) ∈ I := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    I : FractionalIdeal S P
    ⊢ Iff (LE.le 1 I) (Membership.mem I 1)
  -/
  rw [← coe_le_coe, coe_one, Submodule.one_le, mem_coe]
  /-
    🎉 no goals
  -/


/-- `coeIdealHom (S : Submonoid R) P` is `(↑) : Ideal R → FractionalIdeal S P` as a ring hom -/
@[simps]
def coeIdealHom : Ideal R →+* FractionalIdeal S P where
  toFun := coeIdeal
  map_add' := coeIdeal_sup
  map_mul' := coeIdeal_mul
                 /-
                   R : Type u_1
                   inst✝² : CommRing R
                   S : Submonoid R
                   P : Type u_2
                   inst✝¹ : CommRing P
                   inst✝ : Algebra R P
                   ⊢ Eq (↑1) 1
                 -/
  map_one' := by rw [Ideal.one_eq_top, coeIdeal_top]
                 /-
                   🎉 no goals
                 -/
  map_zero' := coeIdeal_bot


theorem coeIdeal_pow (I : Ideal R) (n : ℕ) : ↑(I ^ n) = (I : FractionalIdeal S P) ^ n :=
  (coeIdealHom S P).map_pow _ n


theorem coeIdeal_finprod [IsLocalization S P] {α : Sort*} {f : α → Ideal R}
    (hS : S ≤ nonZeroDivisors R) :
    ((∏ᶠ a : α, f a : Ideal R) : FractionalIdeal S P) = ∏ᶠ a : α, (f a : FractionalIdeal S P) :=
  MonoidHom.map_finprod_of_injective (coeIdealHom S P).toMonoidHom (coeIdeal_injective' hS) f


/-- The fractional ideals of a Noetherian ring are finitely generated. -/
lemma fg_of_isNoetherianRing [hR : IsNoetherianRing R] (hS : S ≤ R⁰) (I : FractionalIdeal S P) :
    FG I.coeToSubmodule := by
  /-
    R : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : Nontrivial R
    S : Submonoid R
    P : Type u_4
    inst✝³ : Nontrivial P
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : NoZeroSMulDivisors R P
    hR : IsNoetherianRing R
    hS : LE.le S (nonZeroDivisors R)
    I : FractionalIdeal S P
    ⊢ (↑I).FG
  -/
  have := hR.noetherian I.num
  /-
    R : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : Nontrivial R
    S : Submonoid R
    P : Type u_4
    inst✝³ : Nontrivial P
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : NoZeroSMulDivisors R P
    hR : IsNoetherianRing R
    hS : LE.le S (nonZeroDivisors R)
    I : FractionalIdeal S P
    this : Submodule.FG I.num
    ⊢ (↑I).FG
  -/
  rw [← fg_top] at this ⊢
  /-
    R : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : Nontrivial R
    S : Submonoid R
    P : Type u_4
    inst✝³ : Nontrivial P
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : NoZeroSMulDivisors R P
    hR : IsNoetherianRing R
    hS : LE.le S (nonZeroDivisors R)
    I : FractionalIdeal S P
    this : Top.top.FG
    ⊢ Top.top.FG
  -/
  exact fg_of_linearEquiv (I.equivNum <| coe_ne_zero ⟨(I.den : R), hS (SetLike.coe_mem I.den)⟩) this
  /-
    🎉 no goals
  -/


